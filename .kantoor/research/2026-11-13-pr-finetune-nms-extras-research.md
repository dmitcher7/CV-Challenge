# Onderzoek: PR-curves origineel vs fine-tuned, IoU/NMS-drempels en extra analyses

Datum: 2026-11-13. Rol: researcher. Er is niets gebouwd, getraind of aangepast; alleen gelezen en kleine metingen gedaan met `venv\Scripts\python.exe`.

## Samenvatting
- `best.pt` is getraind op `data/dataset_yolo/` (Roboflow-export; lokaal alleen een `train/`-map met 100 beelden). Die beelden dekken frames 1-62, 200-245 en 525-658 van de 660-frame clip. Er is dus waarschijnlijk **leakage** met de notebook-splits (val, test). Een eerlijke testset bestaat nu niet.
- YOLO26 in ultralytics 8.4.182 is **niet standaard NMS-vrij**: het model heeft twee koppen (one2many en one2one). Standaard (`nms=None`) draait de one2many-kop met externe NMS en heeft `iou` effect. Met `nms=False` draait de NMS-vrije one2one-kop en doet `iou` niets.
- CPU is genoeg om te evalueren (94 beelden `val()` is circa 1 min), niet om YOLO26l te finetunen (uren). Training hoort op Kaggle/Colab of bij een teamgenoot.

## 1. Wat is `best.pt` en wat is held-out?

Checkpoint `train_args` (gelezen):
- data: `C:\school\AI\Deep Learning\ComputerVision\CV-Challenge\data\dataset_yolo\data.yaml` (pad van de teamgenoot), model `yolo26l.pt`, `imgsz 960`, `epochs 100`, `batch 2`, `mosaic 0.0`, `cls 0.5`, `box 7.5`, `iou 0.7`.
- `train_metrics` (eigen val-set van de teamgenoot): P 0,889 / R 0,897 / mAP50 0,895 / mAP50-95 0,546.

Lokale data:
- `data/dataset_yolo/data.yaml` verwijst naar `../train`, `../valid`, `../test`, maar lokaal bestaat alleen `train/` met **100 beelden**: frames 1-31, 34, 35, 38, 39, 42, 46, 50, 54, 58, 62, circa 30 frames in 200-245 en circa 30 in 525-658. Geen enkel frame in 63-199 of 246-524.
- `kobe/notebooks/dataset/` (94 beelden) is een deelverzameling van dezelfde frames: train 0-62 + 525-588, val 200-245 (de notebook zegt 150-299, maar alleen 200-245 is gelabeld), test 600-658.
- Onbekend: hoe Roboflow de 100 beelden over train/valid/test verdeeld heeft. Het is mogelijk dat een deel van de 100 beelden de valid/test van de teamgenoot was (dan waren die voor hem held-out), maar dat staat nergens lokaal. Vraag het hem.
- Meting: `val()` van `best.pt` op `notebooks/dataset` split `test` (15 beelden, 331 objecten): mAP50 0,824; bal AP50 **0,195** (5 ballen, recall 0,2); spelers 0,99. Spelers bijna perfect (makkelijk of gezien); de bal is op deze frames moeilijk. Dit getal zegt niets over generalisatie zolang leakage niet is uitgesloten.

Conclusie: behandel alle 100 gelabelde beelden als "gezien". Een eerlijke test moet **nieuwe** frames gebruiken. Er is maar één video (`data/raw_video/match.mp4` is dezelfde clip: 660 frames, 1920x1080, 30 fps).

### Voorstel voor een eerlijke testset
Alle frames in 63-199 en 246-524 zijn zeker ongezien.
1. **Eigen ground truth (geen extra labelwerk):** `data/ground_truth.json` heeft frames 0, 30, ..., 630. Ongezien (minstens 10 frames van een trainingsframe): 90, 120, 150, 180, 270, 300, 330, 360, 390, 420, 450, 480, 510 (13 frames). Maar het zijn **klikken**, geen boxen. Voor de bal kan een klik-in-box-test; voor spelers is geen mAP mogelijk (wel de bestaande `evaluate_system.py`).
2. **Nieuwe boxlabels (aanbevolen):** label circa 40-60 frames uit 63-199 en 246-524, elk 5e-10e frame, met nadruk op de balgaten (241-276, 334-437). Circa 1-2 min per frame; 50 frames is circa 1-1,5 uur. Splits in **tijdsblokken** (bijv. 334-437 volledig buiten training), want opeenvolgende frames lijken sterk op elkaar.
3. Formuleer voorzichtig: één clip, één scène; het meet generalisatie binnen dezelfde clip, niet naar andere wedstrijden.

## 2. PR-curve: origineel (COCO) vs fine-tuned

### Wat te doen
1. Testset (zie boven) in YOLO-formaat met `data.yaml`.
2. Fine-tuned: `model.val(data=..., split=..., imgsz=960, conf=0.001, iou=0.7, plots=True, device='cpu')` geeft `BoxPR_curve.png`, `BoxF1_curve.png` en per-klasse AP50; `box.p_curve/r_curve/f1_curve` voor een eigen bal-plot.
3. COCO-baseline (`kobe/notebooks/yolo26l.pt`, ook `yolo26m/s` in `kobe/homography/pretrained/`): de klassen moeten gemapt worden: `sports ball` (COCO 32) naar ball, `person` (COCO 0) naar een gezamenlijke klasse "speler" (spelers + keeper + scheidsrechter samen; COCO kan geen teams scheiden, dat is juist het verhaal van les 5.10). Goedkoopste route: de GT-labels herschrijven naar een tweede dataset met klasse-indices 0 en 32 en gewoon `val()` draaien (circa 30 regels script). Alternatief: eigen predict + `ultralytics.utils.metrics.ap_per_class`.
4. Plot per klasse, met nadruk op de bal: AP50, recall bij de gekozen conf, PR-curve van `ball` voor beide modellen.

### Verwachting
COCO `sports ball` mist waarschijnlijk veel kleine ballen (5-10 px op 1920x1080). Fine-tuned is beter, maar de bal-AP op een leakagevrije set kan laag blijven (zie 0,195). Dat is een eerlijk resultaat.

### Kosten (CPU Kobe)
| Taak | Uren | CPU-minuten |
|---|---|---|
| Testframes labelen (50-60) | 1-2 | n.v.t. |
| `val()` fine-tuned op 94 beelden (gemeten: 15 beelden in 10 s, 0,64 s/beeld) | 0,5 | circa 1-2 |
| COCO-baseline `val()` + mapping-script | 1-2 | circa 1-2 |
| Plots + tabel (hergebruik `kobe/training/evaluate.py`, `error_analysis.py`) | 1-2 | circa 5 |

Moeilijkheid: laag/middel (COCO-mapping is het vervelendste).

### Fine-tunen specifiek voor de bal
- **Data:** meer balvoorbeelden, vooral moeilijke frames uit de gaten 241-276, 334-437, 607-659 (bal van circa 5 px tegen keeper/net). Streef naar 100-200 balvoorbeelden; één clip beperkt diversiteit sterk.
- **Hogere resolutie:** imgsz 1280-1600 (`brent_v2` gebruikte 1600). Op CPU nauwelijks haalbaar.
- **Class weights:** ultralytics heeft geen per-klasse gewichten. Alternatieven: balframes oversamplen (kopieen in train), `cls`/`box` aanpassen, of een aparte 1-klasse baldetector.
- **Tiling/SAHI:** inferentie op tegels (bv. 2x2) maakt de bal relatief groter. `sahi` werkt met ultralytics-modellen. Kost circa 4-6x inferentietijd (circa 2-3 s per frame, 660 frames circa 25-30 min per run op CPU) maar **geen training**, dus goedkope eerste stap. Trainen op tegels vraagt een tegel-dataset (extra script).
- **Training:** op een T4 (Kaggle/Colab) yolo26l, 960, 100 epochs op circa 150 beelden: schatting 20-40 min (batch 2-4). Op CPU: circa 2 s/beeld (forward 0,65 s, training circa 3x), circa 4-5 min per epoch, **100 epochs circa 7-8 uur**: niet realistisch. yolo26s/n is 5-10x sneller maar ziet de kleine bal slechter.
- Daarna `val()` op dezelfde held-out set en PR-curve van het nieuwe model naast de andere twee.

Aanbevolen volgorde: eerst SAHI + hogere imgsz (geen training), dan pas een echte finetune.

### Kaggle/Colab
Realistisch: Kaggle circa 30 GPU-uur per week (T4/P100), Colab gratis T4 met sessielimiet. Dataset (100-200 jpg) uploaden en een notebook draaien (basis: `kobe/notebooks/02_model.ipynb`, `kobe/training/train.py`). Risico's: telefoonverificatie, sessies die wegvallen, andere ultralytics-versie (zet `ultralytics==8.4.182` vast, anders laadt `best.pt` mogelijk anders). Moeilijkheid middel. Alternatief: teamgenoot met GPU (minder werk, wel afhankelijk). De bestaande afspraak is dat teamgenoten de `.pt` trainen.

### Risico's
- **Leakage**: rapporteer altijd welke beelden gezien zijn.
- Weinig ballen in test: AP50 van de bal springt bij een frame sterk; geef TP/FP/FN naast AP.
- Roboflow "version 2" kan augmentaties bevatten: kopieen van hetzelfde frame in train en valid zijn mogelijk.
- `val()` gebruikt `conf=0.001`; de pipeline cachet op 0,05. Bereken PR/F1 met lage conf en kies daarna de drempel.

## 3. IoU en NMS correct toepassen en drempels kiezen

### Hoe werkt het in ultralytics 8.4.182 voor YOLO26 (broncode gelezen, gemeten)
- Het checkpoint heeft `model.end2end = False` en de Detect-kop bevat **beide** takken (`one2one_cv2` bestaat). Standaard (`nms=None`) gebruiken `predict` en `val` de one2many-kop met **externe NMS**: `iou` heeft effect.
- Gemeten op een frame, conf 0,05: frame 300: iou 0,3 geeft 25 boxen, 0,5 geeft 25, 0,7 geeft 28, 0,9 geeft 64 (dubbele boxen). Frame 420: constant 30. Met `nms=False`: 31 en 30 boxen, onafhankelijk van `iou` (NMS-vrije one2one-kop; `predictor.py` regel 476 `end2end=self.args.nms is False`).
- Conclusie: de pipeline (`core/detection.py`: `model.predict(..., conf=DETECT_CONF, iou=iou)`, `iou=0.7`) gebruikt klassieke NMS. "NMS-vrij" geldt alleen met `nms=False`. De cache-key bevat `iou` maar niet `nms`: voeg die toe als je beide modi vergelijkt.
- `val(nms=False)` werkt ook (`validator.py` regel 167), dus de varianten zijn met mAP vergelijkbaar.

### Ontwerp van de drempelstudie
1. **Conf per klasse:** `val(conf=0.001)` geeft de F1-curve per klasse (`box.f1_curve`, `box.px`); kies de conf met maximale F1, bal apart (liever hogere recall voor de bal; Kalman/tracking ruimt valse ballen op). Les 3.8 en 5.4.
2. **NMS-IoU-sweep:** `val(iou=x)` voor x in {0,3; 0,4; 0,5; 0,6; 0,7; 0,8} plus `nms=False`; rapporteer mAP50, F1 en aantal boxen. Circa 1 min per punt, circa 10 min totaal.
3. **Effect op de pipeline:** een andere `iou` betekent **opnieuw detecteren** (cache-key bevat iou): gemeten 381 s voor 660 frames (0,58 s/frame). Per waarde circa 6,5 min; 4-5 waarden circa 30 min. Goedkoper: stride 2 (circa 3 min) of alleen de 22 GT-frames (circa 15 s per waarde). Daarna `evaluate_system.py --anchor body` per waarde (spelers-F1, juiste rol, bal gevonden, valse ballen).
4. **Conf-sweep in de pipeline is gratis**: de cache staat op 0,05; `run(conf=..., ball_conf=...)` filtert achteraf (`pipeline.py`). Sweep conf 0,1-0,6 en ball_conf 0,05-0,4.
5. Resultaat: tabel plus F1-vs-conf en F1-vs-iou-grafiek.

### Waar komen de drempels
- `core/pipeline.py` `run(..., iou=0.7, conf=0.3, ball_conf=0.1)`; `core/detection.py` `detect(..., iou=0.7)` en `DETECT_CONF = 0.05`; CLI `--conf --ball-conf` in `run_minimap.py`; slider "NMS IoU" in `app/app.py`. Een `--iou/--nms`-optie is nodig om `nms=False` te testen.

### Kosten
3-5 uur; CPU circa 10 min (val-sweep) plus 5-30 min (pipeline-sweep). Moeilijkheid laag/middel. Risico: met 5 balbeelden is de keuze statistisch zwak; gebruik de 13 ongeziene GT-frames en nieuwe labels.

## 4. Extra's: wat bestaat al, wat ontbreekt

Bestaat (`analytics/`, `run_analytics.py`): afstand en gemiddelde/max snelheid per track met afvlakking en 36 km/h-cap (`stats.py`), teamsamenvatting, balbezit en wissels, heatmaps per team en bal, gemiddelde posities (`plots.py`), markdown-rapport. Streamlit-app met sliders.

| Extra | Wat ontbreekt | Uren | Betrouwbaarheid (22 s, homografieruis) | Moeilijkheid |
|---|---|---|---|---|
| Stilstaand vs bewegend, snelheidszones (wandelen/joggen/rennen/sprint) | Snelheid per frame bestaat in `track_stats`; zones en tijd per zone ontbreken | 2-3 | Redelijk voor grove zones; stilstaan gevoelig voor ruis (afvlakken helpt); percentages indicatief | laag |
| Kilometers per speler/team | Afstand per track bestaat (30-80 m in circa 21 s); alleen optellen per team. ID-switches geven lagere totalen; extrapolatie naar km is misleidend | 1 | Matig | laag |
| Welk doel valt elk team aan | Gemiddelde x per team (bestaat), keeperzijde en baltrajectrichting; regel: een team valt de kant van de andere keeper aan | 2 | Goed als de keeper in beeld is | laag |
| Schoten/aanvallen richting doel | Baltrajectorie (Kalman in `core/tracking.py`) plus richting/snelheid naar de doellijn en zone. Bal gevonden in 71-75 % | 6-10 | Laag-middel: weinig balframes, bal in de lucht niet via homografie; liever "aanvalszone-tijd" | hoog |
| Heatmaps verbeteren | Bestaan al; per speler/rol, vaste kleurschaal, tijdvenster, veld-overlay | 2-4 | Visueel goed, niet kwantitatief | laag |
| Automatisering (een commando detectie, tracking, analytics, evaluatie, rapport; optioneel Streamlit-tab) | `run_minimap.py`, `run_analytics.py`, `evaluate_system.py` bestaan los; wrapper `run_all.py` en een tab in `app/app.py` | 3-5 | Betrouwbaar (alleen orkestratie) | laag/middel |

## 5. Geprioriteerde tabel

| Item | moet/extra | uren | rekenkost | moeilijkheid | afhankelijkheden | advies |
|---|---|---|---|---|---|---|
| Eerlijke testset (nieuwe frames 63-199, 246-524; 13 ongeziene GT-frames) | moet | 1-2 | geen | laag | eigenaar labelt | eerst doen; zonder dit zijn PR-curves niet eerlijk |
| PR-curve fine-tuned vs COCO (bal + spelers) | moet | 3-5 | circa 5 CPU-min | middel | testset | doen op CPU |
| Conf per klasse uit F1/PR (bal apart) | moet | 1-2 | circa 2 CPU-min | laag | testset | doen |
| NMS-IoU-sweep (+ `nms=False`) en pipeline-effect | moet | 3-5 | circa 10 plus 5-30 CPU-min | laag/middel | testset | doen |
| Zones en stilstaand | extra | 2-3 | verwaarloosbaar | laag | `stats.py` | doen |
| Km per speler/team | extra | 1 | verwaarloosbaar | laag | `stats.py` | doen met voorbehoud |
| Doelkant per team | extra | 2 | verwaarloosbaar | laag | gemiddelde posities, keeper | doen |
| Automatisering (`run_all.py`, Streamlit-tab) | extra | 3-5 | geen | laag/middel | bestaande scripts | na de moet-items |
| SAHI/tiling voor de bal | extra | 4-6 | circa 25-30 CPU-min per run | middel | testset | goedkoop alternatief voor finetune |
| Fine-tune bal (nieuwe labels, imgsz 1280+) | extra | 6-10 | 0 CPU; circa 30-60 GPU-min | middel/hoog | labels; Kaggle of teamgenoot | alleen bij tijd |
| Heatmaps verbeteren | extra | 2-4 | laag | laag | `plots.py` | lage prioriteit |
| Schoten/aanvallen richting doel | extra | 6-10 | laag | hoog | balbereik, balgaten | overslaan of "aanvalszone" |

## Vragen voor de eigenaar
1. Mag ik de teamgenoot vragen hoe Roboflow de 100 beelden over train/valid/test verdeeld heeft en of er een afgesplitste testset bestaat? (Bepaalt of er al een held-out set is.)
2. Wil je zelf circa 50 extra frames boxlabelen (vooral bal in 241-276, 334-437, 607-659) als testset buiten de training?
3. Fine-tunen: via Kaggle/Colab zelf (meer werk, circa 30-60 GPU-min per run) of door een teamgenoot met GPU? Is een nieuwe finetune gewenst gezien de afspraak dat teamgenoten trainen?
4. Is SAHI/tiling (geen training) een geaccepteerde verbetering, of moet het een nieuw model zijn?
5. Welke extra's wil je: zones, km, doelkant en automatisering, of ook schoten (hoog risico)?
