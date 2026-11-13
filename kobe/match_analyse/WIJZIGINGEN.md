# Wijzigingen — match_analyse (2026-11-13)

Branch: `kantoor/match-analyse` (niet gecommit). Plan: `.kantoor/plannen/match-analyse-plan.md`.

## Waarom deze map
Een audit van `kobe/` tegen de cursussamenvatting (`DL_course_summary.md`) liet zien dat de minimap-code drie keer bestond
(`homography/`, `notebooks/minimap/`, `stream-lit/pipeline.py`) met onderling verschillende resultaten, en dat analyses
(les 5.9) en een evaluatie van het hele systeem (les 5.10, stap 4) ontbraken. Training valt buiten scope: modellen komen
van teamgenoten met een GPU.

## Wat er nu is

| Onderdeel | Bestanden | Nieuw tegenover de oude code |
|---|---|---|
| Gedeeld pakket | `core/` | Eén versie van veld, homografie, detectie+cache, teams, tracking, balbezit, rendering; alles importeerbaar |
| Minimap | `run_minimap.py` | CLI met het nieuwe model; schrijft video + `tracks_*.csv` + `frames_*.csv` |
| Homografiecontrole | `check_homography.py` | **Nieuw:** reprojectiefout in meters per keyframe + spreidings-/collineariteitscheck (les 4.6), naast de bestaande driftscore |
| Analyses | `analytics/`, `run_analytics.py` | **Nieuw:** afstand, gemiddelde/max snelheid (km/h) per track, teamsamenvatting, heatmaps per team en bal, gemiddelde posities, balbezitwissels |
| Systeemevaluatie | `evaluation/`, `label_ground_truth.py`, `evaluate_system.py` | **Nieuw:** labeltool voor echte posities/bal/balbezit; positiefout (m), P/R/F1 spelers, teamnauwkeurigheid, balfout, balbezitnauwkeurigheid, ID-switches/IDF1-lite |
| App | `app/app.py` | Streamlit-app op `core/` (zelfde resultaten als de scripts) |
| Tests | `tests/` | 49 pytest-tests met echte asserts (de oude `test_pipeline.py` printte alleen) |
| Data/model | `models/best.pt`, `data/match.mp4`, `data/keypoints.json` | Kopieën, zodat de map los werkt |

## Opgeloste inconsistenties
- **Eén tracker:** overal ultralytics `BYTETracker` (de app gebruikte `supervision`). `track_buffer` schaalt mee met de stride.
- **Bal in meters:** Kalman + interpolatie in meters overal (de app interpoleerde in pixels).
- **Teams:** komen nu uit de modelklassen (`player_A`/`player_B`/keeper/referee) met een meerderheidsstem per track;
  K-means op shirtkleur blijft alleen als terugval voor COCO-modellen.
- **Toeschouwers:** personen buiten het veld (±3 m) worden vóór de tracking weggefilterd, zodat ze geen track-ID krijgen.
- **Paden:** alles relatief aan deze map; geen verwijzing meer naar `kobe/homography`, `stream-lit` of `notebooks`.
  Getest door de map naar een andere plek te kopiëren: alle tests slagen daar.

## Resultaten met het nieuwe model (volledige clip, 660 frames, stride 1)
- Detectie op CPU: ±6,5 min eerste keer, daarna uit de cache; video renderen ±25 s.
- 48 tracks. Persoon-frames: PSG 6702, Inter 6572, Inter-keeper 660, scheidsrechter 1711.
- Bal: gedetecteerd in 302/660 frames, geïnterpoleerd in 119, geen bal in 239.
- Balbezit: PSG 93 % / Inter 7 %, 4 wissels (in 428 frames met balbezit).
- Homografie: 30 van 39 keyframes controleerbaar (≥ 5 punten); RMSE 0,07–0,28 m (mediaan 0,16 m). 9 keyframes hebben
  maar 4 punten: hun fout is altijd 0, dus niet controleerbaar. Driftscore mediaan 0,51; keyframes 150, 420, 441, 458,
  466, 480, 488, 510, 523, 540 scoren laag en zijn het herbekijken waard.
- Analyses: volledige spelers lopen 30–80 m in ±21 s (gemiddeld 5–13 km/h); hoogste piek 36 km/h (track 33, op de rand van
  wat geloofwaardig is).

## Weggelaten of veranderd tegenover de oude app
Tracker aan/uit, keuze balafvlakking, balbevestigingsfilter, K-means-schakelaar, klassenfilter, veldmarge-slider en
weergaveschakelaars (boxes/labels/trails) zijn weg: ze hoorden bij de oude dubbele code. Tracker- en balinstellingen
zijn wel nog instelbaar.

## Aanvulling: systeemevaluatie op de eigen labels (2026-11-13, middag)

De 22 gelabelde frames (`data/ground_truth.json`, elk 30e frame) zijn met de **lichaamsklik** gemaakt: de mediaan van de klikken
ligt op 40 % van de boxhoogte vanaf het hoofd, niet bij de voeten. Een klik boven de voeten ligt niet op het grasvlak, dus
de homografie zet hem 3–4 m te ver. Een vergelijking in meters gaf daardoor schijnbaar slechte cijfers (F1 0,33, teams 53 %).

Aanpassingen:
- `tracks_*.csv` heeft vier extra kolommen achteraan: `box_x1,box_y1,box_x2,box_y2` (pixels). Lezers op kolomnaam blijven werken.
- `evaluate_system.py --anchor body`: koppelt de klik aan het midden van de box in **pixels** (afstand gedeeld door de boxhoogte,
  straal 0,6). Detectie-, team- en trackingcijfers zijn dan geldig; een positiefout in meters wordt niet gerapporteerd.
- In de standaardmodus (`--anchor feet`) meet het rapport waar de klikken in de boxen liggen en waarschuwt als het geen voetklikken zijn.
- 4 nieuwe tests (53 in totaal).

Resultaat met `--anchor body` (nieuw model, 22 frames, 516 gelabelde personen):

| | Resultaat |
|---|---|
| Spelers gevonden | precision 0,977 · recall 0,994 · F1 0,986 (TP 513, FP 12, FN 3) |
| Juiste rol (PSG/Inter/keeper/scheidsrechter) | 99,6 % (2 fouten op 513: één PSG ↔ Inter-verwisseling in elke richting) |
| Bal gevonden (17 frames met bal) | 70,6 %; fout mediaan 0,26 m, p90 0,56 m |
| Bal gemeld zonder zichtbare bal (5 frames) | 3 van 5 (vermoedelijk geïnterpoleerd achter een speler; niet nagekeken) |
| Balbezit juist | 68 % (15/22): alle 22 frames zijn als PSG gelabeld; de pipeline zegt 6× "niemand", 1× Inter |
| ID-switches / IDF1-lite | **niet bruikbaar**: de labels bevatten 66 verschillende ID's voor 21–25 personen per frame. De labeltool neemt een ID over binnen 5 m in het vorige gelabelde frame; met 1 s tussen de frames en lichaamsklikken die meters verschoven zijn, wisselt dat te vaak |

### Voetlabels: positiefout in meters

6 frames (0, 120, 240, 360, 480, 600) opnieuw gelabeld op de voeten in `data/ground_truth_feet.json`
(`evaluate_system.py --gt data/ground_truth_feet.json` → `output/evaluation/system_eval_best_feet.md`):

| | Resultaat |
|---|---|
| Spelers gevonden | F1 0,989 (TP 140, FP 3, FN 0) |
| Juiste rol | 100 % (140/140) |
| Positiefout spelers | gemiddeld 1,13 m, mediaan 1,06 m, p90 1,77 m |
| Bal | 3 van 4 gevonden, fout mediaan 0,14 m; 1 van 2 frames zonder bal toch een bal |
| Balbezit | 4/6 juist (2× "niemand" waar PSG gelabeld is) |

De 1,1 m is vooral een **vaste verschuiving**: de pipeline staat gemiddeld 1,06 m verder in y (dieper in het beeld) en 0,30 m
in x. Zonder die verschuiving is de fout 0,48 m (mediaan 0,42 m). De verklaring: de klikken liggen op 0,83 van de boxhoogte
(kwartielen 0,78–0,86), dus nog iets boven de onderkant van de box, en een paar pixels hoger is ver weg al ±1 m.
De 21 klikken die echt onderaan liggen (≥ 0,9) hebben een fout van 0,45 m, de rest 1,24 m.
**Conclusie:** de echte positiefout van de minimap ligt tussen ±0,5 m en ±1,1 m; 1,1 m is een bovengrens.
De ID-switchcijfers uit deze 6 frames zijn niet bruikbaar (frames 4 s uit elkaar, 75 GT-ID's).

## Wat nog openstaat
1. ~~Ground truth labelen~~ gedaan: 22 frames lichaamsklik + 6 frames voetklik (zie aanvulling hierboven).
2. De laag scorende keyframes opnieuw klikken (zie hierboven).
3. Korte tracks van de scheidsrechter (1–2 s) geven onbetrouwbare snelheden; ID-switches splitsen afstanden.
4. Niet getest met een COCO-model (K-means-terugval alleen op synthetische data getest).
5. Modelevaluatie (COCO-baseline, conf-sweep) en het verslag (ethiek, beperkingen) vielen buiten deze opdracht.
6. De oude mappen (`homography/`, `notebooks/minimap/`, `stream-lit/`) zijn **niet** aangepast of verwijderd.

## Balvoorspelling (2026-11-13)

**Onderzocht** (`experiments/ball_prediction/run_experiments.py`; resultaten en conclusie in
[`experiments/ball_prediction/results.md`](experiments/ball_prediction/results.md)): kan de bal in frames zonder detectie
voorspeld of teruggevonden worden?
- M0: de bestaande tracker (basis). M1: langere interpolatie / extrapolatie. M2: zwakke detecties (conf ≥ 0,05) binnen een poort.
- M3: YOLO opnieuw op een uitsnede (crop) rond de verwachte positie. M4: visuele trackers (template matching, TrackerMIL).
- M5: de poort van de bal-Kalman herstellen. M6: een combinatie.

**Toegevoegd en waarom**
- **Poortfix in `core/tracking.py` (standaard aan).** Het gat bij frames 204–218 was geen gemiste bal maar een weggegooide track:
  na een trap gaat de bal ±1,4 m/frame, de poort groeide maar 1 m per gemist frame. Nu: `BALL_GATE_GROWTH` 1,0 → 1,5,
  `BALL_MAX_MISSED` 15 → 30 en een **herstart**: een zekere kandidaat (≥ 0,4) buiten de poort start de track opnieuw als een
  kandidaat in de volgende 3 frames hem voortzet (≤ 1,5 m/frame). Eén losse uitschieter neemt de track dus niet over. Interpolatie
  blijft beperkt tot `MAX_BALL_GAP` (15 frames): een langere `max_missed` verzint geen ballen. Werkt ook met een stride.
- **Herdetectie in crops, `core/ball_refine.py` (optioneel, standaard uit).** Voor frames zonder bal: crop van 640 px op ware grootte
  rond de verwachte positie (geïnterpoleerd tussen de bekende ballen, hoogstens 60 frames ver), YOLO op `imgsz=320` (zelfde schaal
  als het volle frame; opgeschaalde crops vinden niets). Een treffer telt pas met conf ≥ 0,3 **en** een bevestiging: een
  gedetecteerde bal of een andere crop-treffer (≥ 0,15) binnen 3 frames, op ≤ 1,5 m/frame + 1 m. Zulke ballen krijgen
  `ball_state = refined`; korte gaten (≤ 15 frames) ernaast worden geïnterpoleerd. De ruwe crop-detecties staan in
  `cache/ball_crops_<model>_<hash>.json` (sleutel: video en gewichten incl. wijzigingstijd, cropinstellingen).
  Analyses en evaluatie werken ongewijzigd (`refined` telt als gevonden bal); de video toont "ball: crop".

**Voor / na** (volledige clip, 660 frames; GT = `ground_truth.json` lichaamsklik + `ground_truth_feet.json` voetklik,
gemeten met `evaluate_system.py --anchor body` en `--gt data/ground_truth_feet.json`)

| | Voor | Poortfix (nieuwe standaard) | Poortfix + `--ball-refine` |
|---|---|---|---|
| Bal gedetecteerd / geïnterpoleerd / refined / geen | 302 / 119 / – / 239 | 332 / 136 / – / 192 | 332 / 136 / 1 / 191 |
| Bal gevonden op GT-frames met bal (body 17 + voet 4) | 12 + 3 = 15/21 | 13 + 3 = 16/21 | 14 + 3 = 17/21 |
| Bal gemeld op GT-frames zonder bal (body 5 + voet 2) | 3 + 1 = 4/7 | 4 + 1 = 5/7 | 4 + 1 = 5/7 |
| Balfout body (mediaan / p90, m) | 0,26 / 0,56 | 0,29 / 0,93 | 0,29 / 0,88 |
| Balfout voet (mediaan, m) | 0,14 | 0,14 | 0,14 |
| Balbezit juist (body / voet) | 68 % / 67 % | 73 % / 67 % | 77 % / 67 % |
| Balbezit hele clip | PSG 93 % / Inter 7 %, 4 wissels | PSG 95 % / Inter 5 %, 2 wissels | PSG 95 % / Inter 5 % |

- De extra "valse" bal is **frame 270**: de poortfix volgt nu een speler die met de bal langs de zijlijn dribbelt (detecties 260–262
  en 271–277, op de beelden nagekeken). Frame 270 is geïnterpoleerd tussen 262 en 271; de bal zit daar achter de voeten van die
  speler en is daarom als "niet zichtbaar" gelabeld, maar de positie klopt. Het experiment schreef "geen extra valse ballen" terwijl
  de tabel 5 i.p.v. 4 gaf: de tabel klopt, maar het is geen verzonnen bal.
- De hogere p90-fout komt van GT-frame 330 (ver aan de zijlijn): 2,45 → 3,02 m; de geïnterpoleerde bal ligt daar 1–2 m naast de
  klik (enkele pixels in beeld).
- Dode-balzones (571–601 bal in het net, 607–630 keeper) blijven leeg, ook met `--ball-refine`.
- Herdetectie vindt op deze clip weinig extra, omdat de poortfix de meeste crop-treffers uit het experiment (204–218) al via gewone
  detecties oppikt. Wat overblijft: frame 390 (GT, 0,3 m van de klik). 192 crops kosten op CPU ±20 s, daarna 0 s.
  Het is beter op GT zonder extra valse ballen, maar het verschil is één frame: daarom blijft het een optie en is de
  standaarduitvoer (`output/*_best.*`, analyses, evaluatie) gemaakt **zonder** `--ball-refine`. Aanbevolen voor een eindrapport.

**Niet toegevoegd en waarom**
- Extrapolatie voorbij ±10 frames en interpolatie tot 60 frames (M1): vult veel frames, maar in de dode-balzones ontstaan valse
  ballen (GT-frames 360, 630) en de fout loopt op tot 17–38 px.
- Visuele trackers (M4): de cirkel blijft op gras, lijn of doelnet hangen; meer valse ballen.
- Zwakke detecties met een ruime poort zonder bevestiging (M2): bescheiden winst, valse treffers niet uit te sluiten.
- Crop-treffers onder 0,3 als bal: die pakken vaste plekken op (zijlijn 334–337 in het experiment, nu een plek op (99,5; 24,8) m
  in 347–358 met conf ±0,14).

**Gebruik**
- CLI: `python run_minimap.py --ball-refine` (eerste keer YOLO op de crops, daarna uit de cache).
- Code: `core.pipeline.run(..., ball_refine=True)`.
- App: vink onder "Bal" **Herdetectie in crops** aan. Ontbreken er crops in de cache, dan verschijnt de knop **Herdetecteer bal**;
  tot dan toont de app het resultaat zonder herdetectie. De app start YOLO nooit vanzelf.
- 10 nieuwe tests (63 in totaal): de poortfix volgt een getrapte bal die de oude instellingen kwijtraakten, de herstart vereist
  bevestiging, geen bal in lange gaten, en de bevestigingsregel / gatvulling / cropplaatsing van de herdetectie (zonder model).

## Extra statistieken (2026-11-13)

**Wat:** snelheidszones, sprints, kilometers, aanvalsrichting + aanvallen (laatste derde, 16 m-gebied), pijl op de teamheatmaps,
`run_all.py` (alles in één commando + `output/RAPPORT_<model>.md`) en een tab **Analyses** in de app. `core/` is niet aangepast.

**Code:** `analytics/zones.py` (zones, sprints, km), `analytics/direction.py` (richting, aanvallen), `analytics/extra.py` (alles in één aanroep),
`analytics/report.py` + `analytics/rapport.py` (Nederlandse tekst), `analytics/plots.py` (pijl, zonefiguren). `run_analytics.py`,
`run_minimap.py` en `evaluate_system.py` zijn in functies opgesplitst (`analyze`, `make_minimap`, `evaluate_files`); de CLI's werken ongewijzigd.
13 nieuwe tests in `tests/test_extra.py` (77 in totaal).

**Hoe berekend**
- *Zones:* afgevlakte posities (0,5 s), snelheid per stap over een venster van 0,5 s rond de stap; tijd en afstand per zone:
  stilstaand < 2, wandelen 2-7, joggen 7-15, lopen 15-20, sprint > 20 km/h (bij benadering de gangbare voetbaldrempels, geen officiële standaard).
  Sprint = minstens 1 s aaneengesloten boven 20 km/h, niet over een gat heen.
- *Kilometers:* som over de getrackte spelertracks. Ondergrens (ID-switches splitsen spelers, korte tracks vallen weg). "Per 90 min" is een
  indicatieve opschaling van 22 s.
- *Aanvalsrichting:* gewogen stem van (a) keeperpositie (keeper telt dubbel), (b) gemiddelde x veldspelers van beide teams (zwak), (c) netto
  balrichting bij balbezit; met uitleg welke signalen het eens zijn en een betrouwbaarheid (hoog/middel/laag/onbekend).
- *Aanvallen:* balbezitfase (gaten tot 15 frames zonder bezit toegestaan) waarin de bal het laatste derde (<= 35 m van de aangevallen doellijn) in komt;
  16 m-gebied = bal binnen 16,5 m van de doellijn en 13,8 < y < 54,2 m.

**Beperkingen:** schoten zijn niet geteld (te onbetrouwbaar); balbezit en balpositie bevatten fouten (balbezit-nauwkeurigheid 0,67-0,73 in de evaluatie);
korte clip, dus de aantallen zijn klein; km per 90 min is alleen indicatief.

**Voorbeeldcijfers (match.mp4, model best):** PSG: 5% stilstaand, 47% wandelen, 38% joggen, 8% lopen, 2% sprint, 2 sprints; Inter: 6/40/43/8/3%, 2 sprints.
Km (ondergrens): PSG 0,52 km, Inter 0,58 km. PSG valt aan richting het rechter doel (x=105): Inter-keeper op x=100,9, balrichting +44 m onder PSG-bezit,
betrouwbaarheid hoog (3 van 3). PSG: 3 aanvallen en 2 ingangen van het 16 m-gebied; Inter: 1 balbezitfase, 0 aanvallen.

## GPU-script voor teamgenoot (2026-11-13)

**Waarom:** Kobe heeft geen GPU. PR-curves (COCO ↔ eigen model ↔ fine-tune), drempelkeuze, een NMS-studie en een
bal-fine-tune (les 3.8, 5.4, 5.6, 5.10) moeten dus bij een teamgenoot met een RTX 4060 draaien: één commando, één
resultatenmap terug. Onderzoek: `.kantoor/research/2026-11-13-pr-finetune-nms-extras-research.md`.

**Wat:** map `gpu/` (uitleg voor de teamgenoot: `gpu/README.md`):
- `run_all.py`: alle stappen na elkaar, met tijden en een Nederlandse `results/<tijdstip>/SUMMARY.md`; een mislukte stap
  stopt de rest niet (alleen een kapotte omgeving stopt meteen). `--quick` = rooktest zonder GPU (imgsz 320, 1 epoch, 22 frames).
  **Hervatbaar:** de stand per stap staat in `state.json`; hetzelfde commando opnieuw zet de laatste onafgewerkte run met
  dezelfde instellingen verder en slaat afgewerkte stappen over (`--force` = alles opnieuw, `--new` = nieuwe map); een
  onderbroken fine-tune gaat verder vanaf `last.pt` (ultralytics `resume=True`).
- Stap 9 draait de pijplijn ook met het fine-tuned model; `SUMMARY.md` geeft een **oordeel** "is `best_ball_ft.pt` beter
  dan `best.pt`?" (bal-AP50 op de testset én bal gevonden / valse ballen in de pijplijn) en de exacte terugstuur- en
  overnamestappen. Alle pijplijnrijen: `pipeline_effect.csv`. Het model gaat ook in de resultatenmap (`model/`), de
  detectiecaches in `cache_export/`: de teamgenoot stuurt één zip terug.
- `check_env.py` (torch/CUDA/GPU/VRAM, ultralytics-versie, bestanden → `env.json`), `pr_curves.py`, `thresholds.py`,
  `nms_sweep.py`, `finetune_ball.py`, `export_unseen_frames.py`, `summary.py`, `common.py`, `prepare_datasets.py` (eenmalig).
- `datasets/` (nieuw, zelfstandig): `roboflow_train/` (de 100 frames van `best.pt`, kopie van `data/dataset_yolo`),
  `notebook_split/` (kopie van `kobe/notebooks/dataset`, deelverzameling daarvan) en `ball_finetune/` (zelfde 100 frames per
  **tijdsblok**: train < 200 en 246–599, val 200–245, frames ≥ 600 weggelaten). Data.yaml's zonder `path`: ultralytics zoekt
  vanaf de yaml; de jobs schrijven bovendien een kopie met absolute paden in de resultatenmap.
- `models/yolo26l_coco.pt`: kopie van `kobe/notebooks/yolo26l.pt` (offline COCO-basismodel; anders download).

**Kleine wijziging in `core/`:** `detect()`, `cache_path()` en `pipeline.run()` kregen een parameter `nms=None`
(standaard = huidig gedrag). `nms=False` gebruikt de NMS-vrije one2one-kop van YOLO26. `nms` zit alleen in de cachesleutel
als het niet `None` is. App en CLI's zijn ongewijzigd.

**Draagbare cachesleutel (`core/detection.py`, `core/ball_refine.py`):** de sleutel bevatte absolute paden en
wijzigingstijden, dus een gekopieerde map (pc van de teamgenoot, of terug op de laptop) liet YOLO opnieuw draaien. Nu:
naam + grootte + md5 van de inhoud van video en model (md5 per proces onthouden, ±0,1 s per bestand) + imgsz, iou, nms,
framebereik, stride, conf. Een hertraind `best.pt` (zelfde naam, andere inhoud) krijgt dus nog altijd een nieuwe sleutel.
**Achterwaarts compatibel:** bestaat alleen een cachebestand met de oude sleutel voor dezelfde instellingen, dan wordt dat
gebruikt (bestaande `cache/` draait niet opnieuw); nieuwe caches krijgen de draagbare naam. `cache_path(..., portable=True)`
geeft altijd de draagbare naam (gebruikt voor `cache_export/`). Tests: zelfde bestanden in twee mappen → zelfde sleutel;
oude cache wordt nog gevonden; crop-cache idem.

**Nagekeken in ultralytics 8.4.182 (broncode):**
- Curves: `metrics.box.px` (1000 punten), `box.prec_values` (precisie bij recall = px), `box.f1_curve/p_curve/r_curve`
  (tegen confidence = px), per klasse geordend zoals `box.ap_class_index`; `metrics.curves_results` geeft dezelfde arrays.
- `nms=False` in `val()`/`predict()` zet `end2end` aan (one2one-kop, `iou` doet niets); standaard (`None`) = one2many + NMS.
  Een predictor onthoudt zijn kopmodus, daarom laadt de sweep per instelling een vers model.
- **Klassengewicht bestaat**: `cls_pw` (0..1) → gewicht = (1/frequentie)^cls_pw in de BCE-klassenloss (`detect/train.py`
  `set_class_weights`, `utils/loss.py`). De fine-tune gebruikt `cls_pw 0.5`.
- Ultralytics verkleint zelf de batch bij CUDA OOM in de eerste epoch; het script probeert daarna nog kleinere batch/imgsz.

**Eerlijke vergelijking met COCO:** COCO kent geen teams, dus een "merged"-weergave met 2 klassen (bal, persoon = spelers +
keeper + scheidsrechter). De testlabels worden omgezet; de voorspellingen van **elk** model (COCO én onze) gaan na NMS door
dezelfde `RemapValidator` (subklasse van `DetectionValidator`). Afwijking van de vraag (COCO-id's 0/32 + COCO-namen): zo loopt
elk model door identieke code en hoeft `best.pt` (5 klassen) niet tegen een 80-klassen-yaml (ultralytics zou dan labels
weggooien). Controle op CPU, imgsz 960, notebook-test: best.pt per klasse bal AP50 0,195 / mAP50 0,824 (= researchrapport);
merged: persoon AP50 0,994 (best.pt) vs 0,976 (COCO), bal 0,195 vs 0,000.

**Leakage (beslissing eigenaar: optie B, nu geen nieuwe labels):** de standaardtestset (`notebook_split` test, frames
600–658) zit volledig (15/15, zelfde bestand) in de trainingsdata van `best.pt`. `run_all.py` toont dat in een banner en
bovenaan `SUMMARY.md`. De fine-tune telt de overlap van zijn train/val-beelden met de testset (framenummer of md5) en
**waarschuwt** (banner + aantal in `SUMMARY.md`); alleen met `--strict-leakage` weigert hij.
**Afwijking:** de fine-tune gebruikt standaard `datasets/ball_finetune` (tijdsblokken, 0/15 overlap, echte val-split
200–245 voor early stopping) i.p.v. `datasets/roboflow_train` (bevat de testset volledig en heeft geen aparte val-split:
early stopping en "beste epoch" zouden op de trainingsbeelden zelf gebeuren). `--train-data datasets/roboflow_train/data.yaml`
kan nog altijd. Eerlijke testset: `gpu/export_unseen_frames.py` (63–199,
246–524, elk 5e frame) → labelen in Roboflow → `--test-data` (stappen in `gpu/README.md`).

**Drempels:** per klasse de confidence met de hoogste F1, na afvlakken van de F1-curve over 5 % van de as (de ruwe curve piekt
vaak net voor een klif). Met 5 ballen in de test is de baldrempel statistisch zwak (vermeld in `SUMMARY.md`).

**NMS (CPU-meting, best.pt, imgsz 960, notebook-test):** mAP50 0,824 voor elke IoU 0,3–0,8 en 0,825 met `nms=False`:
op deze testset maakt NMS nauwelijks uit. Het effect op de pijplijn (hele clip) meet de GPU-run.

**Rooktest hier:** `venv\Scripts\python gpu\run_all.py --quick --device cpu` → alle 8 stappen ok in ±1 min (zie het verslag).
14 nieuwe tests in `tests/test_gpu_jobs.py` (omzetting naar bal/persoon, lekcontrole, drempelkeuze, cachesleutel met nms, ...).

**Open:** de echte GPU-run (duur geschat 45–75 min op een RTX 4060) en een ongeziene, gelabelde testset (optie A later:
`export_unseen_frames.py` → Roboflow → `--test-data`).

## Resultaten GPU-run (teamgenoot, 2026-11-13 avond)

Run op een RTX 4070 Laptop GPU (8 GB), 27 min, alle 9 stappen ok. Volledige resultaten: `gpu/results/2026-11-13_160350/`
(`SUMMARY.md`, PR-curves in `pr/`, drempels in `thresholds/`, NMS-sweep in `nms/`, trainingscurves in `finetune/`).
Overgenomen: `models/best_ball_ft.pt` en de 6 detectiecaches (`cache/`), zodat de CPU niets opnieuw hoeft te detecteren.

**PR-curves (testset van 15 frames, 5 ballen; die frames zaten in de training van `best.pt` → te optimistisch):**

| model | bal AP50 | persoon AP50 |
|---|---|---|
| COCO yolo26l (origineel) | 0,000 | 0,976 |
| `best.pt` (fine-tuned door teamgenoot) | 0,195 | 0,994 |
| `best_ball_ft.pt` @ 1280 | 0,328 | 0,992 |

COCO vindt de kleine bal helemaal niet; fine-tunen op eigen data (les 5.10) maakt het verschil. Met 5 ballen is elke bal een
stap van 0,2 recall op de PR-curve, dus 0,195 → 0,328 is in feite **één extra bal**: statistisch zwak.

**NMS (les 5.4):** de IoU-drempel (0,3–0,8) verandert mAP50 niet (0,823–0,824); de NMS-vrije kop (`nms=False`) geeft 0,825.
In de pijplijn: NMS-vrij vult meer frames in (none 175 i.p.v. 192) en geeft minder valse ballen (3/5 i.p.v. 4/5), maar een
lagere balbezitnauwkeurigheid (0,68 i.p.v. 0,77). Standaard blijft daarom NMS met IoU 0,7.

**Fine-tune voor de bal mislukte grotendeels:** beste epoch = **1** van 16 (early stopping). Daarna daalt de val-mAP50 van 0,94 naar
±0,77 en stijgt de val-cls-loss: het al getrainde model "vergeet" (te hoge leersnelheid lr0 = 0,01 voor verder fine-tunen op ±85 frames).
Les 3.4: fine-tunen doe je met een **lagere** leersnelheid en eventueel een bevroren backbone. Een volgende poging: `lr0=0.001`, `freeze=10`.

**De echte winst: inferentie op imgsz 1280 in plaats van 960** (zelf nagemeten op `ground_truth.json` + `ground_truth_feet.json`,
21 frames met zichtbare bal, 7 zonder):

| instelling | bal gevonden | valse bal | frames zonder bal (clip) | balbezit juist (body / voet) |
|---|---|---|---|---|
| `best.pt`, 960, baldrempel 0,1 (oude standaard) | 16/21 | 5/7 | 192 | 0,77 / 0,67 |
| `best.pt`, 1280, baldrempel 0,1 | 19/21 | 5/7 | 130 | 0,77 / 0,67 |
| **`best.pt`, 1280, baldrempel 0,2 (nieuwe standaard)** | **19/21** | **5/7** | **161** | **0,77 / 0,67** |
| `best.pt`, 1280, baldrempel 0,3 | 17/21 | 1/7 | 238 | 0,77 / 0,67 |
| `best_ball_ft.pt`, 1280, baldrempel 0,1 | 19/21 | 5/7 | 137 | 0,77 / 0,67 |

Op 1280 vindt het model de echte bal vaker (visueel gecontroleerd: frames 386–394 en 638–648), maar ook **stilliggende valse ballen**
met lage confidence (frames 366–376, conf 0,11–0,14). Die trekken de interpolatie naar een verkeerde plek. Baldrempel 0,2 haalt die weg
en blijft op elke meting minstens even goed als de oude standaard. Daarom is de standaard nu `imgsz 1280` en `--ball-conf 0.2`
(constanten `DEFAULT_IMGSZ` en `DEFAULT_BALL_CONF` in `core/pipeline.py`; ook in `run_minimap.py`, `run_all.py` en de app).
Baldrempel 0,3 is de "precisie-keuze" (bijna geen valse ballen, minder frames met bal) — de klassieke precision–recall-afweging (les 3.8).

**Bekende fout die blijft:** frames 583–587 (bal ligt in het net) geven een valse bal op (103, 18) m met conf 0,20–0,40; de
interpolatie ervoor en erna (±575–600) ligt daardoor op leeg gras. `best_ball_ft.pt` wordt dus **niet** gebruikt: geen winst tegenover
`best.pt` op 1280.

**Kost op de CPU:** 1280 detecteert ±2× trager dan 960. Voor de volledige clip staat de cache er al (van de GPU-run); voor een
ander framebereik of een andere stride op de laptop: `--imgsz 960` is sneller.
