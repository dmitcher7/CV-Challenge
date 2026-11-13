# GPU-job voor een teamgenoot (RTX 4060)

Eén commando dat op een GPU draait wat op Kobe's laptop (geen GPU) niet lukt, en een resultatenmap die je terugstuurt:

1. **PR-curves** van het COCO-basismodel (yolo26l), ons `best.pt` en een bal-fine-tune (les 3.8 / 5.6 / 5.10)
2. **Drempelkeuze** per klasse uit de F1-curve (les 5.4)
3. **NMS-IoU-sweep** + de NMS-vrije kop van YOLO26, en het effect op de hele pijplijn (bal gevonden, rollen, balbezit)
4. **Fine-tune voor de bal**: `best.pt` verder trainen op imgsz 1280 met klassengewichten

Alles staat in de map `kobe/match_analyse/` (data, modellen, video); er wordt niets buiten die map gelezen.

## Wat je nodig hebt

- Windows met een NVIDIA-GPU (getest gedacht voor RTX 4060, 8 GB) en een recente NVIDIA-driver
- Python 3.10–3.13
- De map `kobe/match_analyse/` **met** de bestanden die niet in git staan (Kobe stuurt ze apart of als zip):
  `models/best.pt`, `models/yolo26l_coco.pt`, `data/match.mp4`
  (`datasets/` en `data/*.json` zitten wel in de map)

## Installatie (eenmalig, ±10 min)

Vanuit de root van de repo (de map waar `kobe/` in staat), in PowerShell:

```powershell
python -m venv venv-gpu
venv-gpu\Scripts\activate
# 1. PyTorch MET CUDA (eerst!). cu126 = dezelfde torch-versie (2.14.x) als waarmee alles getest is, en werkt met een NVIDIA-driver van 2024 of nieuwer; zie https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# 2. de rest (ultralytics staat vast op 8.4.182)
pip install -r kobe/match_analyse/gpu/requirements-gpu.txt
# 3. controle: moet "True" en de naam van je GPU tonen
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Draaien: één commando

```powershell
python kobe/match_analyse/gpu/run_all.py
```

Verwachte duur op een RTX 4060: **±30–60 min** (waarvan de fine-tune ±20–40 min; hij stopt vroeger als hij 15 epochs
niet meer verbetert). De andere stappen (evaluaties en ±6 keer YOLO over de 660 frames van de clip) duren samen
±10–15 min. Je hoeft er niet bij te blijven.

**Onderbroken of een stap mislukt?** Gewoon **hetzelfde commando opnieuw** draaien: het script zet de laatste
onafgewerkte run verder, slaat de stappen over die al klaar waren en een onderbroken fine-tune gaat verder vanaf zijn
laatste epoch. (`--new` = toch een nieuwe map, `--force` = alles opnieuw.)

Eerst testen of alles werkt (±1–3 min, kan ook zonder GPU):

```powershell
python kobe/match_analyse/gpu/run_all.py --quick
```

Opties (`--help` toont alles):

| Optie | Betekenis |
|---|---|
| `--quick` | rooktest: imgsz 320, 1 epoch, ¼ van de trainingsbeelden, 2 IoU-waarden, 22 frames van de clip (elk 30e = de GT-frames); de getallen betekenen niets |
| `--device 0` / `--device cpu` | GPU 0 (standaard als CUDA er is) of de processor |
| `--test-data <data.yaml> --test-split test` | andere testset (zie "Eerlijke testset") |
| `--train-data <data.yaml>` | andere data voor de fine-tune (standaard `datasets/ball_finetune`) |
| `--imgsz 1280 --epochs 60 --batch 4 --cls-pw 0.5` | instellingen van de fine-tune |
| `--skip-finetune` | geen fine-tune (gebruikt `models/best_ball_ft.pt` als die er al is) |
| `--strict-leakage` | fine-tune weigeren als train- en testbeelden overlappen (standaard: alleen een luide waarschuwing + aantal in `SUMMARY.md`) |
| `--new` / `--force` / `--out <map>` | nieuwe resultatenmap / alle stappen opnieuw / een bepaalde map verderzetten |

Losse stappen kunnen ook apart: `check_env.py`, `pr_curves.py`, `nms_sweep.py`, `finetune_ball.py`
(elk met `--help`), `thresholds.py --out <resultatenmap>`.

## Wat stuur je terug?

Eén zip: de map **`kobe/match_analyse/gpu/results/<datum_tijd>/`** (±100 MB). Daarin zitten ook het fine-tuned model
(`model/best_ball_ft.pt`) en de detectiecaches (`cache_export/*.json`). Het script print het pad op het einde; onderaan
`SUMMARY.md` staat wat Kobe ermee doet. (Niet de `_quick`-map van de rooktest.)

Lukt een stap niet, dan lopen de andere stappen gewoon verder; de fout staat in `SUMMARY.md` (sectie "Fouten") en
in `log.txt`. Stuur de map dan ook terug.

## Wat staat er in de resultatenmap?

| Bestand | Inhoud |
|---|---|
| `SUMMARY.md` | alles samengevat in tabellen (Nederlands): leakage, mAP, drempels, NMS, fine-tune, omgeving, duur |
| `pr/pr_ball_compare.png`, `pr/pr_person_compare.png` | PR-curves van alle modellen over elkaar, AP50 in de legende |
| `pr/metrics_all.csv`, `pr/metrics_<model>_<view>.csv` | P, R, mAP50, mAP50-95 per klasse |
| `pr/<model>_<view>/` | figuren van ultralytics: `BoxPR_curve.png`, `BoxF1_curve.png`, confusion matrix, ... |
| `thresholds/thresholds.csv`, `thresholds/f1_*.png` | beste confidence per klasse (max F1) |
| `nms/sweep_val.csv`, `nms/sweep_val.png` | mAP en F1 per NMS-IoU en voor `nms=False` |
| `pipeline_effect.csv` | pijplijn op de hele clip per model/instelling: bal gevonden, valse ballen, rollen, balbezit |
| `nms/pipeline_*/` | `tracks.csv` + `frames.csv` van die pijplijnruns |
| `cache_export/*.json` | YOLO-detecties van de hele clip per model/instelling (draagbare cachesleutel: in `cache/` van een andere pc gezet, worden ze meteen hergebruikt) |
| `finetune/train_*/` | trainingsrun (`results.png`, `results.csv`, ...) |
| `model/best_ball_ft.pt` | het fine-tuned model (kopie van `models/best_ball_ft.pt`) |
| `env.json`, `log.txt`, `state.json` | omgeving, volledige uitvoer, stand van de stappen (voor verderzetten) |

## Eerlijke testset (belangrijk)

`best.pt` is getraind op 100 Roboflow-frames: 0–62, 200–245 en 525–658. De standaardtestset
(`datasets/notebook_split`, split `test` = frames 600–658) zit daar **volledig in** → de testcijfers van `best.pt` zijn
te optimistisch (leakage). Het script meldt dat luid en zet het aantal overlappende beelden in `SUMMARY.md`.

Een eerlijke testset maken (door Kobe of jou, ±1–2 u labelen):

1. Frames exporteren die het model nooit zag (63–199 en 246–524, elk 5e frame):
   `python kobe/match_analyse/gpu/export_unseen_frames.py` → `gpu/unseen_frames/frame_0065.jpg`, ...
   (`--every 10` voor minder frames, `--ranges 63-199,334-437` voor andere stukken)
2. Uploaden naar Roboflow, boxen tekenen met **dezelfde 5 klassen in dezelfde volgorde**:
   `ball, player_A, player_B, player_B_goal_keeper, referee`. Alle beelden in de split **test** zetten.
   Bestandsnamen niet wijzigen (de lekcontrole herkent `frame_<nummer>`).
3. Exporteren als **YOLO (YOLOv8/YOLO26-formaat)**, uitpakken naar `kobe/match_analyse/datasets/unseen_test/`
   (zodat daar `data.yaml` en `test/images`, `test/labels` staan).
4. Draaien met: `python kobe/match_analyse/gpu/run_all.py --test-data kobe/match_analyse/datasets/unseen_test/data.yaml --test-split test`

## De fine-tune

- Start van `models/best.pt`, data `datasets/ball_finetune/` = de 100 Roboflow-frames per **tijdsblok** gesplitst:
  train = frames < 200 en 246–599 (59 beelden), val = 200–245 (26 beelden); frames ≥ 600 (de testset) zitten er niet in.
- imgsz 1280 (de bal is maar 5–10 px op 960), max. 60 epochs, patience 15, seed 0, deterministic, cos_lr, close_mosaic 10.
- **Klassengewicht**: ultralytics 8.4.182 heeft `cls_pw` — gewicht per klasse = (1 / frequentie)^cls_pw in de
  classificatieverlies. Met 0,5 weegt de zeldzame bal (±1 per beeld tegen ±20 personen) een paar keer zwaarder.
- Lekcontrole: telt de testbeelden die ook in de train/val-data zitten (zelfde framenummer of zelfde bestand), meldt
  dat luid en zet het aantal in `SUMMARY.md`. Met de standaarddata is dat 0. Alleen met `--strict-leakage` weigert hij
  te trainen bij overlap (bv. met `--train-data datasets/roboflow_train/data.yaml`, die de testset volledig bevat).
- Let op: ook de fine-tune vertrekt van `best.pt`, dat de testframes 600–658 al zag. Alleen met een nieuwe, ongeziene
  testset zijn de cijfers eerlijk.

## Problemen oplossen

| Probleem | Oplossing |
|---|---|
| `PyTorch ziet geen CUDA-GPU` / `torch.cuda.is_available()` is `False` | De CPU-versie van torch staat erop. `pip uninstall -y torch torchvision` en opnieuw installeren met de `--index-url .../cu126`-regel hierboven. Driver bijwerken (GeForce Experience / nvidia.com). `nvidia-smi` moet je GPU tonen. |
| `CUDA out of memory` | Doet het script zelf: ultralytics halveert de batch, daarna probeert het script kleinere batch/imgsz (1280→1024→960). Sluit andere programma's die de GPU gebruiken (games, browsers met hardwareversnelling). Handmatig: `--batch 2` of `--imgsz 1024`. |
| `ultralytics x gevonden, verwacht 8.4.182` | `pip install ultralytics==8.4.182` (andere versies laden `best.pt` mogelijk anders). |
| `Bestand ontbreekt (...)` | `models/best.pt`, `models/yolo26l_coco.pt` of `data/match.mp4` zit niet in de map: vraag ze aan Kobe. Zonder `yolo26l_coco.pt` downloadt ultralytics `yolo26l.pt` (internet nodig). |
| `GEWEIGERD (--strict-leakage): ...` | Train- en testset overlappen en je vroeg `--strict-leakage`. Kies een andere `--test-data` of `--train-data`, of laat de optie weg. |
| Commando vanuit een andere map | Kan: alle paden worden berekend vanaf de scriptmap. Relatieve `--test-data`/`--train-data` gelden vanaf de map waarin je het commando typt. |
| Het script hangt bij `Scanning ...` / dataloader op Windows | Opnieuw starten; zo nodig in een gewone PowerShell (niet in een notebook). |
| Rode tracebacks `ValueError: x1 must be greater than or equal to x0` | Komt uit een tekenthread van ultralytics bij een model dat nog niets kan; bij `--quick` daarom uitgeschakeld. Onschuldig, de run gaat door. |
| Eerste run downloadt iets (`yolo26n.pt`, `Arial.ttf`) | Normaal: ultralytics doet een AMP-controle en heeft een lettertype voor de figuren nodig. Zonder internet slaat hij de AMP-controle over. |
