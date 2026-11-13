# GPU-run match_analyse — 2026-11-13_160350

- Apparaat: `0` — GPU: NVIDIA GeForce RTX 4070 Laptop GPU (8.0 GB), torch 2.14.1+cu126, CUDA 12.6, ultralytics 8.4.182
- Totale duur: 27.2 min
- Val-imgsz: 960

## ⚠️ LET OP: leakage (lees dit eerst)

> **WAARSCHUWING: 15 van de 15 testbeelden zitten in de 100 Roboflow-frames waarop `best.pt` getraind is** (zelfde framenummer: 15, zelfde bestand: 15).
>
> De cijfers van `best.pt` (en van de fine-tune, die vertrekt van `best.pt`) op deze testset zijn dus **te optimistisch**: het model heeft deze beelden (of hun buren) al gezien. Gebruik ze niet als eerlijke generalisatie. Een eerlijke test vraagt nieuwe, gelabelde frames uit 63–199 en 246–524 (`gpu/export_unseen_frames.py`, zie `gpu/README.md`). De vergelijking COCO ↔ best.pt blijft wel zinvol als illustratie, maar met dit voorbehoud.

- Overlap fine-tune-data (train + val) ↔ testset: **0 / 15** beelden.
- Testset: `C:\Users\brent\3ACS01\Deep Learning\CV-ChallenngeKobe\match_analyse\datasets\notebook_split\data.yaml` (split `test`)

## Stappen

| stap | status | minuten | opmerking |
|---|---|---|---|
| 1. omgeving controleren | ok | 0.039 |  |
| 2. leakage-overzicht | ok | 0.003 |  |
| 3. PR-curves COCO + best.pt | ok | 0.270 |  |
| 4. drempels uit F1-curve | ok | 0.007 |  |
| 5. NMS-sweep + pijplijn | ok | 1.492 |  |
| 6. fine-tune bal | ok | 23.649 |  |
| 7. PR-curves fine-tune | ok | 0.356 |  |
| 8. drempels (met fine-tune) | ok | 0.024 |  |
| 9. pijplijn met fine-tune | ok | 1.356 |  |

## 1. PR-curves en mAP (les 3.8, 5.6, 5.10)

Testset: 15 beelden, 5 ballen, 326 personen. `val()` met conf 0,001 en NMS-IoU 0,7 (ultralytics-standaard).

### Eerlijke vergelijking met COCO: bal + persoon (alle spelers, keeper en scheidsrechter samen)

COCO kent geen teams: `sports ball` → bal, `person` → persoon. Ook onze modellen worden zo samengevoegd (zelfde code voor alle modellen).

| model | imgsz | ball P | ball R | ball AP50 | ball AP50-95 | person P | person R | person AP50 | person AP50-95 |
|---|---|---|---|---|---|---|---|---|---|
| COCO yolo26l | 960 | 1.000 | 0.000 | 0.000 | 0.000 | 0.943 | 0.906 | 0.976 | 0.669 |
| best.pt | 960 | 0.934 | 0.200 | 0.195 | 0.078 | 0.997 | 0.992 | 0.994 | 0.815 |
| ball-finetune | 960 | 0.901 | 0.200 | 0.195 | 0.058 | 0.991 | 0.980 | 0.992 | 0.733 |
| ball-finetune@1280 | 1280 | 0.489 | 0.400 | 0.328 | 0.130 | 0.960 | 0.988 | 0.992 | 0.735 |
| best.pt@1280 | 1280 | 0.854 | 0.200 | 0.195 | 0.117 | 0.988 | 0.975 | 0.991 | 0.783 |

Figuren: `pr/pr_ball_compare.png`, `pr/pr_person_compare.png`. P en R zijn de waarden bij de confidence met de hoogste gemiddelde F1 (ultralytics); bij R = 0 meldt ultralytics P = 1.

### Per klasse (onze modellen)

| model | imgsz | class | instances | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| best.pt | 960 | ball | 5 | 0.866 | 0.200 | 0.195 | 0.078 |
| best.pt | 960 | player_A | 142 | 0.986 | 0.981 | 0.994 | 0.783 |
| best.pt | 960 | player_B | 149 | 0.987 | 0.987 | 0.993 | 0.843 |
| best.pt | 960 | player_B_goal_keeper | 15 | 0.966 | 1.000 | 0.995 | 0.820 |
| best.pt | 960 | referee | 20 | 0.908 | 0.985 | 0.942 | 0.685 |
| best.pt | 960 | all | 331 | 0.943 | 0.831 | 0.824 | 0.642 |
| ball-finetune | 960 | ball | 5 | 0.924 | 0.200 | 0.195 | 0.058 |
| ball-finetune | 960 | player_A | 142 | 0.985 | 0.950 | 0.992 | 0.734 |
| ball-finetune | 960 | player_B | 149 | 0.991 | 0.987 | 0.991 | 0.736 |
| ball-finetune | 960 | player_B_goal_keeper | 15 | 0.982 | 1.000 | 0.995 | 0.713 |
| ball-finetune | 960 | referee | 20 | 0.849 | 0.950 | 0.860 | 0.612 |
| ball-finetune | 960 | all | 331 | 0.946 | 0.817 | 0.807 | 0.571 |
| ball-finetune@1280 | 1280 | ball | 5 | 0.493 | 0.400 | 0.328 | 0.130 |
| ball-finetune@1280 | 1280 | player_A | 142 | 0.966 | 0.972 | 0.991 | 0.754 |
| ball-finetune@1280 | 1280 | player_B | 149 | 0.960 | 0.987 | 0.990 | 0.726 |
| ball-finetune@1280 | 1280 | player_B_goal_keeper | 15 | 0.877 | 1.000 | 0.995 | 0.745 |
| ball-finetune@1280 | 1280 | referee | 20 | 0.800 | 0.950 | 0.933 | 0.697 |
| ball-finetune@1280 | 1280 | all | 331 | 0.819 | 0.862 | 0.848 | 0.611 |
| best.pt@1280 | 1280 | ball | 5 | 0.783 | 0.200 | 0.195 | 0.117 |
| best.pt@1280 | 1280 | player_A | 142 | 0.963 | 0.972 | 0.991 | 0.767 |
| best.pt@1280 | 1280 | player_B | 149 | 0.987 | 0.987 | 0.989 | 0.798 |
| best.pt@1280 | 1280 | player_B_goal_keeper | 15 | 0.928 | 1.000 | 0.995 | 0.784 |
| best.pt@1280 | 1280 | referee | 20 | 0.865 | 0.963 | 0.971 | 0.721 |
| best.pt@1280 | 1280 | all | 331 | 0.905 | 0.824 | 0.828 | 0.637 |

Ultralytics-figuren per model: `pr/<model>_<view>/BoxPR_curve.png`, `BoxF1_curve.png`, `confusion_matrix.png`. Alle cijfers: `pr/metrics_all.csv`.

## 2. Drempelkeuze per klasse uit de F1-curve (les 5.4)

Per klasse de confidence met de hoogste F1 op de testset. De pijplijn cachet YOLO op conf 0,05, lagere drempels werken daar als 0,05.

| model | view | imgsz | class | best_conf | f1 | precision | recall |
|---|---|---|---|---|---|---|---|
| COCO yolo26l | merged | 960 | ball | – | 0.000 | – | – |
| COCO yolo26l | merged | 960 | person | 0.220 | 0.924 | 0.899 | 0.950 |
| best.pt | classes | 960 | ball | 0.468 | 0.331 | 0.967 | 0.200 |
| best.pt | classes | 960 | player_A | 0.573 | 0.985 | 0.992 | 0.979 |
| best.pt | classes | 960 | player_B | 0.726 | 0.990 | 0.993 | 0.987 |
| best.pt | classes | 960 | player_B_goal_keeper | 0.769 | 0.999 | 0.998 | 1.000 |
| best.pt | classes | 960 | referee | 0.348 | 0.951 | 0.907 | 1.000 |
| best.pt | merged | 960 | ball | 0.468 | 0.331 | 0.967 | 0.200 |
| best.pt | merged | 960 | person | 0.395 | 0.995 | 0.996 | 0.994 |
| ball-finetune | classes | 960 | ball | 0.609 | 0.332 | 0.977 | 0.200 |
| ball-finetune | classes | 960 | player_A | 0.421 | 0.978 | 0.985 | 0.972 |
| ball-finetune | classes | 960 | player_B | 0.644 | 0.990 | 0.993 | 0.987 |
| ball-finetune | classes | 960 | player_B_goal_keeper | 0.731 | 0.999 | 0.998 | 1.000 |
| ball-finetune | classes | 960 | referee | 0.581 | 0.901 | 0.857 | 0.950 |
| ball-finetune | merged | 960 | ball | 0.609 | 0.332 | 0.977 | 0.200 |
| ball-finetune | merged | 960 | person | 0.420 | 0.987 | 0.987 | 0.988 |
| ball-finetune@1280 | classes | 1280 | ball | 0.196 | 0.468 | 0.564 | 0.400 |
| ball-finetune@1280 | classes | 1280 | player_A | 0.551 | 0.978 | 0.985 | 0.972 |
| ball-finetune@1280 | classes | 1280 | player_B | 0.618 | 0.990 | 0.993 | 0.987 |
| ball-finetune@1280 | classes | 1280 | player_B_goal_keeper | 0.699 | 0.998 | 0.997 | 1.000 |
| ball-finetune@1280 | classes | 1280 | referee | 0.541 | 0.883 | 0.825 | 0.950 |
| ball-finetune@1280 | merged | 1280 | ball | 0.196 | 0.468 | 0.564 | 0.400 |
| ball-finetune@1280 | merged | 1280 | person | 0.542 | 0.987 | 0.987 | 0.988 |
| best.pt@1280 | classes | 1280 | ball | 0.742 | 0.332 | 0.976 | 0.200 |
| best.pt@1280 | classes | 1280 | player_A | 0.659 | 0.978 | 0.985 | 0.972 |
| best.pt@1280 | classes | 1280 | player_B | 0.720 | 0.990 | 0.993 | 0.987 |
| best.pt@1280 | classes | 1280 | player_B_goal_keeper | 0.760 | 0.997 | 0.993 | 1.000 |
| best.pt@1280 | classes | 1280 | referee | 0.463 | 0.929 | 0.868 | 1.000 |
| best.pt@1280 | merged | 1280 | ball | 0.742 | 0.332 | 0.976 | 0.200 |
| best.pt@1280 | merged | 1280 | person | 0.580 | 0.985 | 0.986 | 0.985 |

Voorstel voor de pijplijn (best.pt): `--ball-conf 0.47`, `--conf 0.40` (start van een persoonstrack). Huidige standaard: 0,1 en 0,3. Met weinig ballen in de testset is de baldrempel statistisch zwak; de Kalman-tracker ruimt losse valse ballen op, dus een iets lagere baldrempel (meer recall) is verdedigbaar.

Figuren: `thresholds/f1_<run>.png`.

## 3. NMS: IoU-sweep en NMS-vrije kop (les 5.4)

YOLO26 heeft twee koppen: standaard draait de one2many-kop **met** NMS (`iou` telt); met `nms=False` draait de NMS-vrije one2one-kop (`iou` doet dan niets).

| setting | iou | mAP50 | mAP50-95 | AP50_ball | AP50_player_A | AP50_player_B | AP50_player_B_goal_keeper | AP50_referee | bestF1_ball | bestF1_player_A | bestF1_player_B | bestF1_player_B_goal_keeper | bestF1_referee |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NMS iou=0.3 | 0.300 | 0.824 | 0.641 | 0.195 | 0.994 | 0.993 | 0.995 | 0.942 | 0.331 | 0.985 | 0.990 | 0.999 | 0.951 |
| NMS iou=0.4 | 0.400 | 0.824 | 0.641 | 0.195 | 0.994 | 0.993 | 0.995 | 0.942 | 0.331 | 0.985 | 0.990 | 0.999 | 0.951 |
| NMS iou=0.5 | 0.500 | 0.824 | 0.641 | 0.195 | 0.994 | 0.993 | 0.995 | 0.942 | 0.331 | 0.985 | 0.990 | 0.999 | 0.951 |
| NMS iou=0.6 | 0.600 | 0.824 | 0.642 | 0.195 | 0.994 | 0.993 | 0.995 | 0.942 | 0.331 | 0.985 | 0.990 | 0.999 | 0.951 |
| NMS iou=0.7 | 0.700 | 0.824 | 0.642 | 0.195 | 0.994 | 0.993 | 0.995 | 0.942 | 0.331 | 0.985 | 0.990 | 0.999 | 0.951 |
| NMS iou=0.8 | 0.800 | 0.823 | 0.644 | 0.195 | 0.994 | 0.992 | 0.995 | 0.942 | 0.331 | 0.985 | 0.990 | 0.999 | 0.951 |
| nms=False (NMS-vrij) | – | 0.825 | 0.642 | 0.195 | 0.994 | 0.992 | 0.995 | 0.949 | 0.332 | 0.982 | 0.990 | 0.998 | 0.914 |

Figuur: `nms/sweep_val.png`.

### Effect op de hele pijplijn, best.pt (volledige clip, alles ook in `pipeline_effect.csv`; GT body = `ground_truth.json`, voet = `ground_truth_feet.json`)

| model | setting | imgsz | frames | seconds | ball_detected | ball_interpolated | ball_none | body_ball_found | body_false_ball | body_players_f1 | body_role_acc | body_possession_acc | feet_ball_found | feet_false_ball | feet_players_f1 | feet_role_acc | feet_possession_acc | possession_A_pct | possession_B_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best.pt | NMS iou=0.7 | 960 | 660 | 25.700 | 332 | 136 | 192 | 13/17 | 4/5 | 0.986 | 0.990 | 0.773 | 3/4 | 1/2 | 0.989 | 0.986 | 0.667 | 100.000 | – |
| best.pt | nms=False (NMS-vrij) | 960 | 660 | 26.100 | 322 | 163 | 175 | 13/17 | 3/5 | 0.977 | 0.971 | 0.682 | 3/4 | 1/2 | 0.993 | 0.979 | 0.667 | 91.071 | 8.929 |
| best.pt | NMS iou=0.5 | 960 | 660 | 25.500 | 332 | 136 | 192 | 13/17 | 4/5 | 0.988 | 0.977 | 0.773 | 3/4 | 1/2 | 0.989 | 0.971 | 0.667 | 100.000 | – |

`ball_found` = bal gevonden op GT-frames met zichtbare bal; `false_ball` = bal gemeld op GT-frames zonder zichtbare bal; `role_acc` = juiste rol (team/keeper/scheidsrechter) van gevonden spelers.

## 4. Fine-tune voor de bal

- Data: `C:\Users\brent\3ACS01\Deep Learning\CV-ChallenngeKobe\match_analyse\datasets\ball_finetune\data.yaml`
- imgsz 1280, batch 4, 16 van max. 60 epochs (patience 15), beste epoch 1, seed 0, deterministic, cos_lr, close_mosaic, cls_pw 0.5 (klassengewicht = 1/frequentie^cls_pw → bal weegt zwaarder)
- Val-split van de fine-tune-data (standaard frames 200–245): mAP50 0.940, mAP50-95 0.468
- Duur: 23.6 min — gewichten: `C:\Users\brent\3ACS01\Deep Learning\CV-ChallenngeKobe\match_analyse\models\best_ball_ft.pt`
- Trainingscurves: `finetune/train_1280_b4/results.png`

- Lekcontrole: 0 / 15 testbeelden in de fine-tune-data.

## Oordeel: is `best_ball_ft.pt` beter dan `best.pt`?

**imgsz 960: NEE**

- bal-AP50 testset: 0.195 -> 0.195 (niet beter)
- pijplijn (body + voeten): bal gevonden 16 -> 16, valse ballen 5 -> 5 (niet beter)

**imgsz 1280: TWIJFEL**

- bal-AP50 testset: 0.195 -> 0.328 (beter)
- pijplijn (body + voeten): bal gevonden 19 -> 19, valse ballen 5 -> 5 (niet beter)

Criteria: hogere bal-AP50 op de testset (per klasse) én in de pijplijn meer ballen gevonden zonder meer valse ballen (`ground_truth.json` + `ground_truth_feet.json` samen, NMS iou 0,7). Let op de leakage hierboven: beide modellen zagen de testframes al tijdens het trainen van `best.pt`; de pijplijncijfers op de hele clip zijn de eerlijkste vergelijking die er nu is (de GT-frames liggen ook buiten de trainingsframes).

## Omgeving

| sleutel | waarde |
|---|---|
| python | 3.13.7 |
| platform | Windows-11-10.0.26200-SP0 |
| cpu_count | 16 |
| device_requested | 0 |
| quick | nee |
| torch | 2.14.1+cu126 |
| cuda_available | ja |
| cuda_version | 12.6 |
| gpu | NVIDIA GeForce RTX 4070 Laptop GPU |
| vram_gb | 8.000 |
| ultralytics | 8.4.182 |
| coco_classes | 80 |
| best_classes | ['ball', 'player_A', 'player_B', 'player_B_goal_keeper', 'referee'] |
| test_images | 15 |
| test_data | C:\Users\brent\3ACS01\Deep Learning\CV-ChallenngeKobe\match_analyse\datasets\notebook_split\data.yaml |
| test_split | test |
| root | C:\Users\brent\3ACS01\Deep Learning\CV-ChallenngeKobe\match_analyse |

## Terugsturen naar Kobe

Zip de hele map `kobe/match_analyse/gpu/results/2026-11-13_160350/` en stuur die zip terug. Daarin zit alles:

- `SUMMARY.md` (dit bestand), `log.txt`, `env.json`, `state.json`, CSV's en figuren (`pr/`, `thresholds/`, `nms/`, `finetune/`, `pipeline_effect.csv`)
- `model/best_ball_ft.pt` — het fine-tuned model
- `cache_export/*.json` — 6 detectiecaches van de hele clip (YOLO-uitvoer per instelling)

### Wat Kobe ermee doet (op de laptop, zonder GPU)

Vanuit `kobe/match_analyse/`:

1. `model/best_ball_ft.pt` kopiëren naar `models/best_ball_ft.pt`.
2. Alle `cache_export/*.json` kopiëren naar `cache/`. De cachesleutel is draagbaar (bestandsnaam, grootte en inhoud-hash van video en model, geen absolute paden of wijzigingstijden), dus de pijplijn gebruikt ze meteen en YOLO hoeft op de CPU niet opnieuw te draaien.
3. `python run_all.py --no-video --weights models/best_ball_ft.pt` (minimap-CSV's, analyses, evaluatie, `output/RAPPORT_best_ball_ft.md`). Met `--imgsz 1280` als de fine-tune op 1280 beter scoorde (zie het oordeel hierboven; die cache zit er ook bij).
4. Voor de NMS-instellingen: de caches van `best.pt` met andere `iou`/`nms` worden alleen gebruikt door code die die instelling vraagt (bv. `core.pipeline.run(..., iou=0.5)`); de standaardinstelling (iou 0,7) gebruikt de cache van `best.pt` met iou 0,7.
