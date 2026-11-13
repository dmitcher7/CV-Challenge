# match_analyse — van video tot minimap en analyses

Zelfstandige map: met alleen deze map en een venv werkt alles. Er wordt **niet getraind**; het model komt van een teamgenoot.

```
video (data/match.mp4)
  → detectie (models/best.pt, YOLO26l, imgsz 1280)         core/detection.py  (cache in cache/)
  → tracking (ByteTrack) + bal-Kalman in meters            core/tracking.py
  → (optioneel) bal herdetecteren in crops                 core/ball_refine.py (cache in cache/)
  → homografie per frame (geklikte keypoints)              core/homography.py
  → minimap + balbezit                                     core/render.py, core/possession.py
  → analyses (afstand, snelheid, zones, km, aanvalsrichting, heatmaps)  analytics/
  → systeemevaluatie tegen eigen ground truth              evaluation/
```

## Installatie

```powershell
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

Nodig in de map (niet in git, te groot): `models/best.pt` en `data/match.mp4`; voor de GPU-job ook
`models/yolo26l_coco.pt` (COCO-basismodel). De gelabelde beelden voor die job staan in `datasets/`
(kopie van `data/dataset_yolo` en `kobe/notebooks/dataset`, gemaakt met `gpu/prepare_datasets.py`).

## Gebruik (vanuit deze map)

| Wat | Commando | Uitvoer |
|---|---|---|
| **Alles in één keer** | `python run_all.py --no-video` (opties: `--weights --imgsz --stride --ball-refine --no-video`; detectie uit `cache/` indien aanwezig) | minimap-csv's, `output/analytics/`, `output/evaluation/`, **`output/RAPPORT_<model>.md`** |
| Minimapvideo + CSV's | `python run_minimap.py` (opties: `--start --end --stride --conf --ball-conf --weights --no-video --ball-refine`) | `output/minimap_best.mp4`, `output/tracks_best.csv`, `output/frames_best.csv` |
| Bal herdetecteren in crops (optioneel) | `python run_minimap.py --ball-refine` (eerste keer ±0,1–0,3 s per frame zonder bal, daarna uit `cache/`) | zelfde uitvoer; `ball_state` kan `refined` zijn |
| Homografie controleren | `python check_homography.py` | reprojectiefout in meters per keyframe, spreiding, drift → `output/homography_*.csv` |
| Keypoints klikken | `python click_keypoints.py` | `data/keypoints.json` |
| Analyses | `python run_analytics.py` | `output/analytics/` (tabellen, heatmaps met aanvalsrichting, gemiddelde posities, snelheidszones, kilometers, aanvalsrichting + aanvallen, rapport) |
| Ground truth labelen | `python label_ground_truth.py --every 30` | `data/ground_truth.json` |
| Systeem evalueren | `python evaluate_system.py` (personen op het lichaam geklikt: `--anchor body`; ander labelbestand: `--gt`) | `output/evaluation/system_eval_best_<anchor>.md/.json` |
| Streamlit-app | `streamlit run app/app.py` | interactief (zie `app/README.md`) |
| **GPU-job (teamgenoot)** | `python gpu/run_all.py` (rooktest: `--quick --device cpu`; uitleg: `gpu/README.md`) | `gpu/results/<tijdstip>/SUMMARY.md`: PR-curves COCO ↔ best.pt ↔ fine-tune, drempels, NMS-sweep, `best_ball_ft.pt`, detectiecaches |
| Resultaat GPU-job overnemen | `model/best_ball_ft.pt` → `models/`, `cache_export/*.json` → `cache/`, dan `python run_all.py --no-video --weights models/best_ball_ft.pt` | YOLO draait niet opnieuw (draagbare cachesleutel) |
| Frames voor een eerlijke testset | `python gpu/export_unseen_frames.py` | `gpu/unseen_frames/` (63–199, 246–524, elk 5e) om in Roboflow te labelen |
| GPU-job voor een teamgenoot (PR-curves COCO ↔ best.pt ↔ fine-tune, drempels, NMS-sweep, bal-fine-tune) | `python gpu/run_all.py` (rooktest zonder GPU: `--quick --device cpu`; zie `gpu/README.md`) | `gpu/results/<tijdstip>/SUMMARY.md` + CSV's/figuren, `models/best_ball_ft.pt` |
| Ongeziene frames exporteren om te labelen (eerlijke testset) | `python gpu/export_unseen_frames.py` | `gpu/unseen_frames/` |
| Tests | `python -m pytest -q` (zonder het echte model: `-m "not smoke"`) | |

Gebruik `venv\Scripts\python` als de venv niet geactiveerd is.

## Datacontract

- `tracks_<model>.csv`: `frame,time_s,track_id,group,team,x_m,y_m,conf,box_x1,box_y1,box_x2,box_y2` (box in pixels) — één rij per persoon per frame.
- `frames_<model>.csv`: `frame,time_s,ball_x_m,ball_y_m,ball_state,possession_team,possession_track_id`.
  `ball_state` = `detected` / `interpolated` / `refined` (alleen met `--ball-refine`: gevonden in een uitsnede rond de verwachte positie) / `none`.
- Meters; veld 105 × 68; oorsprong = verre zijlijn × linker doellijn. Team A = PSG (`player_A`), B = Inter.

## Labeltool (systeemevaluatie)

`a`/`b`/`k`/`r` kies team A / B / keeper / scheidsrechter · linksklik = voeten · rechtsklik = verwijderen ·
`g` + klik = bal · `v` = geen bal zichtbaar · `1`/`2`/`0` = balbezit A / B / niemand · `u` undo · `c` frame leegmaken ·
spatie/enter = bevestigen · `s` overslaan · `p` vorige · `q` stoppen. Wordt na elke wijziging opgeslagen.

## Beperkingen

- Homografie geldt alleen voor punten op het veld: voeten kloppen, een bal in de lucht niet.
- Keypoints zijn met de hand geklikt en gelden alleen voor `match.mp4`; een andere video heeft nieuwe keypoints nodig.
- Snelheden komen uit afgevlakte posities; homografieruis en ID-switches beïnvloeden afstand en snelheid.
- Eén korte clip (±22 s) van één wedstrijd: geen algemene uitspraken.

Zie `WIJZIGINGEN.md` voor wat er tegenover de oude mappen veranderd is.
