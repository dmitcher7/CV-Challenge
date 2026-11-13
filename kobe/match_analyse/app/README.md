# Streamlit app (match_analyse)

Interactive frontend on `core/`: play with the parameters of video -> detection -> tracking -> minimap.

```powershell
venv\Scripts\pip install -r kobe\match_analyse\requirements.txt   # once
venv\Scripts\streamlit run kobe/match_analyse/app/app.py
```

Needs `data/match.mp4` and at least one `models/*.pt` (not in git; `.pt` files can also be uploaded in the app).

Sidebar groups (split by what must be recomputed):

| Group | Parameters | Cost |
|---|---|---|
| 1. Model | weights, imgsz, NMS IoU, frame range, stride | YOLO, only on **Detecteer**, cached in `cache/` |
| 2. Post-processing | conf players / ball, tracker (lost buffer, matching threshold), ball (max size, max gap, start conf) | fast, `core.pipeline.run` on the cached detections |
| 2. Post-processing (optional) | ball: "Herdetectie in crops" | YOLO on crops of frames without a ball, only on **Herdetecteer bal**, cached in `cache/` |
| 3. Display | preview width | instant |

Tabs: Frame (slider preview, video + minimap), Video (H.264 render + download), Statistieken (tracks, ball %,
possession share, download `tracks.csv` / `frames.csv`, used parameters), Analyses (attacking direction, attacks,
speed zones, kilometres, team heatmaps with attack arrow; computed from the current result, cached, fast).

Dropped compared to the old `kobe/stream-lit` app: supervision/ByteTrack toggle and activation threshold,
Kalman/interpolation smoothing choice, ball confirmation filter, K-means team toggle, outlier factor, class filter,
pitch margin, display toggles (boxes, labels, trails, camera view): `core/` no longer has these options.
