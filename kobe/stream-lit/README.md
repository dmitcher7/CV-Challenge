# Streamlit app: video → detections → minimap

An interactive frontend to **play with the parameters** of the whole pipeline on our clip:
YOLO detections → filters / teams / tracking / ball smoothing → homography → 2D minimap.
The app never trains a model.

## Run (from the repo root)

```powershell
venv\Scripts\pip install -r kobe\stream-lit\requirements.txt   # once: streamlit, supervision<0.31, imageio-ffmpeg
venv\Scripts\streamlit run kobe/stream-lit/app.py
```

Needed files (not in git): `data/raw_video/match.mp4` and at least one weights file:
`kobe/training/runs/<name>/weights/best.pt`, any `kobe/stream-lit/models/*.pt`, or the COCO models
`kobe/homography/pretrained/yolo26s.pt` / `yolo26m.pt`. The app says so when one is missing.

Test the pipeline without Streamlit (frames 0–60, stride 2, yolo26s, ≈15 s):

```powershell
venv\Scripts\python kobe\stream-lit\test_pipeline.py
```

## Files

| File | What |
|---|---|
| `app.py` | Streamlit UI only (sidebar, tabs, caching) |
| `pipeline.py` | detection + disk cache, post-processing, rendering; no Streamlit, so it can be tested |
| `test_pipeline.py` | runs every pipeline step on a short frame range and prints the results |
| `models/` | put extra `.pt` files here (e.g. downloaded from Kaggle); they appear in the dropdown |
| `cache/` | detections (`detections_<weights>_<imgsz>_<hash>.json`) and uploaded videos; not committed |
| `output/` | rendered videos; not committed |

`pitch.py`, `homography.py` and `keypoints.json` are **imported/read from `kobe/homography/`, not copied**:
one source of truth, so new clicks in `keypoints.json` or a fix in the homography are used here immediately
(the cache key contains the modification time of `keypoints.json`). `preview_minimap.py` cannot be imported
(it parses arguments and renders a video at import time), so its post-processing (`on_pitch`, `jersey_colour`,
`assign_teams`, `ball_track`) is adapted in `pipeline.py`, with a comment where it comes from.
All paths are computed from `__file__`, so the folder can move to `Final/` (adjust `HOMOGRAPHY_DIR` then).

## The three parameter groups (sidebar)

The split follows **what has to be recomputed** when a parameter changes.

| Group | Parameters | Cost when changed |
|---|---|---|
| **1. Model** | weights, inference `imgsz` (640/960/1280), NMS IoU, frame range, stride | YOLO again: only on **Detecteer**, minutes on CPU |
| **2. Post-processing** | confidence players / ball, classes, ball filter (max size, confirmation window + speed, max gap, pitch margin), teams (K-means, outlier factor), ByteTrack (on/off, activation, lost buffer, matching), ball smoothing (none / interpolation / Kalman + its noise) | < 0.2 s, on the cached detections |
| **3. Display** | pitch lines, boxes, labels, track IDs, trail length, minimap scale, camera view | only redraws |

Why this works: YOLO runs **once at confidence 0.05** and *all* boxes are stored (frame, class, conf, xyxy,
and the jersey colour of every person, measured while the frame is in memory). Every threshold above 0.05 is
then just a filter on that list, so the confidence sliders are instant. The cache key is
(video + mtime, weights + mtime, imgsz, IoU, frame range, stride): changing one of those shows
"druk op Detecteer"; going back to an earlier combination reuses its cache file.

Training parameters (epochs, augmentations, training imgsz, model size) are not in the app: they
change the weights, so train another model (`kobe/training/`) and pick it in the dropdown.

### Post-processing in order (`pipeline.postprocess`)

1. **People filter:** confidence ≥ threshold, class shown, feet (bottom-centre of the box) on the pitch + margin.
2. **Teams:** K-means (k = 2) on the Lab jersey colour of all `player` boxes; a jersey further than
   `outlier factor` × the typical distance from both centres is `other` (goalkeeper, referee). Skipped for a model
   without a merged `player` class (it has its own team classes).
3. **Tracking:** `supervision.ByteTrack` on the boxes, frame by frame (`frame_rate` = fps / stride). With teams
   on, every track gets the team it got most often (one bad jersey measurement no longer flips the colour).
4. **Ball:** most confident ball per frame that is small enough, on the pitch and not on a painted spot →
   kept only if another frame within the window saw a ball nearby → smoothing:
   - *interpolation*: linear in image pixels for gaps ≤ max gap (as in `preview_minimap.py`);
   - *Kalman*: `BallKalman`, constant velocity in **pitch metres** (state x, y, vx, vy). `predict` moves the ball
     with its velocity and grows the uncertainty (acceleration std = kicks); `update` mixes in the measurement
     (measurement std). After the last measurement it predicts for at most *max gap* frames; it restarts after a
     longer gap or when a measurement is > 10 m from the prediction.

## Tabs

- **Frame:** frame slider (detected frames only), video with boxes next to the minimap. The playground: updates
  in ≈0.2 s after a group-2/3 change.
- **Video:** *Render video* → side-by-side video of the detected range, H.264 via `imageio-ffmpeg` (OpenCV's
  `mp4v` does not play in the browser), at fps / stride so it plays in real time. Download button.
- **Statistieken:** detections per class/team after filtering, % of frames with a ball (raw / confirmed /
  after smoothing), number of tracks, detection time, and a table with all current parameters (for the report).

## Known limitations

- **Planar assumption:** the homography is only valid on the grass. A ball in the air is projected too far
  from the camera, so its minimap position is wrong (and the Kalman filter sees it as a jump).
- **Minimap only for `match.mp4`:** `keypoints.json` was clicked for that clip. An uploaded mp4 gets detections,
  tracking and teams, but no minimap, pitch lines or pitch-margin filter.
- **CPU detection time:** ≈0.65 s per frame at 1280 with yolo26s, so ≈7 min for the full clip (≈2 min at 640, but
  then the ball is almost never found). Use frame range / stride to try things; the cache makes it a one-off cost.
- Track IDs restart from 1 at every post-processing change (the tracker runs again), and ByteTrack is
  class-agnostic. `supervision` must stay < 0.31 (ByteTrack moved to the separate `trackers` package there).
