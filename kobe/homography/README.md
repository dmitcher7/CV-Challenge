# Homography: broadcast view → 2D top-down minimap

YOLO only finds players and the ball in the image. To put them on a 2D pitch we need a
**homography H**: a 3×3 matrix that maps every point of the (flat) pitch in the image to its position
on the pitch in metres. H is computed from at least 4 points whose position is known in both:
in the image (clicked) and on the pitch (FIFA dimensions, `pitch.py`).

The camera in our clip **pans and zooms**, so H is different for every frame. We click points on
keyframes (default every 30th frame = 1 s) and interpolate H in between.

## Usage (from the repo root)

```powershell
venv\Scripts\python kobe\homography\click_keypoints.py              # 1. click points (every 30th frame)
venv\Scripts\python kobe\homography\preview_minimap.py --lines-only # 2. check: projected lines on the video
venv\Scripts\python kobe\homography\preview_minimap.py              # 3. video + minimap with detections
```

### 1. `click_keypoints.py`

Layout: the frame on top; below it the minimap (left), status (middle) and a magnifier (right).

| Action | What |
|---|---|
| click a point on the **minimap** (or `n` / `p`) | choose which pitch point you are going to click (red ring) |
| **left click** in the frame | place that point (use the magnifier for precision) |
| **right click** | delete the nearest point |
| `x` / `c` | delete the selected point / all points of this frame |
| `space` / `enter`, `a` | next / previous keyframe |
| `q` / `esc` / close window | stop (everything is saved after every change) |

Tips:
- Click **5–8 points spread over the visible pitch**, not all on one line (points on one line do not
  define a plane). Good points in this clip: the corners of the right penalty area and goal area, the
  penalty spot, where the arc meets the penalty area, the halfway line with the touchline and the centre circle.
- From 4 points on, the **cyan pitch lines** are drawn over the frame. They must lie on the real lines,
  also in the parts of the image where you did not click.
- From 5 points on, every point shows its **error in pixels**: green < 5, orange < 10, red ≥ 10.
  A red point is a misclick or a wrong name: RANSAC already ignores it, but delete or re-click it.
- On a new keyframe the lines of the previous keyframe are shown in **grey** as a guide.
- The tool restarts at the first keyframe with fewer than 4 points, so you can stop and continue later.
- Extra keyframes where the camera moves fast: `--frames 45 75` (existing points are kept), or `--step 15`.

Result: `keypoints.json` (commit it: it is annotation data, just like the labels).

### 2. and 3. `preview_minimap.py`

With Brent's model: `venv\Scripts\python kobe\homography\preview_minimap.py --weights brent\smoke_test\best.pt --imgsz 1600`
(the first run takes ≈35 min on CPU, after that ≈1 min from the cache).

**Faster on the Intel Arc GPU (≈4 min instead of ≈35 min)** via OpenVINO (`pip install openvino`, once):

```powershell
copy brent\smoke_test\best.pt kobe\homography\pretrained\best.pt
venv\Scripts\python -c "from ultralytics import YOLO; YOLO('kobe/homography/pretrained/best.pt').export(format='openvino', imgsz=1600, half=True)"
venv\Scripts\python kobe\homography\preview_minimap.py --weights kobe\homography\pretrained\best_openvino_model --imgsz 1600 --device intel:gpu
```

Measured per frame at imgsz 1600: PyTorch CPU 0.77 s, OpenVINO CPU 0.68 s, **OpenVINO Arc GPU 0.28 s**
(the NPU does not support this model). The FP16 export gives almost the same detections (ball in 563 vs 562 frames).
Export again whenever there is a new `best.pt`.

Writes `output/lines_preview.mp4` or `output/minimap_<model>_<imgsz>.mp4` (one file per model, so models can be
compared): the video with the projected pitch lines next to the minimap, with an **info bar** on top:

- row 1: frame number, time, whether it is a clicked keyframe, possession so far (% per team);
- row 2: players per team, keeper and referees in this frame, keeper yes/no, ball yes (with confidence) /
  yes (interpolated) / no, and which team has the ball.

The minimap also shows a **ball trail** (last 20 frames, older = darker) and a white ring around the player
who has the ball (nearest player within 2 m). The same numbers per frame are written to
`output/minimap_<model>_<imgsz>_frames.csv`, and a summary of the whole clip is printed at the end.
Possession is only as good as the ball detection: a false ball on a player's boot counts as possession. **Where the lines drift away from the real lines between two keyframes, add a
keyframe there.** The cyan quadrilateral on the minimap is the part of the pitch the camera sees.

Detections: default the pretrained COCO model (`person`, `sports ball`) at `imgsz 1280` (≈7 min on CPU),
or a trained model: `--weights kobe\training\runs\final\weights\best.pt`. Detections are cached in
`output/detections_<model>_<imgsz>.json`, so re-rendering takes ≈1 min (`--redetect` runs YOLO again).

- **Teams:** COCO only knows `person`. The jersey colour of every player is measured (mean colour of the torso,
  grass pixels removed) and split into 2 teams with K-means. Players far from both teams (goalkeeper, referee,
  linesmen) keep their own colour. Coloured jerseys are drawn more vivid; dark jerseys stay grey.
  With a trained model that has `player_A`/`player_B` classes, those classes are used instead.
- **Duplicates:** a fine-tuned model sometimes puts two boxes (often with different classes) on one player.
  Of two person boxes with IoU > 0.6 only the most confident is kept (Brent's model: ≈6 per frame removed).
- **Ball:** only found at `imgsz 1280` (at 640 the pretrained model finds it in 1 of 17 test frames, at 1280 in 8 of 17).
  Per frame the most confident ball is kept when it is ≤ 40 px, on the pitch, not on a painted spot (the centre
  and penalty spots were detected as a ball), and confirmed by a ball nearby within 5 frames. Gaps of ≤ 15 frames
  are interpolated (open red circle on the minimap). Known false positive: white boots of players.

### 4. `check_homography.py`: where does the homography drift?

```powershell
venv\Scripts\python kobe\homography\check_homography.py
```

Per frame it projects the pitch lines and measures which fraction of the projected points lies on a white
pixel (± 4 px). On keyframes this is the best achievable (players and shadows hide part of the lines); between
keyframes a much lower score means the interpolated H has drifted. Writes `output/homography_check.png`/`.csv`
and prints ready-to-run `click_keypoints.py --frames ...` commands for:
- **extra keyframes** where the score drops below 75% of the keyframe median (e.g. frame 314: the halfway line
  is ≈50 px off);
- **keyframes that score low themselves**: check their clicks (e.g. keyframe 150 has only 4 points, so its error
  cannot be checked).

## Tracking (`tracking.py`, used by `preview_minimap.py`)

Per-frame detection flickers: a player is PSG in one frame and Inter in the next, a white boot is a "ball" for a
few frames, possession jumps. Tracking connects the detections over time:

| Part | How | Why |
|---|---|---|
| Players | **ByteTrack** (ultralytics): Kalman prediction of every box + IoU matching; weak boxes (conf 0.1–0.3) may continue an existing track | one id per player over time |
| Team per player | **majority vote**: the group with the highest summed confidence over the whole track | one wrong frame no longer flips the team |
| Ball | **Kalman filter** (constant velocity) on the pitch in metres; only candidates within 2 m (+1 m per missed frame) of the prediction are accepted; a weak match is replaced by a confident ball elsewhere; gaps ≤ 15 frames interpolated | false balls (boots, far from the ball's path) are ignored |
| Possession | changes only when another team has the ball for 5 frames in a row | no flickering |

`--no-track` gives the old per-frame behaviour, to compare. Result with Brent's model, checked against the
labeled ball positions (100 labeled frames, ball counted correct within 1.5 m):

| | per frame (`--no-track`) | tracking |
|---|---:|---:|
| ball at the right place | 40 / 71 | **51 / 71** |
| ball at a wrong place | 25 | **7** |
| ball missed | 6 | 13 |
| false ball in a frame without ball (29 frames) | 28 | **9** |
| possession changes in the clip | 10 | 2 |

Caveat: these labeled frames were (probably) used to train Brent's model, so the detector is better on them than
on unseen frames. The comparison between the two methods is fair (same detections), the absolute numbers are not.

ByteTrack needs the package `lap` (ultralytics installs it automatically on first use).

## How it works (`homography.py`)

- **Fit per keyframe:** `cv2.findHomography(image points, pitch points)`. With exactly 4 points the
  solution is exact; with ≥ 5 points RANSAC is used (points more than 1 m off are ignored), so one misclick
  cannot pull H out of shape.
- **Between keyframes:** a 6×5 grid of pitch points is projected into keyframe *a* and keyframe *b*, the image
  positions are interpolated linearly in time, and H is fitted again on that interpolated grid. Before the first /
  after the last keyframe the nearest keyframe is used.
- **Which point of a player?** The bottom-centre of the box (the feet): the homography is only valid for points
  **on the pitch plane** (planar assumption). The head of a player, or a ball in the air, is not on that plane
  and is projected too far away from the camera. For the ball we take the centre of the box; when it is in the
  air its minimap position is wrong. That is a known limitation to mention in the report.
- **Pitch coordinates** (`pitch.py`): metres, origin = far touchline × left goal line, x 0–105, y 0–68.
  Pitch 105 × 68 m (Allianz Arena, Munich). `to_minimap_px` converts metres to pixels on the minimap.

> Note: `scripts/generate_pitch.py` (team template) uses a different scale horizontally (9.62 px/m) than
> vertically (9.41 px/m) and a centre circle that is slightly too large. The minimap here is drawn from the
> metres in `pitch.py` (same scale in both directions). Align on one of the two before `Final/`.
