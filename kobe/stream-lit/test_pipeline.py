"""Quick check of pipeline.py without Streamlit: detect a short frame range, post-process, render.

Run from the repo root:
    venv\\Scripts\\python kobe\\stream-lit\\test_pipeline.py            (frames 0-60, stride 2, yolo26s)
    venv\\Scripts\\python kobe\\stream-lit\\test_pipeline.py --end 120 --imgsz 640
"""

import argparse
import time
from dataclasses import replace

import cv2

import pipeline as pl

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default=str(pl.HOMOGRAPHY_DIR / "pretrained" / "yolo26s.pt"))
parser.add_argument("--imgsz", type=int, default=1280)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=60)
parser.add_argument("--stride", type=int, default=2)
args = parser.parse_args()

video = pl.DEFAULT_VIDEO
info = pl.video_info(video)
print("video:", info)

# ---- Group 1: detection (cached) ----
t0 = time.time()
dets = pl.detect(video, args.weights, args.imgsz, 0.7, args.start, args.end, args.stride,
                 progress=lambda n, total: print(f"\rdetect {n}/{total}", end="") if n % 5 == 0 else None)
print(f"\ndetections: {len(dets['frames'])} frames in {time.time() - t0:.1f} s, classes {dets['meta']['classes']}")
t0 = time.time()
dets_again = pl.detect(video, args.weights, args.imgsz, 0.7, args.start, args.end, args.stride)
print(f"second call (from cache): {time.time() - t0:.2f} s, same: {dets_again['frames'] == dets['frames']}")

# ---- Group 2: every ball smoothing, with and without tracking ----
ids = sorted(dets["frames"])
Hs = pl.frame_homographies(ids)
for p in (pl.PostParams(), pl.PostParams(ball_smoothing="none", tracking=False),
          pl.PostParams(ball_smoothing="kalman"), pl.PostParams(teams=False, conf_people=0.5)):
    t0 = time.time()
    result = pl.postprocess(dets, Hs, p, info["fps"])
    stats = pl.statistics(result, len(ids))
    print(f"{p.ball_smoothing:13s} tracking={p.tracking!s:5s} teams={p.teams!s:5s} "
          f"{time.time() - t0:.2f} s  classes {stats['per_class']}  teams {stats['per_team']}  "
          f"ball % {({k: round(v) for k, v in stats['ball_pct'].items()})}  tracks {stats['tracks']}")

# Without homography (as for an uploaded video): no pitch positions, no minimap
result_noh = pl.postprocess(dets, None, pl.PostParams(ball_smoothing="kalman"), info["fps"])
print("no homography:", result_noh["notes"], pl.statistics(result_noh, len(ids))["per_class"])

# ---- Kalman sanity check: a ball moving at 10 m/s along x, measured every frame ----
kf = pl.BallKalman((0.0, 0.0), accel_std=30, meas_std=0.5)
for k in range(1, 31):
    kf.predict(1 / 30)
    kf.update((10 * k / 30, 0.0))
print(f"Kalman after 1 s: position {kf.x[:2].round(2)} m (true 10, 0), velocity {kf.x[2:].round(2)} m/s")

# ---- Group 3: render one frame and a short video ----
result = pl.postprocess(dets, Hs, replace(pl.PostParams(), ball_smoothing="kalman"), info["fps"])
d = pl.DisplayParams(labels=True, trail=10)
frame = pl.read_frame(video, ids[len(ids) // 2])
t0 = time.time()
image, mini = pl.render_frame(frame, ids[len(ids) // 2], result, Hs[ids[len(ids) // 2]], d)
print(f"render_frame: {time.time() - t0:.3f} s")
pl.OUTPUT_DIR.mkdir(exist_ok=True)
cv2.imwrite(str(pl.OUTPUT_DIR / "test_frame.jpg"), pl.side_by_side(image, mini))
t0 = time.time()
out = pl.render_video(video, ids, result, Hs, d, pl.OUTPUT_DIR / "test_video.mp4", info["fps"] / args.stride)
print(f"render_video: {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f} s")
