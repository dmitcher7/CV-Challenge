"""Optional ball refinement: run YOLO again on a crop around where the ball should be (off by default).

The full frame (1920 px) is seen by YOLO at 960 px, where the ball is only a few pixels and is
sometimes missed. For a frame without a ball we cut a CROP_SIZE crop around the expected position and
run YOLO at CROP_IMGSZ: the same scale as the full frame (the model is scale-sensitive: an UPSCALED
crop finds nothing), but without the clutter of the rest of the frame.
See experiments/ball_prediction/results.md (method M3).

A crop hit (conf >= REFINE_CONF) is only kept when it is confirmed (confirm_hits): close to a detected
ball in a nearby frame, or to another (possibly weaker) crop hit in an adjacent frame, at a plausible
speed. An isolated hit (a white spot on the touchline) is dropped. Accepted hits get ball_state "refined".

YOLO on the crops is slow (~0.3 s per crop on CPU), so the raw crop detections are cached in cache/
(key: video + weights incl. modification time + crop settings). Running again is fast.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from core.detection import DETECT_CONF, load_model
from core.paths import CACHE_DIR
from core.teams import BALL_CLASSES
from core.video import iter_frames

CROP_SIZE = 640          # px: square crop of the original frame around the expected ball position ...
CROP_IMGSZ = 320         # ... seen by YOLO at half size, like the full frame (1920 -> 960); never upscale
REFINE_CONF = 0.3        # a crop hit needs at least this confidence (0.15 picked up touchline spots) ...
REFINE_CONFIRM_CONF = 0.15   # ... while a crop hit that only confirms it in an adjacent frame needs this much
REFINE_SEARCH = 60       # frames: only search when a known ball is at most this far away in time
REFINE_NEIGHBOUR = 3     # frames: a hit is confirmed by a ball / another hit at most this far away ...
REFINE_SPEED = 1.5       # ... and at most this many metres per frame (+ REFINE_SLACK) from it
REFINE_SLACK = 1.0       # m: homography / detection noise on top of the speed limit


def crop_windows(ball_xy, ids, hs, width, height):
    """{frame: (x0, y0)}: top-left corner of the crop for every processed frame without a ball.

    ball_xy: {frame: pitch (x, y)} of the frames that have a ball (detected or interpolated).
    The crop is centred on the ball position interpolated between the previous and the next known ball
    (or the nearest one, when only one side is within REFINE_SEARCH frames) and kept inside the image.
    """
    known = sorted(ball_xy)
    windows = {}
    for idx in ids:
        if idx in ball_xy:
            continue
        before = [k for k in known if idx - REFINE_SEARCH <= k < idx]
        after = [k for k in known if idx < k <= idx + REFINE_SEARCH]
        if before and after:
            a, b = before[-1], after[0]
            t = (idx - a) / (b - a)
            xy = (1 - t) * np.asarray(ball_xy[a], float) + t * np.asarray(ball_xy[b], float)
        elif before or after:
            xy = np.asarray(ball_xy[before[-1] if before else after[0]], float)
        else:
            continue
        cx, cy = hs.pitch_to_image([xy], idx)[0]
        if not (-CROP_SIZE / 2 < cx < width + CROP_SIZE / 2 and -CROP_SIZE / 2 < cy < height + CROP_SIZE / 2):
            continue   # expected far outside the image
        x0 = int(round(min(max(cx - CROP_SIZE / 2, 0), max(width - CROP_SIZE, 0))))
        y0 = int(round(min(max(cy - CROP_SIZE / 2, 0), max(height - CROP_SIZE, 0))))
        windows[idx] = (x0, y0)
    return windows


def crop_cache_path(video, weights):
    """Cache file with the raw crop detections of one video + weights file (+ crop settings).

    Portable key (name + size + content hash of video and weights, see core.detection.cache_path); an existing
    old-style file (absolute paths + mtimes in the key) is still used when the portable one does not exist."""
    from core.detection import legacy_cache_key, portable_cache_key
    params = {"size": CROP_SIZE, "imgsz": CROP_IMGSZ, "conf": DETECT_CONF}

    def file_for(key):
        digest = hashlib.sha1(json.dumps({**key, **params}, sort_keys=True).encode()).hexdigest()[:10]
        return CACHE_DIR / f"ball_crops_{Path(weights).stem}_{digest}.json"
    path = file_for(portable_cache_key(video, weights))
    if not path.exists():
        legacy = file_for(legacy_cache_key(video, weights))
        if legacy.exists():
            return legacy
    return path


def _key(idx, window):
    return f"{idx}|{window[0]}|{window[1]}"


def _load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def missing_crops(video, weights, windows):
    """Frames of `windows` whose crop is not in the cache yet (0 = refinement runs without YOLO)."""
    cache = _load(crop_cache_path(video, weights))
    return [idx for idx, w in windows.items() if _key(idx, w) not in cache]


def detect_crops(video, weights, windows, model=None, progress=None):
    """{frame: [{"name", "box" (full-frame px), "conf"}]} ball detections in the crops, from the cache when possible.

    YOLO only runs for crops that are not cached yet (model loaded from weights when needed);
    progress(done, total) is called after every new crop.
    """
    path = crop_cache_path(video, weights)
    cache = _load(path)
    todo = [idx for idx, w in windows.items() if _key(idx, w) not in cache]
    if todo:
        model = model or load_model(weights)
        t0 = time.time()
        for n, (idx, frame) in enumerate(iter_frames(video, todo)):
            x0, y0 = windows[idx]
            crop = frame[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]
            res = model.predict(crop, imgsz=CROP_IMGSZ, conf=DETECT_CONF, iou=0.7, verbose=False)[0]
            # boxes are in crop pixels (YOLO scales back to the input size): shift them into the full frame
            cache[_key(idx, windows[idx])] = [
                [model.names[c], [round(b[0] + x0, 1), round(b[1] + y0, 1), round(b[2] + x0, 1), round(b[3] + y0, 1)],
                 round(cf, 3)]
                for c, cf, b in zip(res.boxes.cls.int().tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist())
                if model.names[c] in BALL_CLASSES]
            if progress:
                progress(n + 1, len(todo))
        CACHE_DIR.mkdir(exist_ok=True)
        tmp = path.with_suffix(".tmp")   # never leave half a cache file behind
        tmp.write_text(json.dumps(cache))
        tmp.replace(path)
        print(f"ball refine: {len(todo)} crops in {time.time() - t0:.0f} s -> {path.name}")
    return {idx: [{"name": n, "box": b, "conf": c} for n, b, c in cache.get(_key(idx, w), [])]
            for idx, w in windows.items()}


def confirm_hits(hits, known, stride=1):
    """The frames whose crop hit becomes a "refined" ball. Pure function (no model, no video).

    hits:  {frame: (pitch (x, y), conf)} best crop hit per frame (>= REFINE_CONFIRM_CONF, on the pitch).
    known: {frame: pitch (x, y)} detected balls of the tracker.
    A hit with conf >= REFINE_CONF is kept when a known ball or another hit, at most REFINE_NEIGHBOUR frames
    away (at least one processed frame with a stride), lies within REFINE_SPEED m/frame + REFINE_SLACK of it.
    """
    reach = max(REFINE_NEIGHBOUR, stride)

    def near(idx, xy, others):
        return any(0 < abs(j - idx) <= reach
                   and np.hypot(*np.subtract(others[j], xy)) <= REFINE_SPEED * abs(j - idx) + REFINE_SLACK
                   for j in others)

    others = {j: xy for j, (xy, _) in hits.items()}
    return {idx for idx, (xy, conf) in hits.items()
            if conf >= REFINE_CONF and (near(idx, xy, known) or near(idx, xy, others))}


def fill_between(points, refined, ids, max_gap):
    """{frame: pitch xy} linearly interpolated in the gaps next to a refined ball.

    points: {frame: pitch xy} of all measured balls (detected + refined); refined: the refined frames.
    A gap is filled when at least one end is refined, it spans at most max_gap frames and the ball
    would not move faster than REFINE_SPEED m/frame (other gaps were already handled by the tracker).
    """
    filled, order = {}, sorted(points)
    for a, b in zip(order[:-1], order[1:]):
        if (a in refined or b in refined) and 1 < b - a <= max_gap and \
                np.hypot(*np.subtract(points[b], points[a])) <= REFINE_SPEED * (b - a):
            for i in ids:
                if a < i < b:
                    t = (i - a) / (b - a)
                    filled[i] = (1 - t) * np.asarray(points[a], float) + t * np.asarray(points[b], float)
    return filled
