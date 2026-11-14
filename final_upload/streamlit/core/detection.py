"""YOLO detection of the video frames, cached on disk.

YOLO is the slow part, so it runs once per combination of weights / imgsz / iou / frame range at a low
confidence (DETECT_CONF); the real thresholds are applied afterwards (see pipeline.py), which is fast.
The cache key identifies the weights and the video by name, size and content hash (see cache_path): a retrained
a retrained .pt (same name, new content) never reuses old detections, and cache files copied from another PC are reused.
"""

import hashlib
import json
import os
import time
from pathlib import Path

from core.paths import CACHE_DIR
from core.teams import BALL_CLASSES, PEOPLE_CLASSES, jersey_colour
from core.video import frame_ids, iter_frames

DETECT_CONF = 0.05   # YOLO runs once at this low confidence
CPU_THREADS = 6      # torch threads on CPU (more than the P-cores makes a laptop slower)


_FINGERPRINTS = {}


def file_fingerprint(path):
    """Portable identity of a file: name, size and md5 of the content (no absolute path, no mtime, so a copied
    folder on another PC gives the same value). The md5 is memoised per (path, size, mtime) within a process."""
    path = Path(path).resolve()
    st = path.stat()
    memo = (str(path), st.st_size, st.st_mtime)
    if memo not in _FINGERPRINTS:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        _FINGERPRINTS[memo] = {"name": path.name, "bytes": st.st_size, "md5": h.hexdigest()}
    return _FINGERPRINTS[memo]


def legacy_cache_key(video, weights):
    """The old (pre 2026-11-13 evening) key part: absolute paths + mtimes. Only used to find existing caches."""
    video, weights = Path(video).resolve(), Path(weights).resolve()
    return {"video": str(video), "video_mtime": video.stat().st_mtime, "weights": str(weights),
            "weights_mtime": weights.stat().st_mtime}


def portable_cache_key(video, weights):
    return {"video": file_fingerprint(video), "weights": file_fingerprint(weights), "key_version": 2}


def _digest(key):
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]


def cache_path(video, weights, imgsz, iou, start, end, stride, nms=None, portable=False):
    """Cache file for one combination of detection parameters.

    The key is portable: video and weights are identified by name + size + content hash, not by absolute path or
    mtime, so cache files copied to cache/ on another PC (e.g. from the GPU run) are reused. A retrained
    .pt (same name, other content) never reuses old detections.
    Backwards compatible: when only an old-style cache file (key with absolute paths + mtimes) exists for the
    same parameters, that file is returned and keeps being used.

    nms: None = YOLO's default (one2many head + NMS with iou); False = the NMS-free one2one head of YOLO26
    (iou has no effect then). nms is only part of the key when it is not None.
    portable=True: always the portable file name (to copy a cache to another PC).
    """
    params = {"imgsz": imgsz, "iou": iou, "start": start, "end": end, "stride": stride, "conf": DETECT_CONF}
    if nms is not None:
        params["nms"] = nms
    stem = f"detections_{Path(weights).stem}_{imgsz}"
    path = CACHE_DIR / f"{stem}_{_digest({**portable_cache_key(video, weights), **params})}.json"
    if not path.exists() and not portable:
        legacy = CACHE_DIR / f"{stem}_{_digest({**legacy_cache_key(video, weights), **params})}.json"
        if legacy.exists():
            return legacy
        candidates = sorted(CACHE_DIR.glob(f"{stem}_*.json"))
        if candidates:
            return candidates[0]
    return path


def load_model(weights):
    """YOLO model (a .pt file or an exported OpenVINO folder); on CPU torch uses at most CPU_THREADS threads."""
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        torch.set_num_threads(min(CPU_THREADS, os.cpu_count() or 1))
    return YOLO(str(weights), task="detect")


def boxes_of(result, names, frame):
    """YOLO result of one frame -> [{"name", "box", "conf", "jersey"}] for the classes we know."""
    found = []
    for cls, conf, box in zip(result.boxes.cls.int().tolist(), result.boxes.conf.tolist(),
                              result.boxes.xyxy.tolist()):
        name = names[cls]
        if name not in PEOPLE_CLASSES and name not in BALL_CLASSES:
            continue   # e.g. COCO 'car', 'chair'
        found.append({"name": name, "box": [round(v, 1) for v in box], "conf": round(conf, 3),
                      "jersey": jersey_colour(frame, box) if name in PEOPLE_CLASSES else None})
    return found


def detect(video, weights, imgsz=960, iou=0.7, start=0, end=0, stride=1, model=None, progress=None, nms=None):
    """Detections of the frames start..end (inclusive, every stride-th), from the disk cache when possible.

    nms: None (default) = NMS with iou; False = NMS-free one2one head (see cache_path).

    Returns a list [{"frame": idx, "detections": [{"name", "box" [x1, y1, x2, y2] px, "conf",
    "jersey" (Lab colour or None)}]}] in frame order. progress(done, total) is called after every frame.
    """
    path = cache_path(video, weights, imgsz, iou, start, end, stride, nms=nms)
    if path.exists():
        return json.loads(path.read_text())["frames"]
    model = model or load_model(weights)
    ids = frame_ids(start, end, stride)
    frames, t0 = [], time.time()
    extra = {} if nms is None else {"nms": nms}
    for n, (idx, frame) in enumerate(iter_frames(video, ids)):
        result = model.predict(frame, imgsz=imgsz, conf=DETECT_CONF, iou=iou, verbose=False, **extra)[0]
        frames.append({"frame": idx, "detections": boxes_of(result, model.names, frame)})
        if progress:
            progress(n + 1, len(ids))
    meta = {"video": str(video), "weights": str(weights), "imgsz": imgsz, "iou": iou, "start": start,
            "end": end, "stride": stride, "conf": DETECT_CONF, "nms": nms, "seconds": round(time.time() - t0, 1)}
    CACHE_DIR.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")   # write to a temporary file first: a crash never leaves half a cache
    tmp.write_text(json.dumps({"meta": meta, "frames": frames}))
    tmp.replace(path)
    return frames
