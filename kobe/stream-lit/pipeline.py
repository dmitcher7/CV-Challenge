"""Pipeline behind the Streamlit app: detect (cached) -> post-process -> render one frame.

The parameters come in three groups (see README.md):
1. Model:           weights, imgsz, NMS IoU, frame range + stride -> detect() runs YOLO (slow, cached on disk)
2. Post-processing: thresholds, ball filter, teams, tracking, smoothing -> postprocess() (fast, no YOLO)
3. Display:         what to draw -> render_frame() (fast)

No Streamlit imports here, so every function can be tested from a plain Python script.

pitch.py and homography.py are imported from kobe/homography (not copied): there is one
source of truth, so a new click in keypoints.json or a fix in the homography is used here too.
preview_minimap.py cannot be imported (it parses arguments and renders a video at import time),
so its post-processing functions (on_pitch, jersey_colour, assign_teams, ball_track) are adapted here.
"""

import hashlib
import json
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parent          # kobe/stream-lit/
REPO_ROOT = APP_DIR.parent.parent
HOMOGRAPHY_DIR = REPO_ROOT / "kobe" / "homography"
TRAINING_RUNS = REPO_ROOT / "kobe" / "training" / "runs"
MODELS_DIR = APP_DIR / "models"                    # extra weights, e.g. downloaded from Kaggle
CACHE_DIR = APP_DIR / "cache"                      # detections (not committed)
OUTPUT_DIR = APP_DIR / "output"                    # rendered videos (not committed)
DEFAULT_VIDEO = REPO_ROOT / "data" / "raw_video" / "match.mp4"

sys.path.insert(0, str(HOMOGRAPHY_DIR))
from homography import Homographies, project  # noqa: E402
from pitch import KEYPOINTS, KEYPOINTS_JSON, LENGTH, LINES, WIDTH, draw_minimap, to_minimap_px  # noqa: E402

DETECT_CONF = 0.05   # YOLO runs once at this low confidence; the real thresholds are applied afterwards
CPU_THREADS = 6      # fastest on Kobe's laptop (6 P-cores), see kobe/training/config.py
MINI_MARGIN = 4      # metres of grass around the pitch on the minimap
OUT_H = 720          # height of the rendered video

# Class names of both model types -> the names used in the app.
# COCO only knows 'person' and 'sports ball'. player_A/player_B (old 5-class models) are kept as they are.
CLASS_MAP = {"person": "player", "sports ball": "ball", "player_B_goal_keeper": "goalkeeper"}
PEOPLE = {"player", "goalkeeper", "referee", "player_A", "player_B"}
KNOWN = PEOPLE | {"ball"}
# Fixed colours (BGR) when there is no team colour
COLOURS = {"player": (210, 210, 210), "player_A": (120, 40, 10), "player_B": (0, 220, 255),
           "goalkeeper": (0, 140, 255), "referee": (30, 30, 30)}
BALL_COLOUR = (0, 0, 255)
# White painted spots look like a ball; a "ball" this close (m) to one is ignored
SPOTS = [KEYPOINTS[n] for n in ("centre", "L_penalty_spot", "R_penalty_spot")]
SPOT_RADIUS = 1.0
KALMAN_GATE = 10.0   # m; a measurement further than this from the prediction restarts the filter


# ======================================================================
# Files: weights, video, keypoints
# ======================================================================

def list_weights():
    """{label: path} of every weights file the app can use (only files that exist)."""
    found = {}
    for path in sorted(TRAINING_RUNS.glob("*/weights/best.pt")):
        found[f"run: {path.parent.parent.name}"] = path
    for path in sorted(MODELS_DIR.glob("*.pt")):
        found[f"models/{path.name}"] = path
    for name in ("yolo26s.pt", "yolo26m.pt"):
        path = HOMOGRAPHY_DIR / "pretrained" / name
        if path.exists():
            found[f"COCO pretrained: {name}"] = path
    return found


def video_info(path):
    """{"frames", "fps", "width", "height"} of a video, or None if it cannot be opened."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    info = {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


def read_frame(path, idx):
    """One frame (BGR) of the video. Seeking is frame-exact for match.mp4 (checked against data/frames)."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def iter_frames(path, ids):
    """Yield (idx, frame) for the frame indices in ids (sorted), reading the video once from start to end."""
    wanted = set(ids)
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(ids))
    idx = min(ids)
    while idx <= max(ids):
        if idx in wanted:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
        elif not cap.grab():   # skip a frame without decoding it to an image
            break
        idx += 1
    cap.release()


def frame_ids(start, end, stride):
    """Frame indices that are processed: start..end (inclusive), every stride-th frame."""
    return list(range(start, end + 1, stride))


def has_keypoints(video):
    """The clicked keypoints are only valid for match.mp4."""
    return Path(video).resolve() == DEFAULT_VIDEO.resolve() and KEYPOINTS_JSON.exists()


def frame_homographies(ids):
    """{frame idx: H (image px -> pitch m)} for the given frames, or None without usable keypoints."""
    try:
        hs = Homographies()
    except SystemExit:   # Homographies() stops when no keyframe has >= 4 points
        return None
    return {i: hs.for_frame(i) for i in ids}


# ======================================================================
# Group 1: detection (YOLO, cached on disk)
# ======================================================================

def cache_path(video, weights, imgsz, iou, start, end, stride):
    """Cache file for one combination of group-1 parameters.

    The key contains the modification time of the video and the weights, so a retrained
    best.pt (same path, new file) never reuses old detections.
    """
    video, weights = Path(video).resolve(), Path(weights).resolve()
    key = {"video": str(video), "video_mtime": video.stat().st_mtime, "weights": str(weights),
           "weights_mtime": weights.stat().st_mtime, "imgsz": imgsz, "iou": iou,
           "start": start, "end": end, "stride": stride, "conf": DETECT_CONF}
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    return CACHE_DIR / f"detections_{weights.stem}_{imgsz}_{digest}.json"


def load_model(weights):
    """YOLO model; on CPU torch is limited to CPU_THREADS threads (the E-cores make it slower)."""
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        torch.set_num_threads(CPU_THREADS)
    return YOLO(str(weights))


def jersey_colour(frame, box):
    """Mean Lab colour of the torso (upper half of the body, middle half of the width), grass removed.

    Copied from preview_minimap.py. Measured during detection (the frame is in memory then),
    so the team K-means later needs no video at all.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    crop = frame[int(y1 + 0.15 * h):int(y1 + 0.5 * h), int(x1 + 0.25 * w):int(x2 - 0.25 * w)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    grass = (hsv[:, 0] > 30) & (hsv[:, 0] < 90) & (hsv[:, 1] > 50)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)[~grass]
    return [round(v, 1) for v in lab.mean(axis=0).tolist()] if len(lab) >= 5 else None


def boxes_of(result, names, frame):
    """YOLO result of one frame -> [{"name", "box", "conf", "jersey"}] for the classes the app knows."""
    found = []
    for cls, conf, box in zip(result.boxes.cls.int().tolist(), result.boxes.conf.tolist(),
                              result.boxes.xyxy.tolist()):
        name = CLASS_MAP.get(names[cls], names[cls])
        if name not in KNOWN:
            continue   # e.g. COCO 'car', 'chair'
        found.append({"name": name, "box": [round(v, 1) for v in box], "conf": round(conf, 3),
                      "jersey": jersey_colour(frame, box) if name in PEOPLE else None})
    return found


def detect(video, weights, imgsz, iou, start, end, stride, model=None, progress=None):
    """Run YOLO on the selected frames once at DETECT_CONF and store all boxes in the disk cache.

    Same format as the cache of preview_minimap.py ({frame idx: [{"name", "box", "conf"}]}), with an
    extra "jersey" per person and a "meta" block. progress(done, total) is called after every frame.
    """
    path = cache_path(video, weights, imgsz, iou, start, end, stride)
    if path.exists():
        return load_detections(path)
    model = model or load_model(weights)
    ids = frame_ids(start, end, stride)
    frames, t0 = {}, time.time()
    for n, (idx, frame) in enumerate(iter_frames(video, ids)):
        result = model.predict(frame, imgsz=imgsz, conf=DETECT_CONF, iou=iou, verbose=False)[0]
        frames[idx] = boxes_of(result, model.names, frame)
        if progress:
            progress(n + 1, len(ids))
    classes = sorted({CLASS_MAP.get(n, n) for n in model.names.values()} & KNOWN)
    meta = {"video": str(video), "weights": str(weights), "imgsz": imgsz, "iou": iou, "start": start,
            "end": end, "stride": stride, "conf": DETECT_CONF, "classes": classes,
            "seconds": round(time.time() - t0, 1), "frames": len(frames)}
    CACHE_DIR.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")   # write to a temporary file first: a crash never leaves half a cache
    tmp.write_text(json.dumps({"meta": meta, "frames": frames}))
    tmp.replace(path)
    return load_detections(path)


def load_detections(path):
    """{"meta": {...}, "frames": {frame idx (int): [detection, ...]}} from a cache file."""
    data = json.loads(Path(path).read_text())
    return {"meta": data["meta"], "frames": {int(k): v for k, v in data["frames"].items()}}


# ======================================================================
# Group 2: post-processing (on the cached detections, no YOLO)
# ======================================================================

@dataclass(frozen=True)
class PostParams:
    conf_people: float = 0.3          # confidence threshold for player / goalkeeper / referee
    conf_ball: float = 0.1            # separate threshold: the ball is small and gets low scores
    classes: tuple = ("ball", "player", "goalkeeper", "referee", "player_A", "player_B")  # shown classes
    max_ball_size: int = 40           # px; a bigger "ball" box is something else (a head, a sign)
    ball_confirm_frames: int = 5      # a ball needs another ball within this many frames ...
    ball_confirm_speed: float = 40.0  # ... and at most this many px per frame away
    max_ball_gap: int = 15            # frames that may be filled (interpolation / Kalman prediction)
    pitch_margin: float = 3.0         # metres outside the lines that still count as "on the pitch"
    teams: bool = True                # K-means on jersey colour (only for a merged 'player' class)
    outlier_factor: float = 2.5       # jersey further than this x the typical distance from both teams = other
    tracking: bool = True             # ByteTrack on the cached boxes
    track_activation: float = 0.25    # ByteTrack: min confidence to start a new track
    lost_buffer: int = 30             # ByteTrack: frames (at 30 fps) a lost track is kept
    match_threshold: float = 0.8      # ByteTrack: max IoU cost for a match (higher = easier matching)
    ball_smoothing: str = "interpolation"   # "none" / "interpolation" / "kalman"
    kalman_accel: float = 30.0        # Kalman: std of unknown accelerations (kicks), m/s^2
    kalman_meas: float = 0.5          # Kalman: std of the measured position, m


def to_pitch(box, H, is_ball=False):
    """Pitch position (m) of a box: the feet (bottom-centre) for people, the centre for the ball.

    Planar assumption: only points on the grass are mapped correctly, so a ball in the air is wrong.
    """
    if H is None:
        return None
    x1, y1, x2, y2 = box
    point = ((x1 + x2) / 2, (y1 + y2) / 2 if is_ball else y2)
    return project([point], H)[0]


def on_pitch(xy, margin):
    """True when a pitch position is inside the lines plus margin (always True without homography)."""
    if xy is None:
        return True
    return -margin <= xy[0] <= LENGTH + margin and -margin <= xy[1] <= WIDTH + margin


def filter_people(detections, Hs, p):
    """{idx: [object]} with the people above the threshold, of a shown class, on the pitch.

    Every object is a new dict (the cached detections stay untouched) with its pitch position "xy".
    """
    frames = {}
    for idx, ds in detections.items():
        H = Hs[idx] if Hs else None
        kept = []
        for d in ds:
            if d["name"] not in PEOPLE or d["name"] not in p.classes or d["conf"] < p.conf_people:
                continue
            xy = to_pitch(d["box"], H)
            if on_pitch(xy, p.pitch_margin):
                kept.append({**d, "xy": xy, "colour": COLOURS[d["name"]], "team": d["name"], "track": None})
        frames[idx] = kept
    return frames


def lab_to_bgr(lab):
    """Lab jersey colour -> vivid BGR colour for drawing (copied from preview_minimap.py).

    Coloured jerseys get more saturation and brightness (hue unchanged); grey/dark jerseys
    have no reliable hue and are kept as measured.
    """
    pixel = cv2.cvtColor(np.array([[lab]], np.uint8), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV).astype(np.float32)
    if hsv[0, 0, 1] >= 50:
        hsv[..., 1] = np.minimum(255, hsv[..., 1] * 1.8)
        hsv[..., 2] = np.maximum(hsv[..., 2], 180)
    return tuple(int(c) for c in cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)[0, 0])


def assign_teams(frames, outlier_factor):
    """Split all 'player' objects into 2 teams with K-means on the jersey colour (Lab).

    A jersey far from both team centres (goalkeeper, referee, linesman) becomes "other" with its own colour.
    Sets "team" and "colour" on every player. Returns {team name: BGR colour}.
    """
    players = [o for objs in frames.values() for o in objs if o["name"] == "player" and o["jersey"]]
    if len(players) < 2:
        return {}
    data = np.array([o["jersey"] for o in players], np.float32)
    cv2.setRNGSeed(0)   # same teams (and colours) every time the parameters change
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, labels, centres = cv2.kmeans(data, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    dist = np.linalg.norm(data[:, None, :] - centres[None], axis=2)  # distance to both team centres
    typical = np.median(dist.min(axis=1))
    team_colours = {f"team {k + 1}": lab_to_bgr(c) for k, c in enumerate(centres)}
    for o, label, row in zip(players, labels.ravel(), dist):
        if row.min() > outlier_factor * typical:
            o["team"], o["colour"] = "other", lab_to_bgr(o["jersey"])
        else:
            o["team"] = f"team {label + 1}"
            o["colour"] = team_colours[o["team"]]
    return team_colours


def run_tracker(frames, p, fps):
    """Give every object a "track" id with ByteTrack (supervision), frame by frame in time order.

    ByteTrack only needs boxes + confidences, so it runs on the cached detections in a fraction of a second.
    Objects that ByteTrack does not (yet) follow keep track None.
    """
    with warnings.catch_warnings():   # ByteTrack is deprecated in supervision 0.28+ (moved to 'trackers')
        warnings.simplefilter("ignore", FutureWarning)
        import supervision as sv
        tracker = sv.ByteTrack(track_activation_threshold=p.track_activation, lost_track_buffer=p.lost_buffer,
                               minimum_matching_threshold=p.match_threshold, frame_rate=fps)
    for idx in sorted(frames):
        objs = frames[idx]
        if not objs:
            tracker.update_with_detections(sv.Detections.empty())   # empty frame: lost tracks still age
            continue
        dets = sv.Detections(xyxy=np.array([o["box"] for o in objs], np.float32),
                             confidence=np.array([o["conf"] for o in objs], np.float32),
                             class_id=np.zeros(len(objs), int),
                             data={"i": np.arange(len(objs))})   # to find our object back after tracking
        tracked = tracker.update_with_detections(dets)
        for i, track_id in zip(tracked.data["i"], tracked.tracker_id):
            objs[int(i)]["track"] = int(track_id)


def vote_teams(frames, team_colours):
    """Give every track the team it got most often: one wrong jersey measurement no longer flips the colour."""
    votes = {}
    for objs in frames.values():
        for o in objs:
            if o["track"] is not None and o["name"] == "player":
                votes.setdefault(o["track"], Counter())[o["team"]] += 1
    for objs in frames.values():
        for o in objs:
            if o["track"] in votes:
                o["team"] = votes[o["track"]].most_common(1)[0][0]
                if o["team"] in team_colours:
                    o["colour"] = team_colours[o["team"]]


def ball_candidates(detections, Hs, p):
    """{idx: ball} with the most confident ball per frame that passes the filters.

    Filters: confidence, size <= max_ball_size px, on the pitch, not on a painted spot.
    """
    raw = {}
    for idx, ds in detections.items():
        H = Hs[idx] if Hs else None
        balls = []
        for d in ds:
            x1, y1, x2, y2 = d["box"]
            if d["name"] != "ball" or d["conf"] < p.conf_ball or max(x2 - x1, y2 - y1) > p.max_ball_size:
                continue
            xy = to_pitch(d["box"], H, is_ball=True)
            if not on_pitch(xy, p.pitch_margin):
                continue
            if xy is not None and min(np.hypot(*np.subtract(xy, s)) for s in SPOTS) <= SPOT_RADIUS:
                continue
            balls.append({"px": np.array([(x1 + x2) / 2, (y1 + y2) / 2]), "box": d["box"],
                          "conf": d["conf"], "xy": xy, "kind": "detected"})
        if balls:
            raw[idx] = max(balls, key=lambda b: b["conf"])
    return raw


def confirm_balls(raw, p):
    """Keep a ball only if another frame within ball_confirm_frames saw a ball nearby (removes one-frame FPs)."""
    window, speed = p.ball_confirm_frames, p.ball_confirm_speed
    return {i: b for i, b in raw.items()
            if any(j != i and j in raw and np.linalg.norm(raw[j]["px"] - b["px"]) < speed * abs(j - i)
                   for j in range(i - window, i + window + 1))}


def interpolate_balls(kept, ids, Hs, max_gap):
    """Confirmed balls + linear interpolation (in image px) for gaps of at most max_gap frames."""
    track = dict(kept)
    found = sorted(kept)
    for a, b in zip(found[:-1], found[1:]):
        if b - a > max_gap:
            continue
        for i in ids:
            if a < i < b:
                t = (i - a) / (b - a)
                px = (1 - t) * kept[a]["px"] + t * kept[b]["px"]
                xy = project([px], Hs[i])[0] if Hs else None
                track[i] = {"px": px, "box": None, "conf": None, "xy": xy, "kind": "interpolated"}
    return track


class BallKalman:
    """Constant-velocity Kalman filter for the ball on the pitch.

    State x = [x, y, vx, vy] in metres and m/s, P = its uncertainty (4x4 covariance).
    predict(dt): the ball keeps its velocity; the uncertainty grows because of unknown
                 accelerations (kicks, bounces), with std accel_std.
    update(z):   combine the prediction with a measured position z (std meas_std):
                 the Kalman gain K decides how much we trust the measurement vs the prediction.
    """

    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])   # we only measure the position, not the velocity

    def __init__(self, xy, accel_std, meas_std):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])
        self.P = np.diag([meas_std ** 2, meas_std ** 2, 10.0 ** 2, 10.0 ** 2])  # speed unknown (~10 m/s)
        self.accel_var = accel_std ** 2
        self.R = np.eye(2) * meas_std ** 2

    def predict(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt                      # position += velocity * dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])  # effect of an acceleration
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * self.accel_var
        return self.x[:2].copy()

    def update(self, z):
        innovation = np.asarray(z) - self.H @ self.x      # measured - predicted position
        S = self.H @ self.P @ self.H.T + self.R           # uncertainty of that difference
        K = self.P @ self.H.T @ np.linalg.inv(S)          # Kalman gain
        self.x = self.x + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].copy()


def kalman_balls(kept, ids, Hs, p, fps):
    """Run BallKalman over the processed frames (fps = video fps): filtered position where the ball was seen,
    prediction for at most max_ball_gap frames after the last measurement.

    The filter is restarted after a longer gap, or when a measurement is more than KALMAN_GATE m
    from the prediction (a hard kick that the constant-velocity model cannot follow).
    """
    track, kf, last = {}, None, None
    for i in ids:
        measured = kept.get(i)
        if kf is not None and i - last > p.max_ball_gap:
            kf = None                                   # lost for too long: forget the old state
        if kf is None:
            if measured is None:
                continue
            kf, last, prev = BallKalman(measured["xy"], p.kalman_accel, p.kalman_meas), i, i
            track[i] = {**measured, "xy": kf.x[:2].copy()}
            continue
        predicted = kf.predict((i - prev) / fps)
        prev = i
        if measured is not None and np.linalg.norm(measured["xy"] - predicted) > KALMAN_GATE:
            kf = BallKalman(measured["xy"], p.kalman_accel, p.kalman_meas)
            xy = kf.x[:2].copy()
        elif measured is not None:
            xy = kf.update(measured["xy"])
        else:
            xy = predicted
        if measured is not None:
            last = i
        px = project([xy], np.linalg.inv(Hs[i]))[0]   # back into the image, to draw it on the video
        if measured is not None:
            track[i] = {**measured, "xy": xy, "px": px, "kind": "detected"}
        else:
            track[i] = {"px": px, "box": None, "conf": None, "xy": xy, "kind": "predicted"}
    return track


def track_histories(frames):
    """{track id: [(idx, pitch xy, image foot px)]} in time order, for trails and statistics."""
    tracks = {}
    for idx in sorted(frames):
        for o in frames[idx]:
            if o["track"] is not None:
                x1, _, x2, y2 = o["box"]
                tracks.setdefault(o["track"], []).append((idx, o["xy"], ((x1 + x2) / 2, y2)))
    return tracks


def postprocess(detections, Hs, p, video_fps):
    """All group-2 steps on the cached detections. Hs = per-frame homographies or None (no minimap).

    Returns {"frames": {idx: [person]}, "ball": {idx: ball}, "tracks", "team_colours", "notes", "ball_counts"}.
    """
    frames_in = detections["frames"]
    stride = detections["meta"]["stride"]
    ids = sorted(frames_in)
    notes = []
    frames = filter_people(frames_in, Hs, p)

    team_colours = {}
    if p.teams and "player" in detections["meta"]["classes"] and "player" in p.classes:
        team_colours = assign_teams(frames, p.outlier_factor)
    elif p.teams:
        notes.append("Teamkleuren overgeslagen: dit model heeft geen samengevoegde 'player'-klasse.")

    if p.tracking:
        run_tracker(frames, p, video_fps / stride)   # ByteTrack sees one update per processed frame
        if team_colours:
            vote_teams(frames, team_colours)

    raw = ball_candidates(frames_in, Hs, p)
    kept = confirm_balls(raw, p)
    smoothing = p.ball_smoothing
    if smoothing == "kalman" and not Hs:
        notes.append("Kalman werkt in meters op het veld: zonder homografie wordt interpolatie gebruikt.")
        smoothing = "interpolation"
    if smoothing == "kalman":
        ball = kalman_balls(kept, ids, Hs, p, video_fps)   # dt = frame difference / video fps
    elif smoothing == "interpolation":
        ball = interpolate_balls(kept, ids, Hs, p.max_ball_gap)
    else:
        ball = dict(kept)
    if "ball" not in p.classes:
        ball = {}

    return {"frames": frames, "ball": ball, "tracks": track_histories(frames), "team_colours": team_colours,
            "notes": notes, "ball_counts": {"raw": len(raw), "confirmed": len(kept), "track": len(ball)}}


def statistics(result, n_frames):
    """Numbers for the 'Statistieken' tab."""
    per_class = Counter(o["name"] for objs in result["frames"].values() for o in objs)
    per_team = Counter(o["team"] for objs in result["frames"].values() for o in objs)
    ball = result["ball_counts"]
    return {
        "per_class": dict(per_class),
        "per_team": dict(per_team),
        "ball_pct": {k: 100 * v / max(n_frames, 1) for k, v in ball.items()},
        "tracks": len(result["tracks"]),
        "long_tracks": sum(len(h) >= 10 for h in result["tracks"].values()),
    }


# ======================================================================
# Group 3: rendering
# ======================================================================

@dataclass(frozen=True)
class DisplayParams:
    pitch_lines: bool = True    # projected pitch lines on the video (shows the homography quality)
    boxes: bool = True
    labels: bool = False        # class + confidence above every box
    track_ids: bool = True
    trail: int = 0              # number of processed frames of history per track (0 = off)
    mini_scale: int = 8         # minimap pixels per metre
    camera_view: bool = True    # the part of the pitch the camera sees, on the minimap


@lru_cache(maxsize=4)
def base_minimap(scale):
    """Empty minimap at a given scale (drawn once per scale)."""
    return draw_minimap(scale, MINI_MARGIN)


def mini_px(xy, scale):
    """Pitch metres -> minimap pixel (clipped so far-away points cannot overflow cv2)."""
    return to_minimap_px(np.clip(xy, -50, 200), scale, MINI_MARGIN)


def draw_pitch_lines(frame, H):
    """Project the pitch markings into the image: they must lie on the real lines."""
    H_inv = np.linalg.inv(H)
    for line in LINES:
        dense = np.concatenate([np.linspace(a, b, 20) for a, b in zip(line[:-1], line[1:])])
        img = np.clip(project(dense, H_inv), -10000, 10000).astype(np.int32)
        cv2.polylines(frame, [img], False, (255, 255, 0), 2, cv2.LINE_AA)


def draw_camera_view(mini, H, width, height, scale):
    """Quadrilateral on the minimap = the image corners projected onto the pitch."""
    corners = project([[0, 0], [width, 0], [width, height], [0, height]], H)
    poly = np.array([mini_px(c, scale) for c in corners], np.int32)
    cv2.polylines(mini, [poly], True, (255, 255, 0), 1, cv2.LINE_AA)


def draw_trails(frame, mini, result, idx, H, d):
    """Last d.trail positions of every track that is visible in this frame."""
    visible = {o["track"]: o["colour"] for o in result["frames"].get(idx, []) if o["track"] is not None}
    for track_id, colour in visible.items():
        history = [h for h in result["tracks"][track_id] if h[0] <= idx][-d.trail:]
        if len(history) < 2:
            continue
        if mini is not None:
            pts = np.array([mini_px(xy, d.mini_scale) for _, xy, _ in history], np.int32)
            cv2.polylines(mini, [pts], False, colour, 2, cv2.LINE_AA)
        if H is not None:   # project the pitch history through the current H: correct while the camera pans
            img = project([xy for _, xy, _ in history], np.linalg.inv(H))
        else:               # no homography: raw image positions (shift when the camera moves)
            img = np.array([px for _, _, px in history])
        cv2.polylines(frame, [np.clip(img, -10000, 10000).astype(np.int32)], False, colour, 2, cv2.LINE_AA)


def draw_person(frame, mini, o, d):
    """Box (+ label / track id) on the video, a dot on the minimap."""
    x1, y1, x2, y2 = (int(v) for v in o["box"])
    if d.boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)   # black edge: grey boxes stay visible on grass
        cv2.rectangle(frame, (x1, y1), (x2, y2), o["colour"], 2)
    text = []
    if d.labels:
        has_team = o["name"] == "player" and o["team"] != "player"   # "team 1" / "team 2" / "other"
        text.append(f"{o['team'] if has_team else o['name']} {o['conf']:.2f}")
    if d.track_ids and o["track"] is not None:
        text.append(f"#{o['track']}")
    if text:
        cv2.putText(frame, " ".join(text), (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, " ".join(text), (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if mini is not None and o["xy"] is not None:
        p = mini_px(o["xy"], d.mini_scale)
        r = max(3, d.mini_scale)
        cv2.circle(mini, p, r, o["colour"], -1, cv2.LINE_AA)
        cv2.circle(mini, p, r, (255, 255, 255), 2, cv2.LINE_AA)
        if d.track_ids and o["track"] is not None:
            cv2.putText(mini, str(o["track"]), (p[0] + r, p[1] - r), cv2.FONT_HERSHEY_SIMPLEX,
                        0.035 * d.mini_scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_ball(frame, mini, ball, d):
    """Detected ball: red box + white dot. Interpolated / predicted: red circle + open red dot."""
    if ball["box"] is not None:
        x1, y1, x2, y2 = (int(v) for v in ball["box"])
        cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), BALL_COLOUR, 2)
    else:
        cv2.circle(frame, tuple(int(v) for v in ball["px"]), 10, BALL_COLOUR, 2, cv2.LINE_AA)
    if mini is not None and ball["xy"] is not None:
        p = mini_px(ball["xy"], d.mini_scale)
        r = max(3, int(0.7 * d.mini_scale))
        if ball["kind"] == "detected":
            cv2.circle(mini, p, r, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(mini, p, r + 1, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.circle(mini, p, r, BALL_COLOUR, 2, cv2.LINE_AA)


def render_frame(frame, idx, result, H, d):
    """Draw one frame. Returns (video image, minimap image or None when there is no homography)."""
    frame = frame.copy()
    mini = base_minimap(d.mini_scale).copy() if H is not None else None
    if H is not None and d.pitch_lines:
        draw_pitch_lines(frame, H)
    if H is not None and d.camera_view:
        draw_camera_view(mini, H, frame.shape[1], frame.shape[0], d.mini_scale)
    if d.trail > 1:
        draw_trails(frame, mini, result, idx, H, d)
    for o in result["frames"].get(idx, []):
        draw_person(frame, mini, o, d)
    if idx in result["ball"]:
        draw_ball(frame, mini, result["ball"][idx], d)
    cv2.putText(frame, f"frame {idx}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6)
    cv2.putText(frame, f"frame {idx}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    return frame, mini


def side_by_side(frame, mini, height=OUT_H):
    """Video | minimap at the same height, with an even width and height (needed for H.264)."""
    def resize(img):
        w = int(round(img.shape[1] * height / img.shape[0]))
        return cv2.resize(img, (w, height), interpolation=cv2.INTER_AREA)
    combined = resize(frame) if mini is None else np.hstack([resize(frame), resize(mini)])
    h, w = combined.shape[:2]
    return combined[:h - h % 2, :w - w % 2]


def render_video(video, ids, result, Hs, d, out_path, fps, progress=None):
    """Side-by-side video of the processed frames, encoded as H.264 (plays in the browser, unlike mp4v)."""
    import imageio_ffmpeg
    writer = None
    tmp = Path(out_path).with_suffix(".tmp.mp4")
    for n, (idx, frame) in enumerate(iter_frames(video, ids)):
        image, mini = render_frame(frame, idx, result, Hs[idx] if Hs else None, d)
        combined = side_by_side(image, mini)
        if writer is None:
            writer = imageio_ffmpeg.write_frames(
                str(tmp), (combined.shape[1], combined.shape[0]), fps=fps, codec="libx264",
                pix_fmt_out="yuv420p", macro_block_size=1, ffmpeg_log_level="error",
                output_params=["-movflags", "+faststart"])   # moov atom first: the browser can start playing
            writer.send(None)                                # start the generator
        writer.send(np.ascontiguousarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)))
        if progress:
            progress(n + 1, len(ids))
    if writer is not None:
        writer.close()
        tmp.replace(out_path)
    return Path(out_path)
