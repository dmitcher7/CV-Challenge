"""The whole analysis in one call: video -> detection -> tracking -> pitch positions -> ball -> possession.

    from core.pipeline import run, save_tracks_csv, save_frames_csv
    result = run(start=0, end=120, stride=2)

DATA CONTRACT (other tools build on this; keep the names exactly)
-----------------------------------------------------------------
Coordinates are in METRES on the pitch: origin = far touchline x left goal line, x 0..105 along the
touchline, y 0..68 across (same as data/keypoints.json). time_s = frame / fps of the video.

output/tracks_<model>.csv   one row per person per processed frame (only persons that have a track id):
    frame,time_s,track_id,group,team,x_m,y_m,conf,box_x1,box_y1,box_x2,box_y2   (box in image pixels)
    group = player_A / player_B / player_B_goal_keeper / referee (or team 1 / team 2 / other for a COCO model)
    team  = A / B / -     (A = PSG = player_A, B = Inter = player_B + its keeper, - = referee / other)
    x_m,y_m = the feet (bottom-centre of the box) on the pitch; conf = detection confidence

output/frames_<model>.csv   one row per processed frame:
    frame,time_s,ball_x_m,ball_y_m,ball_state,possession_team,possession_track_id
    ball_state = detected / interpolated / refined / none (ball_x_m, ball_y_m are empty when none)
                 refined = found by YOLO on a crop around the expected position (only with ball_refine=True,
                 see ball_refine.py); counts as a found ball like detected
    possession_team = A / B / -   (smoothed, see possession.py)
    possession_track_id = track id of the player nearest to the ball while his team has possession, else empty

run() returns a Result:
    result.ids                 processed frame indices (list)
    result.fps, result.stride, result.video, result.weights, result.homographies
    result.people[frame]       [{"track_id", "group", "team", "x_m", "y_m", "conf", "box" (px xyxy)}]
    result.ball[frame]         {"x_m", "y_m", "state" ("detected"/"interpolated"/"refined"), "conf", "box"},
                               missing = no ball
    result.possession[frame]   {"team" ("A"/"B"/"-"), "track_id" (int or None), "xy" (ball (x, y) or None)}
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core import ball_refine as refine
from core.detection import detect, load_model
from core.homography import Homographies
from core.paths import DEFAULT_WEIGHTS, VIDEO
from core.pitch import KEYPOINTS, LENGTH, WIDTH
from core.possession import raw_possession, scaled, stable_possession
from core.possession import POSSESSION_SWITCH, POSSESSION_TIMEOUT
from core.teams import GENERIC_PEOPLE, assign_kmeans, group_of, is_ball, is_person, team_of
from core.tracking import (BALL_START_CONF, BALL_WEAK, TRACK_LOW_THRESH, remove_duplicates, track_ball,
                           track_people, vote_groups)
from core.video import frame_ids, video_info

PITCH_MARGIN = 3.0       # metres outside the lines that still count as "on the pitch"
MAX_BALL_SIZE = 40       # px: a bigger "ball" box is something else (a head, a sign)
MAX_BALL_GAP = 15        # frames (0.5 s) that are filled by interpolation
# White painted spots look like a ball: a "ball" this close (m) to one is ignored
SPOTS = [KEYPOINTS[n] for n in ("centre", "L_penalty_spot", "R_penalty_spot")]
SPOT_RADIUS = 1.0
# Defaults chosen on the owner's ground truth (WIJZIGINGEN.md, "Resultaten GPU-run"): inference at 1280 finds the
# small ball in more frames than the training size 960 (19/21 vs 16/21 GT balls, same false balls), and a ball
# threshold of 0.2 removes low-confidence static false balls (conf 0.11-0.14) that 1280 adds.
DEFAULT_IMGSZ = 1280
DEFAULT_BALL_CONF = 0.2


@dataclass
class Result:
    ids: list
    fps: float
    stride: int
    video: str
    weights: str
    homographies: object
    people: dict = field(default_factory=dict)
    ball: dict = field(default_factory=dict)
    possession: dict = field(default_factory=dict)


def on_pitch(xy, margin=PITCH_MARGIN):
    """True when a pitch position is inside the lines plus margin."""
    return -margin <= xy[0] <= LENGTH + margin and -margin <= xy[1] <= WIDTH + margin


def ball_candidate(d, idx, hs, min_conf):
    """(pitch xy, conf, box) of a ball detection, or None when it is too weak, too big, off the pitch or on a spot."""
    x1, y1, x2, y2 = d["box"]
    if not is_ball(d) or d["conf"] < min_conf or max(x2 - x1, y2 - y1) > MAX_BALL_SIZE:
        return None
    xy = hs.image_to_pitch([((x1 + x2) / 2, (y1 + y2) / 2)], idx)[0]
    if not on_pitch(xy) or min(np.hypot(*np.subtract(xy, s)) for s in SPOTS) <= SPOT_RADIUS:
        return None
    return xy, d["conf"], tuple(d["box"])


def refine_windows(result):
    """Crops that ball_refine=True needs for this result (computed from a run WITHOUT refinement)."""
    info = video_info(result.video)
    known = {i: (b["x_m"], b["y_m"]) for i, b in result.ball.items()}
    return refine.crop_windows(known, result.ids, result.homographies, info["width"], info["height"])


def run(video=VIDEO, weights=DEFAULT_WEIGHTS, imgsz=DEFAULT_IMGSZ, iou=0.7, start=0, end=None, stride=1,
        conf=0.3, ball_conf=DEFAULT_BALL_CONF, model=None, progress=None, ball_refine=False, refine_progress=None, nms=None):
    """Analyse frames start..end (inclusive, every stride-th) of the video. Returns a Result (see above).

    conf: confidence to start a person track; ball_conf: minimum confidence of a ball candidate;
    iou: NMS threshold of YOLO; model: an already loaded YOLO model (else loaded from weights when needed).
    nms: None = YOLO default (NMS with iou); False = the NMS-free one2one head (load a fresh model for it:
    an ultralytics predictor keeps its head mode between calls).
    ball_refine: look for the ball again in crops of the frames without a ball (slow the first time, then
    cached; see ball_refine.py); refine_progress(done, total) is called per new crop.
    """
    info = video_info(video)
    if info is None:
        raise FileNotFoundError(f"cannot open video {video}")
    end = info["frames"] - 1 if end is None else min(end, info["frames"] - 1)
    ids = frame_ids(start, end, stride)
    hs = Homographies()

    frames = detect(video, weights, imgsz, iou, start, end, stride, model=model, progress=progress, nms=nms)
    detections = {f["frame"]: f["detections"] for f in frames}

    # People: feet on the pitch, duplicates removed
    people = {}
    for idx in ids:
        kept = []
        for d in detections.get(idx, []):
            if not is_person(d) or d["conf"] < TRACK_LOW_THRESH:
                continue
            x1, y1, x2, y2 = d["box"]
            d["xy"] = hs.image_to_pitch([((x1 + x2) / 2, y2)], idx)[0]
            if on_pitch(d["xy"]):
                kept.append(d)
        people[idx] = remove_duplicates(kept)

    # A model that only knows "person": the team comes from the jersey colour
    if any(d["name"] in GENERIC_PEOPLE for ds in people.values() for d in ds):
        assign_kmeans(d for ds in people.values() for d in ds)

    track_people(people, conf=conf, stride=stride)
    vote_groups(people, group_of)

    result = Result(ids=ids, fps=info["fps"], stride=stride, video=str(video), weights=str(weights),
                    homographies=hs)
    for idx in ids:
        result.people[idx] = [
            {"track_id": d["track"], "group": d["group"], "team": team_of(d["group"]), "x_m": float(d["xy"][0]),
             "y_m": float(d["xy"][1]), "conf": d["conf"], "box": d["box"]}
            for d in people[idx] if d.get("track") is not None]

    # Ball: candidates in metres -> Kalman track with gaps interpolated in metres
    candidates, confs = {}, {}
    for idx in ids:
        for d in detections.get(idx, []):
            c = ball_candidate(d, idx, hs, ball_conf)
            if c is not None:
                candidates.setdefault(idx, []).append(c)
                confs[c[2]] = c[1]
    track = track_ball(candidates, ids, MAX_BALL_GAP, BALL_START_CONF, BALL_WEAK, stride=stride)
    for idx, (xy, box, interpolated) in track.items():
        result.ball[idx] = {"x_m": float(xy[0]), "y_m": float(xy[1]), "box": box,
                            "state": "interpolated" if interpolated else "detected",
                            "conf": None if interpolated else confs.get(box)}
    if ball_refine:
        refine_ball(result, weights, stride, model, refine_progress)

    # Possession: nearest team player to the ball, smoothed
    raw, nearest = {}, {}
    for idx in ids:
        ball = result.ball.get(idx)
        raw[idx], nearest[idx] = raw_possession(result.people[idx], (ball["x_m"], ball["y_m"]) if ball else None)
    stable = stable_possession(raw, scaled(POSSESSION_SWITCH, stride), scaled(POSSESSION_TIMEOUT, stride))
    for idx in ids:
        ball = result.ball.get(idx)
        result.possession[idx] = {
            "team": stable[idx] or "-", "track_id": nearest[idx] if raw[idx] == stable[idx] else None,
            "xy": (ball["x_m"], ball["y_m"]) if ball else None}
    return result


def refine_ball(result, weights, stride, model=None, progress=None):
    """Add confirmed crop hits (state "refined") and fill the short gaps next to them (see ball_refine.py)."""
    windows = refine_windows(result)
    hits, boxes = {}, {}
    for idx, ds in refine.detect_crops(result.video, weights, windows, model=model, progress=progress).items():
        cands = [c for c in (ball_candidate(d, idx, result.homographies, refine.REFINE_CONFIRM_CONF) for d in ds) if c]
        if cands:
            hits[idx] = max(cands, key=lambda c: c[1])
    known = {i: (b["x_m"], b["y_m"]) for i, b in result.ball.items() if b["state"] == "detected"}
    accepted = refine.confirm_hits({i: (h[0], h[1]) for i, h in hits.items()}, known, stride)
    for idx in accepted:
        xy, conf, box = hits[idx]
        result.ball[idx] = {"x_m": float(xy[0]), "y_m": float(xy[1]), "box": box, "state": "refined", "conf": conf}
    points = {**known, **{i: hits[i][0] for i in accepted}}
    for idx, xy in refine.fill_between(points, accepted, result.ids, MAX_BALL_GAP).items():
        if idx not in result.ball:
            result.ball[idx] = {"x_m": float(xy[0]), "y_m": float(xy[1]), "box": None, "state": "interpolated",
                                "conf": None}


def save_tracks_csv(result, path):
    """output/tracks_<model>.csv: one row per tracked person per processed frame (see the contract above)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # box_x1..box_y2 (image pixels) were added later at the end, so readers by column name keep working
        writer.writerow(["frame", "time_s", "track_id", "group", "team", "x_m", "y_m", "conf",
                         "box_x1", "box_y1", "box_x2", "box_y2"])
        for idx in result.ids:
            for p in result.people[idx]:
                box = [round(float(v), 1) for v in p["box"]] if p.get("box") is not None else [""] * 4
                writer.writerow([idx, round(idx / result.fps, 3), p["track_id"], p["group"], p["team"],
                                 round(p["x_m"], 2), round(p["y_m"], 2), round(p["conf"], 3), *box])
    return Path(path)


def save_frames_csv(result, path):
    """output/frames_<model>.csv: one row per processed frame (see the contract above)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "ball_x_m", "ball_y_m", "ball_state", "possession_team",
                         "possession_track_id"])
        for idx in result.ids:
            ball, owner = result.ball.get(idx), result.possession[idx]
            writer.writerow([idx, round(idx / result.fps, 3),
                             round(ball["x_m"], 2) if ball else "", round(ball["y_m"], 2) if ball else "",
                             ball["state"] if ball else "none", owner["team"],
                             "" if owner["track_id"] is None else owner["track_id"]])
    return Path(path)


def possession_share(result):
    """{team: percentage of the processed frames with possession}, e.g. {"A": 40.0, "B": 60.0}."""
    teams = [result.possession[i]["team"] for i in result.ids]
    total = sum(t != "-" for t in teams)
    return {t: 100 * teams.count(t) / total for t in ("A", "B") if teams.count(t)} if total else {}


__all__ = ["run", "Result", "refine_windows", "save_tracks_csv", "save_frames_csv", "possession_share", "load_model", "on_pitch"]
