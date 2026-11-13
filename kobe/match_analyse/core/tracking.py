"""Tracking over time: players (ByteTrack + majority vote) and the ball (Kalman filter on the pitch).

Per-frame detection alone flickers: a player is PSG in one frame and Inter in the next,
a white boot is a "ball" for a few frames. Here we connect the detections over time:

- remove_duplicates(): of two overlapping person boxes only the most confident is kept.
- track_people(): ByteTrack (ultralytics BYTETracker, the only tracker in this folder) gives every
  person a track id. A Kalman filter predicts where each box goes; boxes are matched by IoU.
- vote_groups(): every track gets ONE group (team / keeper / referee): the class with the
  highest summed confidence over the whole track. One wrong frame no longer flips the team.
- track_ball(): constant-velocity Kalman filter on the ball position in METRES. Each frame only ball
  candidates close to the predicted position are accepted (gate), so a white boot far from the ball's
  path is ignored. Gaps of at most max_gap frames are filled by linear interpolation in metres.
"""

from collections import Counter, defaultdict
from math import ceil
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

# ByteTrack settings (see ultralytics/cfg/trackers/bytetrack.yaml)
TRACK_LOW_THRESH = 0.1   # second round: weak boxes may continue an existing track
TRACK_BUFFER = 30        # frames a lost track is kept (occlusion of up to 1 s)
MATCH_THRESH = 0.8       # max IoU cost for a match
DUPLICATE_IOU = 0.6      # two person boxes overlapping more than this are one person

# Ball Kalman filter (units: metres, frames)
BALL_GATE = 2.0           # m: a candidate must be this close to the prediction ...
BALL_GATE_GROWTH = 1.5    # ... plus this much per frame without a measurement (a kicked ball does ~1.4 m/frame;
                          # 1.0 lost the kick at frames 204-218, see experiments/ball_prediction/results.md)
BALL_MAX_MISSED = 30      # frames without a measurement before the track is dropped (was 15; the gap
                          # itself is only filled up to max_gap, so this does not invent balls)
BALL_START_CONF = 0.25    # confidence needed to start a new ball track
BALL_SWITCH_CONF = 0.45   # a candidate outside the gate this confident replaces a weak (< BALL_WEAK) match
BALL_WEAK = 0.2           # lower start_conf / weak for a model that is unsure about the ball
BALL_RESTART_CONF = 0.4   # nothing inside the gate: a candidate this confident outside it may restart the track ...
BALL_RESTART_LOOKAHEAD = 3   # ... if a candidate in the next this-many frames continues it ...
BALL_RESTART_SPEED = 1.5  # ... at most this many metres per frame away (one confident outlier never hijacks)


def iou(a, b):
    """IoU of two xyxy boxes."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def remove_duplicates(people):
    """One box per person: of two overlapping boxes (IoU > DUPLICATE_IOU) keep the most confident.

    A fine-tuned model sometimes puts two boxes with different classes on one player
    (e.g. player_A and player_B), which would give two dots on the minimap.
    """
    kept = []
    for d in sorted(people, key=lambda d: -d["conf"]):
        if all(iou(d["box"], k["box"]) <= DUPLICATE_IOU for k in kept):
            kept.append(d)
    return kept


class _Boxes:
    """Minimal detections object for BYTETracker: conf, cls, xywh arrays + boolean indexing."""

    def __init__(self, conf, cls, xywh):
        self.conf, self.cls, self.xywh = conf, cls, xywh

    def __getitem__(self, mask):
        return _Boxes(self.conf[mask], self.cls[mask], self.xywh[mask])

    def __len__(self):
        return len(self.conf)


def track_people(detections, conf=0.3, stride=1):
    """Adds "track" (id) to every person detection that ByteTrack confirms; others get None.

    detections: {frame idx: [{"box", "conf", ...}]} with only persons (e.g. after remove_duplicates).
    conf: minimum confidence to start a track (and for the first matching round).
    stride: distance between the frames, so the lost-track buffer stays about 1 s.
    """
    args = SimpleNamespace(track_high_thresh=conf, track_low_thresh=TRACK_LOW_THRESH, new_track_thresh=conf,
                           track_buffer=max(1, ceil(TRACK_BUFFER / stride)),  # counted in processed frames
                           match_thresh=MATCH_THRESH, fuse_score=True)
    tracker = BYTETracker(args)
    for idx in sorted(detections):
        people = detections[idx]
        for d in people:
            d["track"] = None
        if not people:
            tracker.update(_Boxes(np.zeros(0), np.zeros(0), np.zeros((0, 4))))
            continue
        box = np.array([d["box"] for d in people], np.float32)
        xywh = np.column_stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2,
                                box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]])
        confs = np.array([d["conf"] for d in people], np.float32)
        for row in tracker.update(_Boxes(confs, np.zeros(len(people), np.float32), xywh)):
            people[int(row[7])]["track"] = int(row[4])  # row = x1, y1, x2, y2, id, score, cls, detection index


def vote_groups(detections, group_of):
    """Every track gets the group with the highest summed confidence over all its frames.

    detections: {frame idx: [detection dicts with "track" and "conf"]}. group_of(d) -> group of one detection.
    Sets "group" on every tracked detection and returns {track id: (group, share of the votes)}.
    """
    votes = defaultdict(Counter)
    for ds in detections.values():
        for d in ds:
            if d.get("track") is not None:
                votes[d["track"]][group_of(d)] += d["conf"]
    result = {}
    for track, counter in votes.items():
        group, score = counter.most_common(1)[0]
        result[track] = (group, score / sum(counter.values()))
    for ds in detections.values():
        for d in ds:
            if d.get("track") is not None:
                d["group"] = result[d["track"]][0]
    return result


class BallKalman:
    """Constant-velocity Kalman filter, state = (x, y, vx, vy) in metres and metres/step."""

    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    Q = np.diag([0.05, 0.05, 0.1, 0.1])   # process noise: the ball can change speed (kicks, bounces)
    R = np.diag([0.3, 0.3])               # measurement noise: detection + homography error (m^2)

    def __init__(self, xy):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])
        self.P = np.diag([0.5, 0.5, 1.0, 1.0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def update(self, xy):
        y = np.asarray(xy) - self.Hm @ self.x
        S = self.Hm @ self.P @ self.Hm.T + self.R
        K = self.P @ self.Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.Hm) @ self.P


def interpolate_ball(measured, frame_ids, max_gap):
    """Fill the gaps between measured ball positions by linear interpolation in metres.

    measured: {frame: (xy metres, image box, segment id)}; only gaps inside one segment, spanning at most
    max_gap frames, are filled (only for frames in frame_ids).
    Returns {frame: (xy, box or None, interpolated?)}.
    """
    track = {i: (np.asarray(xy, float), box, False) for i, (xy, box, _) in measured.items()}
    ids = sorted(measured)
    for a, b in zip(ids[:-1], ids[1:]):
        if measured[a][2] == measured[b][2] and 1 < b - a <= max_gap:
            for i in frame_ids:
                if a < i < b:
                    t = (i - a) / (b - a)
                    xy = (1 - t) * np.asarray(measured[a][0], float) + t * np.asarray(measured[b][0], float)
                    track[i] = (xy, None, True)
    return track


def confirmed(cand, idx, candidates, later_ids, lookahead=BALL_RESTART_LOOKAHEAD, speed=BALL_RESTART_SPEED):
    """True when a candidate in one of the later processed frames (at most `lookahead` video frames after idx)
    lies within `speed` m per frame of cand: the candidate is a moving ball, not a one-frame outlier."""
    for j in later_ids:
        if j - idx > lookahead:
            break
        if any(np.hypot(*np.subtract(c[0], cand[0])) <= speed * (j - idx) for c in candidates.get(j, [])):
            return True
    return False


def track_ball(candidates, frame_ids, max_gap, start_conf=BALL_START_CONF, weak=BALL_WEAK, stride=1):
    """Follow the ball through the clip.

    candidates: {frame idx: [(pitch xy in metres, conf, image box)]} (already filtered on size / pitch / spots).
    frame_ids: the processed frames in order; stride: distance between them (the gate grows per frame).
    start_conf / weak: lower them for a model that is unsure about the ball (low confidences overall).
    Returns {frame idx: (pitch xy, image box or None, interpolated?)}.
    """
    max_missed = max(1, ceil(BALL_MAX_MISSED / stride))
    measured = {}      # frame -> (xy, box, segment id): the candidate the filter accepted
    kf, missed, segment = None, 0, 0
    for n, idx in enumerate(frame_ids):
        cands = candidates.get(idx, [])
        if kf is not None:
            pred = kf.predict()
            gate = BALL_GATE + BALL_GATE_GROWTH * stride * missed
            dist = [np.hypot(*np.subtract(c[0], pred)) for c in cands]
            inside = [(d, c) for d, c in zip(dist, cands) if d <= gate]
            best = max(cands, key=lambda c: c[1]) if cands else None
            if inside:
                _, match = min(inside, key=lambda t: t[0])  # closest to where the ball should be
                if match[1] < weak and best[1] >= BALL_SWITCH_CONF and best is not match:
                    kf, segment, match = BallKalman(best[0]), segment + 1, best  # weak match, strong ball elsewhere
                else:
                    kf.update(match[0])
                measured[idx] = (match[0], match[2], segment)
                missed = 0
                continue
            later = frame_ids[n + 1:n + 1 + BALL_RESTART_LOOKAHEAD]
            strong = [c for c in cands if c[1] >= BALL_RESTART_CONF and confirmed(c, idx, candidates, later)]
            if strong:  # the ball went somewhere the gate could not follow (a kick): restart there
                c = max(strong, key=lambda c: c[1])
                kf, missed, segment = BallKalman(c[0]), 0, segment + 1
                measured[idx] = (c[0], c[2], segment)
                continue
            missed += 1
            if missed > max_missed:
                kf = None
        if kf is None:  # (re)start on a confident candidate
            starts = [c for c in cands if c[1] >= start_conf]
            if starts:
                c = max(starts, key=lambda c: c[1])
                kf, missed, segment = BallKalman(c[0]), 0, segment + 1
                measured[idx] = (c[0], c[2], segment)
    return interpolate_ball(measured, frame_ids, max_gap)
