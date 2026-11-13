"""Tracking over time: players (ByteTrack + majority vote) and the ball (Kalman filter on the pitch).

Per-frame detection alone flickers: a player is PSG in one frame and Inter in the next,
a white boot is a "ball" for a few frames, possession jumps around. Here we connect the
detections over time:

- track_people(): ByteTrack (ultralytics) gives every person a track id. A Kalman filter
  predicts where each box goes, and boxes are matched to the predictions by IoU.
- vote_groups(): every track gets ONE group (team / keeper / referee): the class with the
  highest summed confidence over the whole track. One wrong frame no longer flips the team.
- track_ball(): constant-velocity Kalman filter on the ball position in metres. Each frame
  only ball candidates close to the predicted position are accepted (gate), so a white boot
  far from the ball's path is ignored. Gaps are filled by linear interpolation on the pitch.
- stable_possession(): possession only changes when another team has the ball for
  POSSESSION_SWITCH frames in a row.
"""

from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

FPS = 30

# ByteTrack settings (see ultralytics/cfg/trackers/bytetrack.yaml)
TRACKER = dict(track_high_thresh=0.3,   # first matching round: confident boxes
               track_low_thresh=0.1,    # second round: weak boxes may continue an existing track
               new_track_thresh=0.3,    # a new track needs at least this confidence
               track_buffer=30,         # frames a lost track is kept (occlusion of up to 1 s)
               match_thresh=0.8,
               fuse_score=True)

# Ball Kalman filter (units: metres, frames)
BALL_GATE = 2.0           # m: a candidate must be this close to the prediction ...
BALL_GATE_GROWTH = 1.0    # ... plus this much per frame without a measurement (the ball may have sped up)
BALL_MAX_MISSED = 15      # frames without a measurement before the track is dropped
BALL_START_CONF = 0.25    # confidence needed to start a new ball track
BALL_SWITCH_CONF = 0.45   # a candidate outside the gate this confident replaces a weak (< BALL_WEAK) match
BALL_WEAK = 0.2
POSSESSION_SWITCH = 5     # frames another team must have the ball before possession changes
POSSESSION_TIMEOUT = 15   # frames without any team near the ball before possession becomes "-"


class _Boxes:
    """Minimal detections object for BYTETracker: conf, cls, xywh arrays + boolean indexing."""

    def __init__(self, conf, cls, xywh):
        self.conf, self.cls, self.xywh = conf, cls, xywh

    def __getitem__(self, mask):
        return _Boxes(self.conf[mask], self.cls[mask], self.xywh[mask])

    def __len__(self):
        return len(self.conf)


def track_people(detections, is_person):
    """Adds "track" (id) to every person detection that ByteTrack confirms; others get None.

    detections: {frame idx: [{"name", "box", "conf", ...}]}, frames in order.
    """
    tracker = BYTETracker(SimpleNamespace(**TRACKER))
    names = sorted({d["name"] for ds in detections.values() for d in ds})
    for idx in sorted(detections):
        people = [d for d in detections[idx] if is_person(d)]
        for d in people:
            d["track"] = None
        if not people:
            tracker.update(_Boxes(np.zeros(0), np.zeros(0), np.zeros((0, 4))))
            continue
        box = np.array([d["box"] for d in people], np.float32)
        xywh = np.column_stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2,
                                box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]])
        conf = np.array([d["conf"] for d in people], np.float32)
        cls = np.array([names.index(d["name"]) for d in people], np.float32)
        for row in tracker.update(_Boxes(conf, cls, xywh)):
            people[int(row[7])]["track"] = int(row[4])  # row = x1, y1, x2, y2, id, score, cls, detection index


def vote_groups(detections, group_of):
    """Every track gets the group with the highest summed confidence over all its frames.

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


class _BallKalman:
    """Constant-velocity Kalman filter, state = (x, y, vx, vy) in metres and metres/frame."""

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


def track_ball(candidates, frame_ids, max_gap):
    """Follow the ball through the clip.

    candidates: {frame idx: [(pitch xy, conf, image box)]} (already filtered on size / pitch / spots).
    Returns {frame idx: (pitch xy, image box or None, interpolated?)}.
    """
    measured = {}      # frame -> (xy, box, segment id): the candidate the filter accepted
    kf, missed, segment = None, 0, 0
    for idx in frame_ids:
        cands = candidates.get(idx, [])
        if kf is not None:
            pred = kf.predict()
            gate = BALL_GATE + BALL_GATE_GROWTH * missed
            dist = [np.hypot(*np.subtract(c[0], pred)) for c in cands]
            inside = [(d, c) for d, c in zip(dist, cands) if d <= gate]
            best = max(cands, key=lambda c: c[1]) if cands else None
            if inside:
                _, match = min(inside, key=lambda t: t[0])  # closest to where the ball should be
                if match[1] < BALL_WEAK and best[1] >= BALL_SWITCH_CONF and best is not match:
                    kf, segment, match = _BallKalman(best[0]), segment + 1, best  # weak match, strong ball elsewhere
                else:
                    kf.update(match[0])
                measured[idx] = (match[0], match[2], segment)
                missed = 0
                continue
            missed += 1
            if missed > BALL_MAX_MISSED:
                kf = None
        if kf is None:  # (re)start on a confident candidate
            starts = [c for c in cands if c[1] >= BALL_START_CONF]
            if starts:
                c = max(starts, key=lambda c: c[1])
                kf, missed, segment = _BallKalman(c[0]), 0, segment + 1
                measured[idx] = (c[0], c[2], segment)

    track = {i: (np.asarray(xy), box, False) for i, (xy, box, _) in measured.items()}
    ids = sorted(measured)
    for a, b in zip(ids[:-1], ids[1:]):  # fill gaps inside one segment
        if measured[a][2] == measured[b][2] and 1 < b - a <= max_gap:
            for i in range(a + 1, b):
                t = (i - a) / (b - a)
                track[i] = ((1 - t) * np.asarray(measured[a][0]) + t * np.asarray(measured[b][0]), None, True)
    return track


def stable_possession(raw):
    """Smooth per-frame possession {frame: team or None} -> {frame: team or None} (hysteresis)."""
    stable, candidate, run, since_seen, out = None, None, 0, 0, {}
    for idx in sorted(raw):
        team = raw[idx]
        if team is None:
            since_seen += 1
            if since_seen > POSSESSION_TIMEOUT:
                stable = None
        else:
            since_seen = 0
            if stable is None:
                stable = team
            elif team == stable:
                run = 0
            else:
                run = run + 1 if team == candidate else 1
                candidate = team
                if run >= POSSESSION_SWITCH:
                    stable, run = team, 0
        out[idx] = stable
    return out
