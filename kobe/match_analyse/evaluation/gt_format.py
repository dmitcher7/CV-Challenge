"""Ground-truth file data/ground_truth.json: reading, writing and the pixel -> metre conversion.

Format (pitch metres like data/keypoints.json: x 0..105 along, y 0..68 across)
{
 "pitch": "...",
 "frames": {
  "120": {
    "done": true,                  # the frame was confirmed in the label tool
    "persons": [ {"id": 3,         # GT person id, carried over between labelled frames (may be null)
                  "role": "A",     # A = team A, B = team B, K = keeper (of team B), R = referee
                  "px": [812.0, 655.5],   # clicked feet, image pixels
                  "m": [61.2, 40.3]} ],   # the same point in metres (per-frame homography)
    "ball_visible": true,          # false = "no ball visible"
    "ball": {"px": [..], "m": [..]} | null,
    "possession": "A"              # A / B / "-" (nobody)
  }
 }
}
"""

import json
from pathlib import Path

import numpy as np

ROLES = ("A", "B", "K", "R")
POSSESSION = ("A", "B", "-")
ID_MATCH_M = 5.0      # a click within this distance (m) of a person of the previous labelled frame keeps his id


def empty_frame():
    return {"done": False, "persons": [], "ball_visible": None, "ball": None, "possession": None}


def load_gt(path):
    """{frame idx (int): frame record}. A missing file gives {}."""
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(i): rec for i, rec in data.get("frames", {}).items()}


def save_gt(path, frames):
    data = {"pitch": "metres, origin = far touchline x left goal line, x 0..105 along, y 0..68 across",
            "roles": "A = team A, B = team B, K = keeper, R = referee",
            "frames": {str(i): frames[i] for i in sorted(frames)}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)


def pixel_to_metre(homographies, frame_idx, px):
    """One image point (x, y) -> [x_m, y_m] with the per-frame homography."""
    return [round(float(v), 2) for v in homographies.image_to_pitch([px], frame_idx)[0]]


def previous_persons(frames, idx):
    """Persons of the nearest earlier labelled frame (for id carry-over), else []."""
    earlier = [i for i in frames if i < idx and frames[i].get("persons")]
    return frames[max(earlier)]["persons"] if earlier else []


def next_free_id(frames):
    ids = [p["id"] for rec in frames.values() for p in rec["persons"] if p.get("id") is not None]
    return max(ids, default=0) + 1


def assign_id(m, previous, used, new_id, max_dist=ID_MATCH_M):
    """Id for a person clicked at metre position m: that of the nearest previous person that is
    not yet used in this frame and within max_dist, else new_id."""
    best, best_d = None, max_dist
    for p in previous:
        if p.get("id") is None or p["id"] in used:
            continue
        d = float(np.hypot(p["m"][0] - m[0], p["m"][1] - m[1]))
        if d <= best_d:
            best, best_d = p["id"], d
    return best if best is not None else new_id


def add_person(frames, idx, px, role, homographies):
    """Add a clicked person (feet at image point px) to frame idx; returns the new person dict."""
    rec = frames.setdefault(idx, empty_frame())
    m = pixel_to_metre(homographies, idx, px)
    used = {p["id"] for p in rec["persons"]}
    pid = assign_id(m, previous_persons(frames, idx), used, max(next_free_id(frames), 1))
    person = {"id": pid, "role": role, "px": [round(float(px[0]), 1), round(float(px[1]), 1)], "m": m}
    rec["persons"].append(person)
    return person


def set_ball(frames, idx, px, homographies):
    """Set the ball at image point px (None = no ball visible)."""
    rec = frames.setdefault(idx, empty_frame())
    if px is None:
        rec["ball_visible"], rec["ball"] = False, None
    else:
        rec["ball_visible"] = True
        rec["ball"] = {"px": [round(float(px[0]), 1), round(float(px[1]), 1)],
                       "m": pixel_to_metre(homographies, idx, px)}


def remove_nearest_person(rec, px, max_px=40):
    """Delete the person whose feet are nearest to image point px (within max_px). Returns True if removed."""
    if not rec["persons"]:
        return False
    d = [float(np.hypot(p["px"][0] - px[0], p["px"][1] - px[1])) for p in rec["persons"]]
    k = int(np.argmin(d))
    if d[k] > max_px:
        return False
    del rec["persons"][k]
    return True


def can_confirm(rec):
    """(ok, message): a frame needs a ball decision and a possession before it counts as done."""
    if rec["ball_visible"] is None:
        return False, "set the ball (g + click) or press v (no ball visible)"
    if rec["possession"] is None:
        return False, "set possession: 1 = A, 2 = B, 0 = nobody"
    return True, ""
