"""Video with projected pitch lines and detections next to the 2D minimap.

Left:  the frame with the pitch lines projected from the minimap into the image.
       If they lie on the real lines, H is correct for that frame (also between keyframes).
Right: the minimap with one dot per person (bottom-centre of the box = the feet, the only
       point of a player that touches the pitch plane) and the ball with its trail.

Passes:
1. Detect: YOLO on every frame, all boxes >= DETECT_CONF (cached in output/, re-rendering is fast).
2. Post-process:
   - Duplicates: of two overlapping person boxes only the most confident is kept.
   - Tracking (tracking.py): ByteTrack gives every person a track id; every track gets one
     group (team / keeper / referee) by majority vote over all its frames.
   - Teams for a model that only knows 'person'/'player': jersey colour (torso, grass removed)
     split into 2 groups with K-means; players far from both (referee, keeper) are "other".
   - Ball: Kalman filter on the pitch; only candidates near the predicted position are accepted,
     gaps of up to MAX_BALL_GAP frames are interpolated (open circle on the minimap).
   - Possession: team of the nearest player within POSSESSION_RADIUS of the ball, smoothed so
     it only changes after a few frames.
   With --no-track all of this is done per frame (the old behaviour), to compare.
3. Render: info bar (frame, time, players per team, keeper, ball, possession), the frame with
   boxes and track ids, the minimap; plus a csv per frame and a summary of the clip.

Detections come from --weights: by default the pretrained COCO model (person, sports ball),
or a trained model (.pt, or an OpenVINO folder with --device intel:gpu).
With --lines-only no model is used (fast check of the homography alone).
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from homography import Homographies
from pitch import (FRAMES_DIR, HOMOGRAPHY_DIR, KEYPOINTS, LENGTH, LINES, OUTPUT_DIR, WIDTH, draw_minimap,
                   to_minimap_px)
from tracking import stable_possession, track_ball, track_people, vote_groups

MINI_SCALE, MINI_MARGIN = 10, 4
OUT_H = 720                 # height of the output video
PITCH_MARGIN = 3            # metres outside the lines that still count as "on the pitch"
DETECT_CONF = 0.1           # everything above this is cached; the tracker also uses the weak boxes
PLAYERS = {"person", "player"}           # classes whose team comes from the jersey colour
BALLS = {"ball", "sports ball"}
# Fixed colours (BGR) for classes of a trained model; other classes are ignored
COLOURS = {"player_A": (120, 40, 10), "player_B": (0, 220, 255), "goalkeeper": (0, 140, 255),
           "player_B_goal_keeper": (0, 140, 255), "referee": (30, 30, 30)}
MAX_BALL_SIZE = 40          # px; a bigger "ball" box is something else (a head, a sign)
BALL_CONFIRM = (5, 40)      # --no-track: a ball needs another ball within 5 frames and 40 px per frame
MAX_BALL_GAP = 15           # frames (0.5 s) that are filled by interpolation
# White painted spots look like a ball; a "ball" this close (m) to one is ignored
SPOTS = [KEYPOINTS[n] for n in ("centre", "L_penalty_spot", "R_penalty_spot")]
SPOT_RADIUS = 1.0
DUPLICATE_IOU = 0.6         # two person boxes overlapping more than this are one person
OUTLIER_FACTOR = 2.5        # jersey further than 2.5x the typical distance from both teams = other

# ---- Header and analysis ----
FPS = 30
HEADER_H = 115              # height of the info bar above the video
BALL_TRAIL = 20             # frames of ball history drawn on the minimap
POSSESSION_RADIUS = 2.0     # metres: the nearest player within this distance "has" the ball
# Names in the header (labeling protocol: A = dark blue = PSG, B = yellow = Inter)
GROUP_LABELS = {"player_A": "PSG (A)", "player_B": "Inter (B)", "player_B_goal_keeper": "Inter keeper",
                "goalkeeper": "keeper", "referee": "referee", "other": "other (ref/keeper)",
                "A": "PSG", "B": "Inter"}
GROUP_ORDER = ["player_A", "player_B", "player_B_goal_keeper", "goalkeeper", "referee",
               "team 1", "team 2", "other"]
KEEPERS = {"player_B_goal_keeper", "goalkeeper"}
# Team of each group, for possession; referees and "other" belong to no team
TEAM_OF = {"player_A": "A", "player_B": "B", "player_B_goal_keeper": "B", "team 1": "team 1",
           "team 2": "team 2"}

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default=str(HOMOGRAPHY_DIR / "pretrained" / "yolo26s.pt"))
parser.add_argument("--imgsz", type=int, default=1280, help="1280 finds the ball; 640 is ~3x faster")
parser.add_argument("--conf", type=float, default=0.3, help="confidence threshold for (new tracks of) people")
parser.add_argument("--ball-conf", type=float, default=0.1, help="confidence threshold for ball candidates")
parser.add_argument("--device", default="cpu",
                    help="cpu, or intel:gpu for an OpenVINO model (folder *_openvino_model) on the Intel Arc GPU")
parser.add_argument("--no-track", action="store_true", help="per-frame only (no ByteTrack / Kalman), to compare")
parser.add_argument("--redetect", action="store_true", help="ignore the cached detections")
parser.add_argument("--lines-only", action="store_true", help="no detections, only the projected lines")
args = parser.parse_args()

homographies = Homographies()
frames = sorted(FRAMES_DIR.glob("*.jpg"), key=lambda p: int(p.stem.split("_idx_")[1]))
frame_ids = [int(p.stem.split("_idx_")[1]) for p in frames]
OUTPUT_DIR.mkdir(exist_ok=True)


def on_pitch(point):
    return -PITCH_MARGIN <= point[0] <= LENGTH + PITCH_MARGIN and -PITCH_MARGIN <= point[1] <= WIDTH + PITCH_MARGIN


def is_person(d):
    return d["name"] in PLAYERS or d["name"] in COLOURS


def jersey_colour(frame, box):
    """Mean Lab colour of the torso (upper half of the body, middle half of the width), grass removed."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    crop = frame[int(y1 + 0.15 * h):int(y1 + 0.5 * h), int(x1 + 0.25 * w):int(x2 - 0.25 * w)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    grass = (hsv[:, 0] > 30) & (hsv[:, 0] < 90) & (hsv[:, 1] > 50)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)[~grass]
    return lab.mean(axis=0).tolist() if len(lab) >= 5 else None


def lab_to_bgr(lab):
    """Lab jersey colour -> vivid BGR colour for drawing.

    Measured jersey colours look dull (shadow, compression): coloured jerseys get more
    saturation and brightness (hue unchanged). Grey/dark jerseys (low saturation) have no
    reliable hue and are kept as measured; the white outline keeps them visible.
    """
    pixel = cv2.cvtColor(np.array([[lab]], np.uint8), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV).astype(np.float32)
    if hsv[0, 0, 1] >= 50:
        hsv[..., 1] = np.minimum(255, hsv[..., 1] * 1.8)
        hsv[..., 2] = np.maximum(hsv[..., 2], 180)
    return tuple(int(c) for c in cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)[0, 0])


# ---- Pass 1: detect (cached) ----
def detect():
    """{frame idx: [{"name", "box", "conf"}]} for every frame, from cache if possible."""
    cache = OUTPUT_DIR / f"detections_{Path(args.weights).stem}_{args.imgsz}_all.json"
    if cache.exists() and not args.redetect:
        print(f"using cached detections {cache.name} (--redetect to run YOLO again)")
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}

    from ultralytics import YOLO
    (HOMOGRAPHY_DIR / "pretrained").mkdir(exist_ok=True)
    model = YOLO(args.weights, task="detect")  # task needed for an exported (OpenVINO) model
    detections = {}
    for n, (idx, path) in enumerate(zip(frame_ids, frames)):
        frame = cv2.imread(str(path))
        res = model.predict(frame, imgsz=args.imgsz, conf=DETECT_CONF, device=args.device, verbose=False)[0]
        found = []
        for cls, conf, box in zip(res.boxes.cls.int().tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist()):
            name = model.names[cls]
            if name in BALLS or name in PLAYERS or name in COLOURS:
                found.append({"name": name, "box": [round(v, 1) for v in box], "conf": round(conf, 3)})
        detections[idx] = found
        if (n + 1) % 50 == 0:
            print(f"detected {n + 1}/{len(frames)} frames")
    cache.write_text(json.dumps(detections))
    return detections


def to_pitch(detection, idx):
    """Pitch position in metres: the feet for people, the centre for the ball."""
    x1, y1, x2, y2 = detection["box"]
    point = ((x1 + x2) / 2, (y1 + y2) / 2 if detection["name"] in BALLS else y2)
    return homographies.image_to_pitch([point], idx)[0]


def iou(a, b):
    """IoU of two xyxy boxes."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def remove_duplicates(detections):
    """One box per person: of two overlapping boxes (IoU > DUPLICATE_IOU) keep the most confident.

    A fine-tuned model sometimes puts two boxes with different classes on one player
    (e.g. player_A and player_B), which would give two dots on the minimap.
    """
    removed = 0
    for idx, ds in detections.items():
        kept = [d for d in ds if d["name"] in BALLS]
        for d in sorted((d for d in ds if d["name"] not in BALLS), key=lambda d: -d["conf"]):
            if all(iou(d["box"], k["box"]) <= DUPLICATE_IOU for k in kept if k["name"] not in BALLS):
                kept.append(d)
            else:
                removed += 1
        detections[idx] = kept
    print(f"removed {removed} duplicate boxes")
    return detections


# ---- Pass 2a: teams from jersey colours (only for 'person'/'player' models) ----
def assign_teams(detections):
    """Adds "team" and "colour" (BGR) to every 'person'/'player' on the pitch. Returns {team: colour}.

    Jersey colours are measured here (not in pass 1), so they can be tuned without running YOLO again.
    """
    players = []
    for idx, path in zip(frame_ids, frames):
        on = [d for d in detections.get(idx, []) if d["name"] in PLAYERS and on_pitch(to_pitch(d, idx))]
        if on:
            frame = cv2.imread(str(path))
            for d in on:
                d["jersey"] = jersey_colour(frame, d["box"])
                if d["jersey"] is not None:
                    players.append(d)
    if len(players) < 2:
        return {}
    data = np.array([d["jersey"] for d in players], np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, labels, centres = cv2.kmeans(data, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    dist = np.linalg.norm(data[:, None, :] - centres[None], axis=2)  # distance to both team centres
    typical = np.median(dist.min(axis=1))
    team_colours = {f"team {k + 1}": lab_to_bgr(c) for k, c in enumerate(centres)}
    for d, label, row in zip(players, labels.ravel(), dist):
        if row.min() > OUTLIER_FACTOR * typical:
            d["team"], d["colour"] = "other", lab_to_bgr(d["jersey"])  # referee, goalkeeper ...
        else:
            d["team"] = f"team {label + 1}"
            d["colour"] = team_colours[d["team"]]
    return team_colours


def group_of(d):
    """Group of one detection: the team from K-means, else the class of the model."""
    return d.get("team") or d["name"]


# ---- Pass 2b: ball ----
def ball_candidates(detections):
    """{frame idx: [(pitch xy, conf, box)]}: small, on the pitch, not on a painted spot."""
    candidates = {}
    for idx, ds in detections.items():
        for d in ds:
            if d["name"] not in BALLS or d["conf"] < args.ball_conf:
                continue
            if max(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]) > MAX_BALL_SIZE:
                continue
            xy = to_pitch(d, idx)
            if on_pitch(xy) and min(np.hypot(*np.subtract(xy, s)) for s in SPOTS) > SPOT_RADIUS:
                candidates.setdefault(idx, []).append((xy, d["conf"], d["box"]))
    return candidates


def ball_per_frame(candidates):
    """--no-track: most confident candidate per frame, confirmed by a nearby frame, short gaps filled."""
    raw = {i: max(c, key=lambda c: c[1]) for i, c in candidates.items()}
    window, speed = BALL_CONFIRM
    centre = {i: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) for i, (_, _, b) in raw.items()}
    kept = [i for i in raw if any(j != i and j in raw and np.linalg.norm(centre[j] - centre[i]) < speed * abs(j - i)
                                  for j in range(i - window, i + window + 1))]
    track = {i: (np.asarray(raw[i][0]), raw[i][2], False) for i in kept}
    ids = sorted(kept)
    for a, b in zip(ids[:-1], ids[1:]):
        if 1 < b - a <= MAX_BALL_GAP:
            for i in range(a + 1, b):
                t = (i - a) / (b - a)
                track[i] = ((1 - t) * track[a][0] + t * track[b][0], None, True)
    return track


# ---- Pass 3: per-frame summary ----
def frame_people(detections, idx):
    """[(group, colour, pitch xy, box, track id)] for every person on the pitch in one frame."""
    people = []
    for d in detections.get(idx, []):
        if not is_person(d):
            continue
        if args.no_track and d["conf"] < args.conf:
            continue
        if not args.no_track and d.get("track") is None:
            continue  # not (yet) a confirmed track
        group = d.get("group") or group_of(d)
        colour = COLOURS.get(group) or team_colours.get(group) or d.get("colour")
        if colour is None:
            continue  # a 'person' off the pitch (no jersey colour measured)
        xy = to_pitch(d, idx)
        if on_pitch(xy):
            people.append((group, colour, xy, d["box"], d.get("track")))
    return people


def possession(people, ball_xy):
    """(team, pitch xy) of the nearest player within POSSESSION_RADIUS metres of the ball, else (None, None)."""
    players = [(TEAM_OF[g], xy) for g, _, xy, _, _ in people if g in TEAM_OF]
    if ball_xy is None or not players:
        return None, None
    dist = [np.hypot(*np.subtract(xy, ball_xy)) for _, xy in players]
    i = int(np.argmin(dist))
    return players[i] if dist[i] <= POSSESSION_RADIUS else (None, None)


# ---- Pass 4: render ----
def label(group):
    return GROUP_LABELS.get(group, group)


def draw_dot(mini, xy, colour, radius, filled=True):
    p = to_minimap_px(xy, MINI_SCALE, MINI_MARGIN)
    if filled:
        cv2.circle(mini, p, radius, colour, -1, cv2.LINE_AA)
    cv2.circle(mini, p, radius, (255, 255, 255) if filled else colour, 2, cv2.LINE_AA)


def put(img, text, x, y, colour=(255, 255, 255), scale=0.8, thickness=2):
    """Text at (x, y); returns the x where the next text can start."""
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)
    return x + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]


def chip(img, x, y, colour, text):
    """Coloured dot + text in the header; returns the x of the next chip."""
    cv2.circle(img, (x + 10, y - 9), 10, colour, -1, cv2.LINE_AA)
    cv2.circle(img, (x + 10, y - 9), 10, (255, 255, 255), 1, cv2.LINE_AA)
    return put(img, text, x + 28, y) + 40


def header(width, idx, people, ball_state, conf, owner, share):
    """Info bar above the video: frame/time, counts per group, keeper, ball, possession."""
    bar = np.full((HEADER_H, width, 3), 30, np.uint8)
    counts = Counter(p[0] for p in people)
    colour_of = {p[0]: p[1] for p in people}

    # Row 1: where are we in the clip, and possession so far
    keyframe = "   keyframe (clicked)" if idx in homographies.keyframes else ""
    mode = "per frame (--no-track)" if args.no_track else "tracking"
    put(bar, f"frame {idx}/{frame_ids[-1]}   t = {idx / FPS:5.2f} s{keyframe}   [{mode}]", 20, 40, scale=0.9)
    if share:
        text = "possession so far: " + "   ".join(f"{label(t)} {p:.0f}%" for t, p in share.items())
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
        put(bar, text, width - 20 - size, 40)

    # Row 2: what is detected in this frame
    x, y = 20, 92
    for g in groups:
        x = chip(bar, x, y, colour_of.get(g, group_colour.get(g, (128, 128, 128))), f"{label(g)}: {counts.get(g, 0)}")
    if any(g in KEEPERS for g in groups):
        keeper = any(g in KEEPERS for g in counts)
        x = put(bar, f"keeper: {'yes' if keeper else 'no'}", x, y, (0, 220, 0) if keeper else (0, 0, 230)) + 40
    ball_text = {"detected": f"ball: yes (conf {conf:.2f})", "interpolated": "ball: yes (interpolated)",
                 "none": "ball: no"}[ball_state]
    ball_colour = {"detected": (0, 220, 0), "interpolated": (0, 200, 255), "none": (0, 0, 230)}[ball_state]
    x = put(bar, ball_text, x, y, ball_colour) + 40
    put(bar, f"possession: {label(owner) if owner else '-'}", x, y)
    return bar


# ---- Run all passes ----
detections = {} if args.lines_only else remove_duplicates(detect())
team_colours = assign_teams(detections) if detections else {}
if detections and not args.no_track:
    track_people(detections, is_person)
    votes = vote_groups(detections, group_of)
    unsure = sum(share < 0.8 for _, share in votes.values())
    print(f"tracking: {len(votes)} tracks, {unsure} with < 80% agreement on their group")

candidates = ball_candidates(detections) if detections else {}
if args.no_track:
    ball = ball_per_frame(candidates)
else:
    ball = track_ball(candidates, frame_ids, MAX_BALL_GAP)
print(f"ball: candidates in {len(candidates)} frames, "
      f"detected {sum(not b[2] for b in ball.values())}, interpolated {sum(b[2] for b in ball.values())}")
ball_conf = {i: max(c[1] for c in cs) for i, cs in candidates.items()}
for i, (xy, box, interpolated) in ball.items():  # confidence of the candidate that was used
    if box is not None:
        ball_conf[i] = next(c[1] for c in candidates[i] if c[2] == box)

people_per_frame = {idx: frame_people(detections, idx) for idx in frame_ids}
raw_owner = {}
owner_xy = {}
for idx in frame_ids:
    team, xy = possession(people_per_frame[idx], ball[idx][0] if idx in ball else None)
    raw_owner[idx], owner_xy[idx] = team, xy
owner_of = raw_owner if args.no_track else stable_possession(raw_owner)

# Groups that occur anywhere in the clip, in a fixed order, so the header does not jump around
seen = {p[0] for people in people_per_frame.values() for p in people}
groups = [g for g in GROUP_ORDER if g in seen] + sorted(seen - set(GROUP_ORDER))
group_colour = {**COLOURS, **team_colours}

minimap = draw_minimap(MINI_SCALE, MINI_MARGIN)
if args.lines_only:
    name = "lines_preview"
else:
    name = f"minimap_{Path(args.weights).stem}_{args.imgsz}" + ("_notrack" if args.no_track else "")
out_path = OUTPUT_DIR / f"{name}.mp4"
writer = None
stats = []           # one row per frame for the csv
owned = Counter()    # frames with possession per team (for the running percentage)

for n, (idx, path) in enumerate(zip(frame_ids, frames)):
    frame = cv2.imread(str(path))
    people = people_per_frame[idx]

    # ---- Pitch lines projected into the image ----
    for line in LINES:
        dense = np.concatenate([np.linspace(a, b, 20) for a, b in zip(line[:-1], line[1:])])
        img = homographies.pitch_to_image(dense, idx)
        cv2.polylines(frame, [np.clip(img, -10000, 10000).astype(np.int32)], False, (255, 255, 0), 2, cv2.LINE_AA)

    # ---- Visible area of the camera on the minimap ----
    mini = minimap.copy()
    h, w = frame.shape[:2]
    corners = homographies.image_to_pitch([[0, 0], [w, 0], [w, h], [0, h]], idx)
    poly = np.array([to_minimap_px(c, MINI_SCALE, MINI_MARGIN) for c in np.clip(corners, -50, 200)], np.int32)
    cv2.polylines(mini, [poly], True, (255, 255, 0), 1, cv2.LINE_AA)

    # ---- Ball trail: where the ball was in the last BALL_TRAIL frames (older = darker) ----
    trail = [ball[i][0] for i in range(idx - BALL_TRAIL, idx + 1) if i in ball]
    for k, (a, b) in enumerate(zip(trail[:-1], trail[1:])):
        shade = int(90 + 165 * (k + 1) / len(trail))
        cv2.line(mini, to_minimap_px(a, MINI_SCALE, MINI_MARGIN), to_minimap_px(b, MINI_SCALE, MINI_MARGIN),
                 (shade, shade, shade), 2, cv2.LINE_AA)

    # ---- Players / referees ----
    if idx in ball:
        ball_xy, state = ball[idx][0], "interpolated" if ball[idx][2] else "detected"
    else:
        ball_xy, state = None, "none"
    owner = owner_of[idx]
    for group, colour, xy, box, track in people:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)  # black edge: grey boxes stay visible on grass
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        if track is not None:
            cv2.putText(frame, str(track), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, str(track), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        draw_dot(mini, xy, colour, 10)
    if owner and raw_owner[idx] == owner:  # ring around the player who has the ball
        cv2.circle(mini, to_minimap_px(owner_xy[idx], MINI_SCALE, MINI_MARGIN), 16, (255, 255, 255), 2, cv2.LINE_AA)

    # ---- Ball ----
    if idx in ball:
        xy, box, interpolated = ball[idx]
        if box is not None:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 255), 2)
        draw_dot(mini, xy, (0, 0, 255) if interpolated else (255, 255, 255), 7, filled=not interpolated)
        if not interpolated:
            cv2.circle(mini, to_minimap_px(xy, MINI_SCALE, MINI_MARGIN), 8, (0, 0, 0), 1, cv2.LINE_AA)

    # ---- Possession so far (only frames in which a team has the ball count) ----
    if owner:
        owned[owner] += 1
    total = sum(owned.values())
    share = {t: 100 * c / total for t, c in sorted(owned.items())} if total else {}

    # ---- Header + side by side ----
    left = cv2.resize(frame, (int(w * OUT_H / h), OUT_H))
    right = cv2.resize(mini, (int(mini.shape[1] * OUT_H / mini.shape[0]), OUT_H))
    body = np.hstack([left, right])
    if not args.lines_only:
        bar = header(body.shape[1], idx, people, state, ball_conf.get(idx, 0.0), owner, share)
        body = np.vstack([bar, body])
    if writer is None:
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, body.shape[1::-1])
    writer.write(body)

    counts = Counter(p[0] for p in people)
    stats.append({"frame": idx, "time_s": round(idx / FPS, 2), **{label(g): counts.get(g, 0) for g in groups},
                  "ball": state,
                  "ball_x_m": round(float(ball_xy[0]), 2) if ball_xy is not None else "",
                  "ball_y_m": round(float(ball_xy[1]), 2) if ball_xy is not None else "",
                  "possession": label(owner) if owner else ""})
    if (n + 1) % 100 == 0:
        print(f"rendered {n + 1} frames")

writer.release()
print(f"Done: {len(frames)} frames -> {out_path}")

# ---- Per-frame table + summary of the whole clip ----
if not args.lines_only:
    csv_path = OUTPUT_DIR / f"{name}_frames.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(stats[0]))
        writer_csv.writeheader()
        writer_csv.writerows(stats)
    states = Counter(r["ball"] for r in stats)
    print(f"\nPer-frame details: {csv_path}")
    print(f"{'group':<20}{'avg per frame':>14}{'min':>6}{'max':>6}")
    for g in groups:
        values = [r[label(g)] for r in stats]
        print(f"{label(g):<20}{np.mean(values):>14.1f}{min(values):>6}{max(values):>6}")
    print(f"ball: detected {states['detected']}, interpolated {states['interpolated']}, "
          f"not found {states['none']} of {len(stats)} frames")
    changes = sum(1 for a, b in zip(stats[:-1], stats[1:]) if a["possession"] and b["possession"]
                  and a["possession"] != b["possession"])
    if owned:
        total = sum(owned.values())
        print("possession: " + ", ".join(f"{label(t)} {100 * c / total:.0f}%" for t, c in sorted(owned.items()))
              + f" (of {total} frames), {changes} changes of possession")
