"""Click pitch keypoints on keyframes of the clip -> keypoints.json.

    venv\\Scripts\\python kobe\\notebooks\\minimap\\click_keypoints.py            (every 30th frame)
    venv\\Scripts\\python kobe\\notebooks\\minimap\\click_keypoints.py --step 15
    venv\\Scripts\\python kobe\\notebooks\\minimap\\click_keypoints.py --frames 0 45 90

1. Choose a keypoint on the minimap below the frame (click it, or n / p).
2. Click that point in the frame. The magnifier (bottom right) helps to click precisely.
3. From 4 points on, the pitch lines are drawn over the frame: they must lie on
   the real lines. From 5 points on, every point shows its error in pixels;
   a red point (> 10 px) is a misclick or a wrong name.
4. Space / Enter: next keyframe. Everything is saved after every change.

Click points spread over the visible pitch (not all on one line), 5-8 per frame.
"""

import argparse
import json

import cv2
import numpy as np

from homography import fit, load_keypoints, project
from pitch import FRAMES_DIR, KEYPOINTS, KEYPOINTS_JSON, LINES, draw_minimap, to_minimap_px

WINDOW = "click keypoints"
BAR_H = 330                       # height of the panel below the frame
MINI_SCALE, MINI_MARGIN = 4, 4    # minimap: pixels per metre, margin in metres
MAG_HALF, MAG_ZOOM = 24, 6        # magnifier: half crop size and zoom factor
NAMES = list(KEYPOINTS)

parser = argparse.ArgumentParser()
parser.add_argument("--step", type=int, default=30, help="every n-th frame is a keyframe")
parser.add_argument("--frames", type=int, nargs="*", help="explicit keyframes instead of --step")
args = parser.parse_args()

frame_files = {int(p.stem.split("_idx_")[1]): p for p in FRAMES_DIR.glob("*.jpg")}
last = max(frame_files)
keyframes = args.frames or sorted(set(range(0, last + 1, args.step)) | {last})
points = load_keypoints()  # {frame idx: {name: [x, y]}}, also frames from earlier sessions


def save():
    data = {
        "pitch": "metres, origin = far touchline x left goal line, x 0..105 along, y 0..68 across",
        "frames": {str(i): points[i] for i in sorted(points) if points[i]},
    }
    KEYPOINTS_JSON.write_text(json.dumps(data, indent=1), encoding="utf-8")


# ---- State shared with the mouse callback ----
state = {"pos": 0, "selected": NAMES.index("R_box_T"), "mouse": None}
# Resume at the first keyframe that has fewer than 4 points
state["pos"] = next((k for k, i in enumerate(keyframes) if len(points.get(i, {})) < 4), 0)
minimap = draw_minimap(MINI_SCALE, MINI_MARGIN)
mini_origin = (10, 0)  # top-left of the minimap inside the bottom panel (set when drawing)


def current():
    idx = keyframes[state["pos"]]
    return idx, points.setdefault(idx, {})


def on_mouse(event, x, y, flags, param):
    frame_h = param["frame_h"]
    state["mouse"] = (x, y)
    idx, pts = current()
    if y < frame_h:  # click in the frame
        if event == cv2.EVENT_LBUTTONDOWN:
            pts[NAMES[state["selected"]]] = [float(x), float(y)]
            save()
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            nearest = min(pts, key=lambda n: np.hypot(pts[n][0] - x, pts[n][1] - y))
            if np.hypot(pts[nearest][0] - x, pts[nearest][1] - y) < 40:
                del pts[nearest]
                save()
    elif event == cv2.EVENT_LBUTTONDOWN:  # click in the panel: select the nearest keypoint on the minimap
        mx, my = x - mini_origin[0], y - frame_h - mini_origin[1]
        dist = {n: np.hypot(*np.subtract(to_minimap_px(KEYPOINTS[n], MINI_SCALE, MINI_MARGIN), (mx, my)))
                for n in NAMES}
        nearest = min(dist, key=dist.get)
        if dist[nearest] < 20:
            state["selected"] = NAMES.index(nearest)


def previous_h(idx):
    """H of the nearest earlier keyframe with >= 4 points (grey guide lines on a new frame)."""
    for i in sorted((i for i in points if i < idx), reverse=True):
        H, _ = fit(points[i])
        if H is not None:
            return H
    return None


def draw_lines(canvas, H, colour, thickness):
    """Draw all pitch lines, projected into the image with the inverse of H."""
    H_inv = np.linalg.inv(H)
    for line in LINES:
        dense = np.concatenate([np.linspace(a, b, 20) for a, b in zip(line[:-1], line[1:])])
        img = project(dense, H_inv)
        img = np.clip(img, -10000, 10000).astype(np.int32)  # far-away points can overflow
        cv2.polylines(canvas, [img], False, colour, thickness, cv2.LINE_AA)


def render(frame):
    global mini_origin
    idx, pts = current()
    frame_h, frame_w = frame.shape[:2]
    canvas = np.zeros((frame_h + BAR_H, frame_w, 3), np.uint8)
    canvas[:frame_h] = frame

    # ---- Pitch lines: fitted H, or the previous keyframe as a grey guide ----
    H, errors = fit(pts)
    if H is not None:
        draw_lines(canvas, H, (255, 255, 0), 2)
    else:
        guide = previous_h(idx)
        if guide is not None:
            draw_lines(canvas, guide, (150, 150, 150), 1)

    # ---- Clicked points (colour = error) ----
    for name, (x, y) in pts.items():
        err = errors.get(name, 0.0) if len(pts) >= 5 else 0.0
        colour = (0, 220, 0) if err < 5 else (0, 165, 255) if err < 10 else (0, 0, 255)
        cv2.circle(canvas, (int(x), int(y)), 6, colour, 2, cv2.LINE_AA)
        label = f"{name} {err:.1f}px" if len(pts) >= 5 else name
        cv2.putText(canvas, label, (int(x) + 9, int(y) - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(canvas, label, (int(x) + 9, int(y) - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1)

    # ---- Minimap with the keypoints ----
    mini = minimap.copy()
    for k, name in enumerate(NAMES):
        p = to_minimap_px(KEYPOINTS[name], MINI_SCALE, MINI_MARGIN)
        if name in pts:
            cv2.circle(mini, p, 6, (0, 220, 0), -1)
        else:
            cv2.circle(mini, p, 5, (200, 200, 200), 1)
        if k == state["selected"]:
            cv2.circle(mini, p, 11, (0, 0, 255), 3)
    mini_origin = (10, (BAR_H - mini.shape[0]) // 2)
    oy = frame_h + mini_origin[1]
    canvas[oy:oy + mini.shape[0], 10:10 + mini.shape[1]] = mini

    # ---- Info text ----
    if H is None:
        quality = f"{len(pts)} points: need at least 4"
    elif len(pts) < 5:
        quality = f"{len(pts)} points: add a 5th to check the error"
    else:
        worst = max(errors, key=errors.get)
        quality = (f"{len(pts)} points: mean error {np.mean(list(errors.values())):.1f} px, "
                   f"max {errors[worst]:.1f} px ({worst})")
    lines = [
        (f"Frame {idx}   (keyframe {state['pos'] + 1}/{len(keyframes)})", (255, 255, 255)),
        (f"Selected: {NAMES[state['selected']]}", (80, 80, 255)),
        (quality, (255, 255, 255)),
        ("", None),
        ("left click: place selected point   right click: delete nearest", (180, 180, 180)),
        ("click minimap or n / p: choose point   x: delete selected   c: clear frame", (180, 180, 180)),
        ("space / enter: next keyframe   a: previous   q / esc: quit (auto-saved)", (180, 180, 180)),
    ]
    tx = 10 + mini.shape[1] + 30
    for k, (text, colour) in enumerate(lines):
        if text:
            cv2.putText(canvas, text, (tx, frame_h + 45 + 40 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    # ---- Magnifier around the mouse ----
    if state["mouse"] and state["mouse"][1] < frame_h:
        mx, my = state["mouse"]
        padded = cv2.copyMakeBorder(frame, MAG_HALF, MAG_HALF, MAG_HALF, MAG_HALF, cv2.BORDER_CONSTANT)
        crop = padded[my:my + 2 * MAG_HALF, mx:mx + 2 * MAG_HALF]
        mag = cv2.resize(crop, None, fx=MAG_ZOOM, fy=MAG_ZOOM, interpolation=cv2.INTER_NEAREST)
        c = MAG_HALF * MAG_ZOOM
        cv2.line(mag, (c, 0), (c, mag.shape[0]), (0, 0, 255), 1)
        cv2.line(mag, (0, c), (mag.shape[1], c), (0, 0, 255), 1)
        oy = frame_h + (BAR_H - mag.shape[0]) // 2
        canvas[oy:oy + mag.shape[0], frame_w - mag.shape[1] - 10:frame_w - 10] = mag
    return canvas


cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)  # resizable; mouse coordinates stay in image pixels
loaded_idx, frame = None, None
while True:
    idx, _ = current()
    if idx != loaded_idx:
        frame = cv2.imread(str(frame_files[idx]))
        loaded_idx = idx
        cv2.setMouseCallback(WINDOW, on_mouse, {"frame_h": frame.shape[0]})
        if state["mouse"] is None:  # first frame: fit the window to ~75 % of a 1920 px screen
            cv2.resizeWindow(WINDOW, 1440, int(1440 * (frame.shape[0] + BAR_H) / frame.shape[1]))
    cv2.imshow(WINDOW, render(frame))

    key = cv2.waitKey(20) & 0xFF
    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        break  # window closed with the X
    if key in (ord("q"), 27):
        break
    if key in (13, 32):
        state["pos"] = min(state["pos"] + 1, len(keyframes) - 1)
    elif key == ord("a"):
        state["pos"] = max(state["pos"] - 1, 0)
    elif key == ord("n"):
        state["selected"] = (state["selected"] + 1) % len(NAMES)
    elif key == ord("p"):
        state["selected"] = (state["selected"] - 1) % len(NAMES)
    elif key == ord("x"):
        current()[1].pop(NAMES[state["selected"]], None)
        save()
    elif key == ord("c"):
        current()[1].clear()
        save()

cv2.destroyAllWindows()
save()
done = sum(len(points.get(i, {})) >= 4 for i in keyframes)
print(f"{done}/{len(keyframes)} keyframes have >= 4 points -> {KEYPOINTS_JSON}")
