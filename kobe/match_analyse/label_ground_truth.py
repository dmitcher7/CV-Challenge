"""Label ground truth by hand -> data/ground_truth.json (input of evaluate_system.py).

    venv\\Scripts\\python kobe\\match_analyse\\label_ground_truth.py              (every 30th frame)
    venv\\Scripts\\python kobe\\match_analyse\\label_ground_truth.py --every 60
    venv\\Scripts\\python kobe\\match_analyse\\label_ground_truth.py --frames 0 90 180

Per frame: click the FEET (bottom-centre) of every person, set the ball and the possession, confirm.
Clicks are converted to metres with the per-frame homography (data/keypoints.json); pixels and
metres are both saved. Everything is saved after every change; a restart resumes at the first
frame that is not confirmed yet. Persons keep their id when you click within 5 m of a person
of the previous labelled frame (needed for the ID-switch metric; click in the same order is not required).

KEYS
    a / b / k / r   choose the role of the next clicks: team A / team B / keeper (of B) / referee (sticky)
    left click      add a person (feet) with the chosen role
    right click     delete the nearest person
    g               the NEXT click is the ball
    v               no ball visible in this frame
    1 / 2 / 0       possession: team A / team B / nobody
    u               undo the last change
    c               clear this frame
    space / enter   confirm the frame and go to the next one (needs ball + possession)
    s               skip this frame (not confirmed, not used in the evaluation)
    p               previous frame
    q / esc         quit (auto-saved)

Format of ground_truth.json: see evaluation/gt_format.py and data/ground_truth.example.json.
"""

import argparse
import copy

import cv2
import numpy as np

from core.homography import Homographies
from core.paths import DATA_DIR, VIDEO
from core.video import read_frame, video_info
from evaluation.gt_format import (add_person, can_confirm, empty_frame, load_gt, pixel_to_metre,
                                  remove_nearest_person, save_gt, set_ball)

WINDOW = "label ground truth"
GT_JSON = DATA_DIR / "ground_truth.json"
BAR_H = 150
ROLE_KEYS = {ord("a"): "A", ord("b"): "B", ord("k"): "K", ord("r"): "R"}
POSSESSION_KEYS = {ord("1"): "A", ord("2"): "B", ord("0"): "-"}
COLOURS = {"A": (255, 120, 0), "B": (0, 0, 255), "K": (0, 200, 255), "R": (0, 220, 0)}   # BGR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(VIDEO))
    parser.add_argument("--gt", default=str(GT_JSON))
    parser.add_argument("--every", type=int, default=30, help="label every n-th frame")
    parser.add_argument("--frames", type=int, nargs="*", help="explicit frames instead of --every")
    args = parser.parse_args()

    last = video_info(args.video)["frames"] - 1
    todo = args.frames or list(range(0, last + 1, args.every))
    frames = load_gt(args.gt)
    hs = Homographies()
    state = {"pos": 0, "role": "A", "ball_mode": False, "msg": "", "history": []}
    state["pos"] = next((k for k, i in enumerate(todo) if not frames.get(i, {}).get("done")), 0)

    def rec_now():
        return frames.setdefault(todo[state["pos"]], empty_frame())

    def snapshot():
        state["history"].append((todo[state["pos"]], copy.deepcopy(rec_now())))

    def commit():
        save_gt(args.gt, {i: r for i, r in frames.items() if r["persons"] or r["ball_visible"] is not None
                          or r["possession"] is not None or r["done"]})

    def on_mouse(event, x, y, flags, param):
        if y >= param["frame_h"]:
            return
        idx, rec = todo[state["pos"]], rec_now()
        if event == cv2.EVENT_LBUTTONDOWN:
            snapshot()
            if state["ball_mode"]:
                set_ball(frames, idx, (x, y), hs)
                state["ball_mode"] = False
            else:
                add_person(frames, idx, (x, y), state["role"], hs)
            rec["done"] = False
            commit()
        elif event == cv2.EVENT_RBUTTONDOWN:
            snapshot()
            if remove_nearest_person(rec, (x, y)):
                rec["done"] = False
                commit()
            else:
                state["history"].pop()

    def render(frame):
        idx, rec = todo[state["pos"]], rec_now()
        h, w = frame.shape[:2]
        canvas = np.zeros((h + BAR_H, w, 3), np.uint8)
        canvas[:h] = frame
        for p in rec["persons"]:
            c, (x, y) = COLOURS[p["role"]], (int(p["px"][0]), int(p["px"][1]))
            cv2.circle(canvas, (x, y), 6, c, 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{p['role']}{p['id']}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
            cv2.putText(canvas, f"{p['role']}{p['id']}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1)
        if rec["ball"]:
            bx, by = int(rec["ball"]["px"][0]), int(rec["ball"]["px"][1])
            cv2.circle(canvas, (bx, by), 9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "ball", (bx + 10, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        ball = ("click the ball now" if state["ball_mode"] else "set" if rec["ball"]
                else "none visible" if rec["ball_visible"] is False else "NOT SET")
        n_done = sum(bool(frames.get(i, {}).get("done")) for i in todo)
        lines = [
            (f"Frame {idx}  ({state['pos'] + 1}/{len(todo)}, {n_done} confirmed)   "
             f"{'CONFIRMED' if rec['done'] else 'open'}", (255, 255, 255)),
            (f"role: {state['role']}   persons: {len(rec['persons'])}   ball: {ball}   "
             f"possession: {rec['possession'] if rec['possession'] is not None else 'NOT SET'}",
             COLOURS[state["role"]]),
            ("a/b/k/r role (A, B, keeper, referee)   click feet: add   right click: delete   g+click: ball   v: no ball",
             (180, 180, 180)),
            ("1/2/0 possession A/B/nobody   u undo   c clear   space/enter confirm+next   s skip   p previous   q quit",
             (180, 180, 180)),
            (state["msg"], (80, 80, 255)),
        ]
        for k, (text, colour) in enumerate(lines):
            cv2.putText(canvas, text, (10, h + 28 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
        return canvas

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    loaded, frame = None, None
    while True:
        idx = todo[state["pos"]]
        if idx != loaded:
            frame = read_frame(args.video, idx)
            loaded = idx
            state["history"].clear()
            cv2.setMouseCallback(WINDOW, on_mouse, {"frame_h": frame.shape[0]})
            cv2.resizeWindow(WINDOW, 1440, int(1440 * (frame.shape[0] + BAR_H) / frame.shape[1]))
        cv2.imshow(WINDOW, render(frame))

        key = cv2.waitKey(20) & 0xFF
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1 or key in (ord("q"), 27):
            break
        rec = rec_now()
        if key in ROLE_KEYS:
            state["role"], state["ball_mode"], state["msg"] = ROLE_KEYS[key], False, ""
        elif key == ord("g"):
            state["ball_mode"] = True
        elif key == ord("v"):
            snapshot()
            set_ball(frames, idx, None, hs)
            rec["done"] = False
            commit()
        elif key in POSSESSION_KEYS:
            snapshot()
            rec["possession"], rec["done"] = POSSESSION_KEYS[key], False
            commit()
        elif key == ord("u") and state["history"]:
            i, old = state["history"].pop()
            frames[i] = old
            commit()
        elif key == ord("c"):
            snapshot()
            frames[idx] = empty_frame()
            commit()
        elif key in (13, 32):
            ok, msg = can_confirm(rec)
            state["msg"] = msg
            if ok:
                rec["done"] = True
                commit()
                state["pos"] = min(state["pos"] + 1, len(todo) - 1)
        elif key == ord("s"):
            state["pos"] = min(state["pos"] + 1, len(todo) - 1)
            state["msg"] = ""
        elif key == ord("p"):
            state["pos"] = max(state["pos"] - 1, 0)
            state["msg"] = ""

    cv2.destroyAllWindows()
    commit()
    print(f"{sum(bool(frames.get(i, {}).get('done')) for i in todo)}/{len(todo)} frames confirmed -> {args.gt}")


if __name__ == "__main__":
    main()
