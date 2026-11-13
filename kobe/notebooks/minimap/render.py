"""Drawing for 02_minimap_video.ipynb: boxes on the frame, the minimap and the info bar on top.

Only presentation lives here; the analysis (detection, tracking, ball, possession) is in the notebook.
"""

from collections import Counter

import cv2
import numpy as np

from pitch import LINES, draw_minimap, to_minimap_px

FPS = 30
OUT_H = 720                 # height of the video part of the output
HEADER_H = 115              # height of the info bar above it
MINI_SCALE, MINI_MARGIN = 10, 4
BALL_TRAIL = 20             # frames of ball history drawn on the minimap

# Colours (BGR) and names per class (labeling protocol: A = dark blue = PSG, B = yellow = Inter)
COLOURS = {"player_A": (120, 40, 10), "player_B": (0, 220, 255), "player_B_goal_keeper": (0, 140, 255),
           "referee": (30, 30, 30)}
LABELS = {"player_A": "PSG (A)", "player_B": "Inter (B)", "player_B_goal_keeper": "Inter keeper",
          "referee": "referee", "A": "PSG", "B": "Inter"}
GROUPS = ["player_A", "player_B", "player_B_goal_keeper", "referee"]  # order in the info bar
KEEPERS = {"player_B_goal_keeper"}

MINIMAP = draw_minimap(MINI_SCALE, MINI_MARGIN)


def label(group):
    return LABELS.get(group, group)


def mini_px(xy):
    return to_minimap_px(xy, MINI_SCALE, MINI_MARGIN)


def _dot(img, xy, colour, radius, filled=True):
    if filled:
        cv2.circle(img, mini_px(xy), radius, colour, -1, cv2.LINE_AA)
    cv2.circle(img, mini_px(xy), radius, (255, 255, 255) if filled else colour, 2, cv2.LINE_AA)


def _put(img, text, x, y, colour=(255, 255, 255), scale=0.8, thickness=2):
    """Text at (x, y); returns the x where the next text can start."""
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)
    return x + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]


def _info_bar(width, idx, last_idx, is_keyframe, people, ball_state, ball_conf, owner, share):
    bar = np.full((HEADER_H, width, 3), 30, np.uint8)
    counts = Counter(p["group"] for p in people)

    # Row 1: where are we, possession so far
    keyframe = "   keyframe (clicked)" if is_keyframe else ""
    _put(bar, f"frame {idx}/{last_idx}   t = {idx / FPS:5.2f} s{keyframe}", 20, 40, scale=0.9)
    if share:
        text = "possession so far: " + "   ".join(f"{label(t)} {p:.0f}%" for t, p in share.items())
        _put(bar, text, width - 20 - cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0], 40)

    # Row 2: what is on the pitch in this frame
    x, y = 20, 92
    for g in GROUPS:
        cv2.circle(bar, (x + 10, y - 9), 10, COLOURS[g], -1, cv2.LINE_AA)
        cv2.circle(bar, (x + 10, y - 9), 10, (255, 255, 255), 1, cv2.LINE_AA)
        x = _put(bar, f"{label(g)}: {counts.get(g, 0)}", x + 28, y) + 40
    keeper = any(g in KEEPERS for g in counts)
    x = _put(bar, f"keeper: {'yes' if keeper else 'no'}", x, y, (0, 220, 0) if keeper else (0, 0, 230)) + 40
    text = {"detected": f"ball: yes (conf {ball_conf:.2f})", "interpolated": "ball: yes (interpolated)",
            "none": "ball: no"}[ball_state]
    colour = {"detected": (0, 220, 0), "interpolated": (0, 200, 255), "none": (0, 0, 230)}[ball_state]
    x = _put(bar, text, x, y, colour) + 40
    _put(bar, f"possession: {label(owner) if owner else '-'}", x, y)
    return bar


def render_frame(frame, idx, last_idx, homographies, people, ball, ball_conf, owner, owner_xy, share):
    """One output frame: info bar on top, the video frame (left) and the minimap (right).

    people: [{"group", "xy", "box", "track"}]; ball: {frame: (xy, box or None, interpolated)}.
    """
    frame = frame.copy()
    h, w = frame.shape[:2]

    # Pitch lines projected into the image: they must lie on the real lines
    for line in LINES:
        dense = np.concatenate([np.linspace(a, b, 20) for a, b in zip(line[:-1], line[1:])])
        img = homographies.pitch_to_image(dense, idx)
        cv2.polylines(frame, [np.clip(img, -10000, 10000).astype(np.int32)], False, (255, 255, 0), 2, cv2.LINE_AA)

    mini = MINIMAP.copy()
    # Part of the pitch the camera sees
    corners = homographies.image_to_pitch([[0, 0], [w, 0], [w, h], [0, h]], idx)
    cv2.polylines(mini, [np.array([mini_px(c) for c in np.clip(corners, -50, 200)], np.int32)], True,
                  (255, 255, 0), 1, cv2.LINE_AA)

    # Ball trail (older = darker)
    trail = [ball[i][0] for i in range(idx - BALL_TRAIL, idx + 1) if i in ball]
    for k, (a, b) in enumerate(zip(trail[:-1], trail[1:])):
        shade = int(90 + 165 * (k + 1) / len(trail))
        cv2.line(mini, mini_px(a), mini_px(b), (shade, shade, shade), 2, cv2.LINE_AA)

    # People: box + track id on the frame, dot on the minimap
    for p in people:
        colour = COLOURS[p["group"]]
        x1, y1, x2, y2 = (int(v) for v in p["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)  # black edge: dark boxes stay visible on grass
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, str(p["track"]), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, str(p["track"]), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
        _dot(mini, p["xy"], colour, 10)
    if owner_xy is not None:  # ring around the player who has the ball
        cv2.circle(mini, mini_px(owner_xy), 16, (255, 255, 255), 2, cv2.LINE_AA)

    # Ball: red box on the frame; white dot (detected) or open red circle (interpolated) on the minimap
    state = "none"
    if idx in ball:
        xy, box, interpolated = ball[idx]
        state = "interpolated" if interpolated else "detected"
        if box is not None:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 255), 2)
        _dot(mini, xy, (0, 0, 255) if interpolated else (255, 255, 255), 7, filled=not interpolated)
        if not interpolated:
            cv2.circle(mini, mini_px(xy), 8, (0, 0, 0), 1, cv2.LINE_AA)

    left = cv2.resize(frame, (int(w * OUT_H / h), OUT_H))
    right = cv2.resize(mini, (int(mini.shape[1] * OUT_H / mini.shape[0]), OUT_H))
    body = np.hstack([left, right])
    bar = _info_bar(body.shape[1], idx, last_idx, idx in homographies.keyframes, people, state, ball_conf,
                    owner, share)
    return np.vstack([bar, body])
