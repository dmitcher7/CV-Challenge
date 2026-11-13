"""Drawing: camera view with boxes and projected pitch lines, next to the minimap, with an info bar on top.

Only presentation lives here; the analysis is in pipeline.py.
"""

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from core.pitch import LINES, draw_minimap, to_minimap_px
from core.video import iter_frames

OUT_H = 720                 # height of the video part of the output
HEADER_H = 116              # height of the info bar above it (even: H.264 yuv420p needs even sizes)
MINI_SCALE, MINI_MARGIN = 10, 4
BALL_TRAIL = 20             # frames of ball history drawn on the minimap

# Colours (BGR) and names per group (labeling protocol: A = dark blue = PSG, B = yellow = Inter)
COLOURS = {"player_A": (120, 40, 10), "player_B": (0, 220, 255), "player_B_goal_keeper": (0, 140, 255),
           "referee": (30, 30, 30), "team 1": (120, 40, 10), "team 2": (0, 220, 255), "other": (150, 150, 150)}
LABELS = {"player_A": "PSG (A)", "player_B": "Inter (B)", "player_B_goal_keeper": "Inter keeper",
          "referee": "referee", "team 1": "team 1", "team 2": "team 2", "other": "other", "A": "PSG", "B": "Inter"}
KEEPERS = {"player_B_goal_keeper", "goalkeeper"}

_MINIMAP = draw_minimap(MINI_SCALE, MINI_MARGIN)


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


def _info_bar(width, idx, last_idx, fps, is_keyframe, people, ball, owner, share):
    bar = np.full((HEADER_H, width, 3), 30, np.uint8)
    counts = Counter(p["group"] for p in people)

    # Row 1: where are we, possession so far
    keyframe = "   keyframe (clicked)" if is_keyframe else ""
    _put(bar, f"frame {idx}/{last_idx}   t = {idx / fps:5.2f} s{keyframe}", 20, 40, scale=0.9)
    if share:
        text = "possession so far: " + "   ".join(f"{label(t)} {p:.0f}%" for t, p in share.items())
        _put(bar, text, width - 20 - cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0], 40)

    # Row 2: what is on the pitch in this frame
    x, y = 20, 92
    for g in COLOURS:
        if g not in counts:
            continue
        cv2.circle(bar, (x + 10, y - 9), 10, COLOURS[g], -1, cv2.LINE_AA)
        cv2.circle(bar, (x + 10, y - 9), 10, (255, 255, 255), 1, cv2.LINE_AA)
        x = _put(bar, f"{label(g)}: {counts[g]}", x + 28, y) + 40
    state = ball["state"] if ball else "none"
    text = {"detected": f"ball: yes (conf {(ball or {}).get('conf') or 0:.2f})", "interpolated": "ball: interpolated",
            "refined": f"ball: crop (conf {(ball or {}).get('conf') or 0:.2f})", "none": "ball: no"}[state]
    colour = {"detected": (0, 220, 0), "interpolated": (0, 200, 255), "refined": (220, 220, 0),
              "none": (0, 0, 230)}[state]
    x = _put(bar, text, x, y, colour) + 40
    _put(bar, f"possession: {label(owner['team']) if owner['team'] != '-' else '-'}", x, y)
    return bar


def render_frame(frame, idx, result, share=None):
    """One output frame: info bar on top, the video frame (left) and the minimap (right).

    result: a pipeline.Result; share: {team: percent} possession so far (optional).
    """
    hs = result.homographies
    frame = frame.copy()
    h, w = frame.shape[:2]
    people, ball, owner = result.people[idx], result.ball.get(idx), result.possession[idx]

    # Pitch lines projected into the image: they must lie on the real lines
    for line in LINES:
        dense = np.concatenate([np.linspace(a, b, 20) for a, b in zip(line[:-1], line[1:])])
        img = hs.pitch_to_image(dense, idx)
        cv2.polylines(frame, [np.clip(img, -10000, 10000).astype(np.int32)], False, (255, 255, 0), 2, cv2.LINE_AA)

    mini = _MINIMAP.copy()
    # Part of the pitch the camera sees
    corners = hs.image_to_pitch([[0, 0], [w, 0], [w, h], [0, h]], idx)
    cv2.polylines(mini, [np.array([mini_px(c) for c in np.clip(corners, -50, 200)], np.int32)], True,
                  (255, 255, 0), 1, cv2.LINE_AA)

    # Ball trail (older = darker)
    trail = [(result.ball[i]["x_m"], result.ball[i]["y_m"]) for i in result.ids
             if idx - BALL_TRAIL <= i <= idx and i in result.ball]
    for k, (a, b) in enumerate(zip(trail[:-1], trail[1:])):
        shade = int(90 + 165 * (k + 1) / len(trail))
        cv2.line(mini, mini_px(a), mini_px(b), (shade, shade, shade), 2, cv2.LINE_AA)

    # People: box + track id on the frame, dot on the minimap
    owner_xy = None
    for p in people:
        colour = COLOURS.get(p["group"], (150, 150, 150))
        x1, y1, x2, y2 = (int(v) for v in p["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)  # black edge: dark boxes stay visible on grass
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, str(p["track_id"]), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, str(p["track_id"]), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
        _dot(mini, (p["x_m"], p["y_m"]), colour, 10)
        if owner["track_id"] is not None and p["track_id"] == owner["track_id"]:
            owner_xy = (p["x_m"], p["y_m"])
    if owner_xy is not None:  # ring around the player who has the ball
        cv2.circle(mini, mini_px(owner_xy), 16, (255, 255, 255), 2, cv2.LINE_AA)

    # Ball: red box on the frame; white dot (detected / refined) or open red circle (interpolated) on the minimap
    if ball:
        interpolated = ball["state"] == "interpolated"
        if ball["box"] is not None:
            x1, y1, x2, y2 = (int(v) for v in ball["box"])
            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 255), 2)
        xy = (ball["x_m"], ball["y_m"])
        _dot(mini, xy, (0, 0, 255) if interpolated else (255, 255, 255), 7, filled=not interpolated)
        if not interpolated:
            cv2.circle(mini, mini_px(xy), 8, (0, 0, 0), 1, cv2.LINE_AA)

    left = cv2.resize(frame, (int(w * OUT_H / h), OUT_H))
    right = cv2.resize(mini, (int(mini.shape[1] * OUT_H / mini.shape[0]), OUT_H))
    body = np.hstack([left, right])
    bar = _info_bar(body.shape[1], idx, result.ids[-1], result.fps, idx in hs.keyframes, people, ball, owner, share)
    return np.vstack([bar, body])


def write_video(result, out_path, progress=None):
    """Render every processed frame and write an H.264 mp4 (plays in browsers, unlike OpenCV's mp4v).

    The output plays at real speed: fps = video fps / stride. progress(done, total) after every frame.
    """
    import imageio_ffmpeg
    writer, owned = None, Counter()
    tmp = Path(out_path).with_suffix(".tmp.mp4")
    for n, (idx, frame) in enumerate(iter_frames(result.video, result.ids)):
        team = result.possession[idx]["team"]
        if team != "-":
            owned[team] += 1
        total = sum(owned.values())
        share = {t: 100 * c / total for t, c in sorted(owned.items())} if total else {}
        image = render_frame(frame, idx, result, share)
        if writer is None:
            writer = imageio_ffmpeg.write_frames(
                str(tmp), (image.shape[1], image.shape[0]), fps=result.fps / result.stride, codec="libx264",
                pix_fmt_out="yuv420p", macro_block_size=1, ffmpeg_log_level="error",
                output_params=["-movflags", "+faststart"])   # moov atom first: a browser can start playing
            writer.send(None)                                # start the generator
        writer.send(np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        if progress:
            progress(n + 1, len(result.ids))
    if writer is not None:
        writer.close()
        tmp.replace(out_path)
    return Path(out_path)
