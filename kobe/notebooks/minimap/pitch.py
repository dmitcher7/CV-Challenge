"""Pitch model in metres: named keypoints, line drawing and the minimap image.

Coordinate system (same orientation as the broadcast view):
    origin = corner of the far touchline and the left goal line
    x = along the touchline, 0 (left goal line) .. 105 (right goal line)
    y = across the pitch,    0 (far touchline, top of the image) .. 68 (near touchline)
"""

from pathlib import Path

import cv2
import numpy as np

HOMOGRAPHY_DIR = Path(__file__).resolve().parent   # kobe/notebooks/minimap/
# Project root = the first parent folder that contains data/frames
REPO_ROOT = next(p for p in HOMOGRAPHY_DIR.parents if (p / "data" / "frames").exists())
FRAMES_DIR = REPO_ROOT / "data" / "frames"
VIDEO_PATH = REPO_ROOT / "data" / "raw_video" / "match.mp4"
KEYPOINTS_JSON = HOMOGRAPHY_DIR / "keypoints.json"   # clicked points, committed
OUTPUT_DIR = HOMOGRAPHY_DIR / "output"               # generated, not committed

# FIFA dimensions in metres (Allianz Arena, Munich: 105 x 68)
LENGTH, WIDTH = 105.0, 68.0
BOX_DEPTH, BOX_WIDTH = 16.5, 40.32        # penalty area
GOAL_AREA_DEPTH, GOAL_AREA_WIDTH = 5.5, 18.32
PENALTY_SPOT = 11.0                       # distance from the goal line
CIRCLE_RADIUS = 9.15                      # centre circle and penalty arc


def _keypoints():
    mid_x, mid_y = LENGTH / 2, WIDTH / 2
    points = {
        "corner_TL": (0, 0), "corner_TR": (LENGTH, 0),
        "corner_BL": (0, WIDTH), "corner_BR": (LENGTH, WIDTH),
        "mid_T": (mid_x, 0), "mid_B": (mid_x, WIDTH), "centre": (mid_x, mid_y),
        "circle_T": (mid_x, mid_y - CIRCLE_RADIUS), "circle_B": (mid_x, mid_y + CIRCLE_RADIUS),
    }
    # Where the penalty arc meets the front line of the penalty area
    arc_dy = np.sqrt(CIRCLE_RADIUS ** 2 - (BOX_DEPTH - PENALTY_SPOT) ** 2)
    for side, x0, d in (("L", 0.0, 1.0), ("R", LENGTH, -1.0)):  # d = direction into the pitch
        points.update({
            f"{side}_box_goal_T": (x0, mid_y - BOX_WIDTH / 2),
            f"{side}_box_goal_B": (x0, mid_y + BOX_WIDTH / 2),
            f"{side}_box_T": (x0 + d * BOX_DEPTH, mid_y - BOX_WIDTH / 2),
            f"{side}_box_B": (x0 + d * BOX_DEPTH, mid_y + BOX_WIDTH / 2),
            f"{side}_goal_area_goal_T": (x0, mid_y - GOAL_AREA_WIDTH / 2),
            f"{side}_goal_area_goal_B": (x0, mid_y + GOAL_AREA_WIDTH / 2),
            f"{side}_goal_area_T": (x0 + d * GOAL_AREA_DEPTH, mid_y - GOAL_AREA_WIDTH / 2),
            f"{side}_goal_area_B": (x0 + d * GOAL_AREA_DEPTH, mid_y + GOAL_AREA_WIDTH / 2),
            f"{side}_penalty_spot": (x0 + d * PENALTY_SPOT, mid_y),
            f"{side}_arc_T": (x0 + d * BOX_DEPTH, mid_y - arc_dy),
            f"{side}_arc_B": (x0 + d * BOX_DEPTH, mid_y + arc_dy),
        })
    return {name: (float(x), float(y)) for name, (x, y) in points.items()}


KEYPOINTS = _keypoints()  # name -> (x, y) in metres


def _circle(cx, cy, r, a0=0.0, a1=2 * np.pi, n=64):
    angles = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1)


def _lines():
    """All pitch markings as polylines (arrays of (x, y) in metres)."""
    mid_x, mid_y = LENGTH / 2, WIDTH / 2
    lines = [
        np.array([(0, 0), (LENGTH, 0), (LENGTH, WIDTH), (0, WIDTH), (0, 0)]),  # touch- and goal lines
        np.array([(mid_x, 0), (mid_x, WIDTH)]),                                # halfway line
        _circle(mid_x, mid_y, CIRCLE_RADIUS),
    ]
    # Half-angle of the penalty arc part that lies outside the penalty area
    arc = np.arccos((BOX_DEPTH - PENALTY_SPOT) / CIRCLE_RADIUS)
    for x0, d in ((0.0, 1.0), (LENGTH, -1.0)):
        for depth, width in ((BOX_DEPTH, BOX_WIDTH), (GOAL_AREA_DEPTH, GOAL_AREA_WIDTH)):
            top, bottom, x1 = mid_y - width / 2, mid_y + width / 2, x0 + d * depth
            lines.append(np.array([(x0, top), (x1, top), (x1, bottom), (x0, bottom)]))
        centre = 0.0 if d > 0 else np.pi  # arc points towards the middle of the pitch
        lines.append(_circle(x0 + d * PENALTY_SPOT, mid_y, CIRCLE_RADIUS, centre - arc, centre + arc, 24))
    return [line.astype(np.float64) for line in lines]


LINES = _lines()


def draw_minimap(scale=10, margin=4):
    """Top-down pitch image. scale = pixels per metre, margin in metres."""
    h, w = int((WIDTH + 2 * margin) * scale), int((LENGTH + 2 * margin) * scale)
    image = np.full((h, w, 3), (40, 120, 40), np.uint8)
    for i in range(0, int(LENGTH), 20):  # lighter mowing stripe of 10 m every 20 m
        p0 = to_minimap_px((i, 0), scale, margin)
        p1 = to_minimap_px((min(i + 10, LENGTH), WIDTH), scale, margin)
        cv2.rectangle(image, p0, p1, (46, 135, 46), -1)
    for line in LINES:
        pts = np.array([to_minimap_px(p, scale, margin) for p in line], np.int32)
        cv2.polylines(image, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)
    for x in (PENALTY_SPOT, LENGTH / 2, LENGTH - PENALTY_SPOT):
        cv2.circle(image, to_minimap_px((x, WIDTH / 2), scale, margin), 3, (255, 255, 255), -1)
    return image


def to_minimap_px(point, scale=10, margin=4):
    """(x, y) in metres -> integer pixel on the minimap of draw_minimap()."""
    return int(round((point[0] + margin) * scale)), int(round((point[1] + margin) * scale))
