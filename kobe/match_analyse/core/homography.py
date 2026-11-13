"""Homography per frame from the clicked keypoints in data/keypoints.json.

On every keyframe with >= 4 clicked points we fit H (image pixels -> pitch metres)
with cv2.findHomography. The camera pans and zooms, so frames between two
keyframes get their own H: we project a grid of pitch points into both keyframes,
interpolate those image positions linearly in time, and fit H again on the
interpolated grid. Frames before the first / after the last keyframe use the
nearest keyframe.

Quality checks of the clicks (both work in metres on the pitch):
- reprojection_report(): per keyframe the RMSE between the clicked points, mapped to the pitch
  with the fitted H, and where they should be. A keyframe with exactly 4 points always has
  error 0 (4 points determine H exactly), so it cannot be checked: add a 5th point.
- spread_check(): are the clicked points spread over the pitch, and not (nearly) on one line?

Usage:
    from core.homography import Homographies
    hs = Homographies()
    feet_m = hs.image_to_pitch([(960, 700)], frame_idx=120)   # -> [[x, y]] in metres
"""

import itertools
import json
from functools import lru_cache

import cv2
import numpy as np

from core.paths import KEYPOINTS_JSON
from core.pitch import KEYPOINTS, LENGTH, WIDTH

# Pitch points used to interpolate between keyframes (6 x 5 grid over the pitch)
GRID = np.array([(x, y) for x in np.linspace(0, LENGTH, 6) for y in np.linspace(0, WIDTH, 5)])
RANSAC_M = 1.0           # a clicked point further than this (metres on the pitch) from the fit is an outlier
SPREAD_MIN_FRAC = 0.03   # the clicked points must cover at least this share of the pitch area (hull area)
COLLINEAR_AREA_M2 = 5.0  # three points spanning a triangle smaller than this (m^2) count as (nearly) collinear
ERROR_WARN_M = 1.0       # a keyframe RMSE above this (metres) is flagged


def project(points, H):
    """Apply homography H to an (N, 2) array of points."""
    pts = np.asarray(points, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def load_keypoints(path=KEYPOINTS_JSON):
    """{frame idx: {keypoint name: [x, y] image pixels}} from keypoints.json."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(idx): points for idx, points in data["frames"].items()}


def fit(points):
    """Fit H (image -> pitch) on {name: [x, y]}. Returns (H, {name: error in px}) or (None, {}).

    The error per point is the distance in pixels between where it was clicked and where
    the fitted H puts it: a large error means a misclick or a wrong point name.
    With exactly 4 points H always fits perfectly, so errors are only meaningful from 5 points.
    From 5 points we fit with RANSAC: a point more than RANSAC_M metres off is ignored,
    so one misclick cannot pull H (and the errors of all other points) out of shape.
    """
    names = [n for n in points if n in KEYPOINTS]
    if len(names) < 4:
        return None, {}
    image_pts = np.array([points[n] for n in names], np.float64)
    pitch_pts = np.array([KEYPOINTS[n] for n in names], np.float64)
    if len(names) == 4:
        H, _ = cv2.findHomography(image_pts, pitch_pts, 0)  # exact solution
    else:
        H, _ = cv2.findHomography(image_pts, pitch_pts, cv2.RANSAC, RANSAC_M)
    if H is None:
        return None, {}
    back = project(pitch_pts, np.linalg.inv(H))
    errors = {n: float(np.linalg.norm(back[i] - image_pts[i])) for i, n in enumerate(names)}
    return H, errors


def reprojection_errors_m(points, H):
    """{name: error in metres}: clicked point mapped to the pitch with H, distance to its true position."""
    names = [n for n in points if n in KEYPOINTS]
    mapped = project([points[n] for n in names], H)
    return {n: float(np.linalg.norm(mapped[i] - KEYPOINTS[n])) for i, n in enumerate(names)}


def reprojection_report(keypoints=None):
    """One row per keyframe with >= 4 points: {"frame", "n_points", "checkable", "rmse_m", "max_m", "worst"}.

    A 4-point keyframe always has error 0: "checkable" is False and the RMSE says nothing.
    """
    keypoints = load_keypoints() if keypoints is None else keypoints
    rows = []
    for idx, points in sorted(keypoints.items()):
        H, _ = fit(points)
        if H is None:
            continue
        errors = reprojection_errors_m(points, H)
        worst = max(errors, key=errors.get)
        rows.append({"frame": idx, "n_points": len(errors), "checkable": len(errors) >= 5,
                     "rmse_m": float(np.sqrt(np.mean(np.square(list(errors.values()))))),
                     "max_m": errors[worst], "worst": worst})
    return rows


def spread_check(points):
    """Are the clicked points spread out? Works on their true pitch positions (metres).

    Returns {"hull_frac", "min_triangle_m2", "n_collinear", "good_quad", "warnings"}:
    - hull_frac: area of the convex hull of the points / pitch area (small = points close together)
    - n_collinear: number of triples spanning a triangle < COLLINEAR_AREA_M2 (nearly on one line)
    - good_quad: is there a set of 4 points of which no 3 are nearly collinear? Without one H is ill-defined.
    """
    names = [n for n in points if n in KEYPOINTS]
    pts = np.array([KEYPOINTS[n] for n in names], np.float64)
    warnings = []
    if len(pts) < 4:
        return {"hull_frac": 0.0, "min_triangle_m2": 0.0, "n_collinear": 0, "good_quad": False,
                "warnings": [f"only {len(pts)} points (need 4)"]}
    hull_frac = float(cv2.contourArea(cv2.convexHull(pts.astype(np.float32)))) / (LENGTH * WIDTH)

    def area(i, j, k):
        a, b, c = pts[i], pts[j], pts[k]
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2

    triangles = {t: area(*t) for t in itertools.combinations(range(len(pts)), 3)}
    flat = {t for t, a in triangles.items() if a < COLLINEAR_AREA_M2}
    good_quad = any(all(t not in flat for t in itertools.combinations(q, 3))
                    for q in itertools.combinations(range(len(pts)), 4))
    if hull_frac < SPREAD_MIN_FRAC:
        warnings.append(f"points cover only {hull_frac:.1%} of the pitch (< {SPREAD_MIN_FRAC:.0%}): spread them out")
    if not good_quad:
        warnings.append("no 4 points without 3 nearly on one line: H is poorly determined")
    elif flat:
        warnings.append(f"{len(flat)} triples nearly collinear (fine if the rest is spread out)")
    return {"hull_frac": hull_frac, "min_triangle_m2": float(min(triangles.values())),
            "n_collinear": len(flat), "good_quad": good_quad, "warnings": warnings}


class Homographies:
    """H (image pixels -> pitch metres) for every frame of the clip."""

    def __init__(self, keypoints=None):
        keypoints = load_keypoints() if keypoints is None else keypoints
        self.keyframes = {}
        for idx, points in sorted(keypoints.items()):
            H, _ = fit(points)
            if H is not None:
                self.keyframes[idx] = H
        if not self.keyframes:
            raise ValueError("No keyframe with >= 4 points yet: run click_keypoints.py first")
        self.idx = sorted(self.keyframes)
        self.for_frame = lru_cache(maxsize=4096)(self._for_frame)

    def _for_frame(self, frame_idx):
        """H for one frame (interpolated between the two surrounding keyframes)."""
        if frame_idx <= self.idx[0]:
            return self.keyframes[self.idx[0]]
        if frame_idx >= self.idx[-1]:
            return self.keyframes[self.idx[-1]]
        b = next(i for i in self.idx if i >= frame_idx)
        if b == frame_idx:
            return self.keyframes[b]
        a = self.idx[self.idx.index(b) - 1]
        t = (frame_idx - a) / (b - a)
        grid_a = project(GRID, np.linalg.inv(self.keyframes[a]))
        grid_b = project(GRID, np.linalg.inv(self.keyframes[b]))
        H, _ = cv2.findHomography((1 - t) * grid_a + t * grid_b, GRID, 0)
        return H

    def image_to_pitch(self, points, frame_idx):
        """Image pixels -> pitch metres, (N, 2) array."""
        return project(points, self.for_frame(frame_idx))

    def pitch_to_image(self, points, frame_idx):
        """Pitch metres -> image pixels, (N, 2) array."""
        return project(points, np.linalg.inv(self.for_frame(frame_idx)))
