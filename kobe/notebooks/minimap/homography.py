"""Homography per frame from the clicked keypoints in keypoints.json.

On every keyframe with >= 4 clicked points we fit H (image pixels -> pitch metres)
with cv2.findHomography. The camera pans and zooms, so frames between two
keyframes get their own H: we project a grid of pitch points into both keyframes,
interpolate those image positions linearly in time, and fit H again on the
interpolated grid. Frames before the first / after the last keyframe use the
nearest keyframe.

Usage:
    from homography import Homographies
    hs = Homographies()
    feet_m = hs.image_to_pitch([(960, 700)], frame_idx=120)   # -> [[x, y]] in metres
"""

import json

import cv2
import numpy as np

from pitch import KEYPOINTS, KEYPOINTS_JSON, LENGTH, WIDTH

# Pitch points used to interpolate between keyframes (6 x 5 grid over the pitch)
GRID = np.array([(x, y) for x in np.linspace(0, LENGTH, 6) for y in np.linspace(0, WIDTH, 5)])
RANSAC_M = 1.0  # a clicked point further than this (in metres on the pitch) from the fit is an outlier


def project(points, H):
    """Apply homography H to an (N, 2) array of points."""
    pts = np.asarray(points, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def load_keypoints():
    """{frame idx: {keypoint name: [x, y] image pixels}} from keypoints.json."""
    if not KEYPOINTS_JSON.exists():
        return {}
    data = json.loads(KEYPOINTS_JSON.read_text(encoding="utf-8"))
    return {int(idx): points for idx, points in data["frames"].items()}


def fit(points):
    """Fit H (image -> pitch) on {name: [x, y]}. Returns (H, {name: error in px}) or (None, {}).

    The error per point is the distance between where it was clicked and where
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
            raise SystemExit("No keyframe with >= 4 points yet: run click_keypoints.py first")
        self.idx = sorted(self.keyframes)

    def for_frame(self, frame_idx):
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
