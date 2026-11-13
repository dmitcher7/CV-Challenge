"""Homography: a known H is recovered, errors in metres, collinear / clustered points are flagged."""

import cv2
import numpy as np

from core.homography import (Homographies, fit, project, reprojection_errors_m, reprojection_report,
                             spread_check)
from core.pitch import KEYPOINTS

# A made-up camera: pitch metres -> image pixels (scale + rotation + perspective); we need its inverse
H_PITCH_TO_IMG = np.array([[16.0, 4.0, 300.0], [-2.0, 9.0, 150.0], [0.0004, 0.0012, 1.0]])
NAMES = ["corner_TL", "corner_TR", "corner_BL", "corner_BR", "centre", "L_penalty_spot", "R_box_T", "mid_T"]


def clicks(names, noise=None):
    """Keypoints 'clicked' in the image of the made-up camera (optionally with pixel noise on some points)."""
    pts = {}
    for n in names:
        px = project([KEYPOINTS[n]], H_PITCH_TO_IMG)[0]
        if noise and n in noise:
            px = px + noise[n]
        pts[n] = px.tolist()
    return pts


def test_fit_recovers_known_homography():
    H, errors = fit(clicks(NAMES))
    for n in NAMES:  # every keypoint maps back to its true position in metres
        assert np.allclose(project([clicks(NAMES)[n]], H)[0], KEYPOINTS[n], atol=1e-3)
    assert max(errors.values()) < 1e-3


def test_error_in_metres_is_zero_for_exact_clicks_and_four_points():
    pts = clicks(NAMES)
    H, _ = fit(pts)
    assert max(reprojection_errors_m(pts, H).values()) < 1e-3
    four = clicks(NAMES[:4])
    rows = reprojection_report({0: four})
    assert rows[0]["n_points"] == 4 and not rows[0]["checkable"] and rows[0]["rmse_m"] < 1e-3


def test_misclick_shows_up_as_metres_error():
    pts = clicks(NAMES, noise={"centre": np.array([0.0, 60.0])})  # 60 px off = several metres
    rows = reprojection_report({0: pts})
    assert rows[0]["checkable"]
    assert rows[0]["worst"] == "centre" and rows[0]["max_m"] > 2.0   # RANSAC ignores it for H, the error stays visible


def test_collinear_points_are_flagged():
    # All on the left goal line (x = 0): collinear
    line = {n: [0, 0] for n in ("corner_TL", "corner_BL", "L_box_goal_T", "L_box_goal_B", "L_goal_area_goal_T")}
    result = spread_check(line)
    assert not result["good_quad"] and result["warnings"]
    assert result["hull_frac"] < 0.001


def test_clustered_points_are_flagged_and_spread_points_pass():
    cluster = ["R_goal_area_T", "R_goal_area_B", "R_goal_area_goal_T", "R_goal_area_goal_B", "R_penalty_spot"]
    assert any("cover only" in w for w in spread_check({n: [0, 0] for n in cluster})["warnings"])
    good = spread_check({n: [0, 0] for n in NAMES})
    assert good["good_quad"] and good["hull_frac"] > 0.5
    assert all("cover only" not in w for w in good["warnings"])


def test_interpolation_between_keyframes():
    # Two keyframes with the same camera: every frame in between must give the same H
    hs = Homographies({0: clicks(NAMES), 30: clicks(NAMES)})
    mid = hs.image_to_pitch(project([KEYPOINTS["centre"]], H_PITCH_TO_IMG), 15)[0]
    assert np.allclose(mid, KEYPOINTS["centre"], atol=1e-3)
    assert hs.for_frame(-5) is hs.keyframes[0] and hs.for_frame(99) is hs.keyframes[30]


def test_homographies_needs_a_keyframe():
    try:
        Homographies({0: {"centre": [1, 1]}})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_cv2_available():
    assert cv2.__version__
