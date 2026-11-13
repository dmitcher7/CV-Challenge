"""Ball refinement (crop re-detection): confirmation of crop hits, gap filling and crop placement. No model."""

import numpy as np

from core.ball_refine import CROP_SIZE, REFINE_CONF, confirm_hits, crop_windows, fill_between


def test_hit_next_to_a_detected_ball_is_accepted():
    known = {10: (50.0, 30.0)}
    hits = {12: ((51.5, 30.0), 0.4)}                   # 2 frames later, 0.75 m/frame
    assert confirm_hits(hits, known) == {12}


def test_isolated_hit_is_rejected():
    known = {10: (50.0, 30.0)}
    assert confirm_hits({40: ((80.0, 5.0), 0.9)}, known) == set()          # no ball nearby in time
    assert confirm_hits({12: ((70.0, 30.0), 0.9)}, known) == set()         # nearby in time, but 10 m/frame


def test_two_adjacent_hits_confirm_each_other_but_a_weak_one_is_not_kept():
    hits = {100: ((60.0, 20.0), 0.35), 101: ((60.8, 20.2), 0.2)}
    assert REFINE_CONF > 0.2
    assert confirm_hits(hits, {}) == {100}             # the weak hit confirms, but is no ball itself
    assert confirm_hits({100: hits[100]}, {}) == set()


def test_confirmation_window_follows_the_stride():
    known = {10: (50.0, 30.0)}
    hits = {15: ((52.0, 30.0), 0.5)}                   # next processed frame with stride 5
    assert confirm_hits(hits, known, stride=1) == set()
    assert confirm_hits(hits, known, stride=5) == {15}


def test_fill_between_only_next_to_refined_balls_and_up_to_max_gap():
    points = {0: (0.0, 0.0), 6: (6.0, 0.0), 40: (8.0, 0.0), 41: (9.0, 0.0)}
    filled = fill_between(points, {6}, list(range(42)), max_gap=15)
    assert set(filled) == set(range(1, 6))             # 0..6 filled; 6..40 is longer than max_gap
    assert np.allclose(filled[3], (3.0, 0.0))
    assert fill_between({0: (0.0, 0.0), 2: (20.0, 0.0)}, {2}, [0, 1, 2], 15) == {}   # 10 m/frame: no


class _Identity:
    """Homography stub: pitch metres == image pixels."""

    def pitch_to_image(self, points, frame_idx):
        return np.asarray(points, float)


def test_crop_windows_cover_frames_without_ball_inside_the_image():
    ball = {0: (100.0, 100.0), 10: (1100.0, 100.0)}
    windows = crop_windows(ball, list(range(11)), _Identity(), 1920, 1080)
    assert set(windows) == set(range(1, 10))           # only frames without a ball
    assert windows[5] == (round(600 - CROP_SIZE / 2), 0)   # centred on the interpolated position, clamped at the top
    assert all(0 <= x <= 1920 - CROP_SIZE and 0 <= y <= 1080 - CROP_SIZE for x, y in windows.values())
    assert crop_windows({0: (100.0, 100.0)}, [0, 200], _Identity(), 1920, 1080) == {}   # too far from any ball
