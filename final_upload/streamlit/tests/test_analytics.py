"""Analytics on synthetic tracks with known ground truth."""

import numpy as np
import pandas as pd

from analytics import (heatmap, player_table, possession_summary, smooth_track, team_summary,
                       track_stats)


def _line(speed=5.0, seconds=2.0, fps=30, stride=1, x0=10.0, y0=30.0):
    t = np.arange(0, seconds + 1e-9, stride / fps)
    return t, x0 + speed * t, np.full_like(t, y0)


def test_constant_speed_distance_and_kmh():
    t, x, y = _line()
    s = track_stats(t, x, y)
    assert abs(s["distance_m"] - 10.0) < 0.05
    assert abs(s["mean_kmh"] - 18.0) < 0.2
    assert abs(s["max_kmh"] - 18.0) < 0.5
    assert not s["clipped"]


def test_strided_frames_same_result():
    t, x, y = _line(stride=2)
    assert abs(track_stats(t, x, y)["distance_m"] - 10.0) < 0.05


def test_jittered_stationary_player_is_much_smaller_than_raw():
    rng = np.random.default_rng(0)
    t = np.arange(0, 4, 1 / 30)
    x = 50 + rng.normal(0, 0.3, len(t))
    y = 30 + rng.normal(0, 0.3, len(t))
    raw = np.hypot(np.diff(x), np.diff(y)).sum()
    smooth = track_stats(t, x, y, smooth_s=0.5)["distance_m"]
    assert raw > 20
    assert smooth < raw / 4


def test_speed_is_capped_and_flagged():
    t = np.arange(0, 2, 1 / 30)
    s = track_stats(t, 10 + 20.0 * t, np.zeros_like(t), smooth_s=0)  # 72 km/h
    assert s["max_kmh"] == 36.0 and s["clipped"]


def test_gap_is_not_bridged():
    t1, x1, y1 = _line(seconds=1.0)
    t2, x2, y2 = _line(seconds=1.0, x0=60.0)   # teleports 50 m during a 2 s gap
    t = np.concatenate([t1, t2 + 3.0])
    x, y = np.concatenate([x1, x2]), np.concatenate([y1, y2])
    s = track_stats(t, x, y)
    assert abs(s["distance_m"] - 10.0) < 0.1   # 5 m + 5 m, not 55 m
    sx, _ = smooth_track(t, x, y)
    assert sx[len(t1) - 1] < 20 < 60 <= sx[len(t1)]  # no averaging across the gap


def test_player_table_drops_short_tracks_and_team_summary():
    rows = []
    for tid, team, dur in ((1, "A", 2.0), (2, "B", 2.0), (3, "A", 0.5)):
        t, x, y = _line(seconds=dur)
        rows.append(pd.DataFrame({"frame": range(len(t)), "time_s": t, "track_id": tid,
                                  "group": f"player_{team}", "team": team, "x_m": x, "y_m": y}))
    table = player_table(pd.concat(rows), min_track_s=1.0)
    assert sorted(table["track_id"]) == [1, 2]
    summary = team_summary(table).set_index("team")
    assert summary.loc["A", "n_tracks"] == 1
    assert abs(summary.loc["B", "total_distance_m"] - 10.0) < 0.1


def test_heatmap_sums_and_peak():
    xs, ys = np.full(100, 50.5), np.full(100, 30.5)
    assert heatmap(xs, ys, sigma_m=0).sum() == 100
    blurred = heatmap(xs, ys, sigma_m=2.5)
    assert abs(blurred.sum() - 100) < 1e-6
    assert blurred.shape == (68, 105)
    assert np.unravel_index(blurred.argmax(), blurred.shape) == (30, 50)
    assert heatmap([-5, 200], [-5, 200], sigma_m=0).sum() == 2   # clipped, not dropped


def test_possession_summary():
    frames = pd.DataFrame({"possession_team": ["A", "A", None, "B", "B", "B", "A"]})
    p = possession_summary(frames)
    assert p["frames_with_possession"] == 6
    assert p["changes"] == 2
    assert abs(p["share_A"] - 0.5) < 1e-9
