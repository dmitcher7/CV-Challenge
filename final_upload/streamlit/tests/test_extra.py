"""Extra statistics (speed zones, sprints, kilometres, attacking direction, attack counts) and run_all on synthetic data."""

import numpy as np
import pandas as pd
import pytest

import evaluate_system
import run_all
import run_analytics
from analytics import (attack_counts, attack_direction, classify_speed, count_sprints, kilometres,
                       player_table, team_zone_summary, track_zones, zone_table)
from analytics.zones import step_table

FPS = 30


def _moving(segments, x0=10.0, y0=30.0):
    """Track from (seconds, m/s) segments, 30 fps -> (t, x, y)."""
    t_all, x_all, t0, x = [], [], 0.0, x0
    for secs, v in segments:
        t = t0 + np.arange(0, int(round(secs * FPS))) / FPS
        t_all.append(t)
        x_all.append(x + v * (t - t0))
        x, t0 = x + v * secs, t0 + secs
    t, xs = np.concatenate(t_all), np.concatenate(x_all)
    return t, xs, np.full_like(t, y0)


def test_classify_speed_boundaries():
    assert classify_speed(0.0) == "standing" and classify_speed(1.99) == "standing"
    assert classify_speed(2.0) == "walking" and classify_speed(6.99) == "walking"
    assert classify_speed(7.0) == "jogging" and classify_speed(14.99) == "jogging"
    assert classify_speed(15.0) == "running" and classify_speed(19.99) == "running"
    assert classify_speed(20.0) == "sprinting" and classify_speed(60.0) == "sprinting"
    assert list(classify_speed(np.array([1.0, 5.0, 25.0]))) == ["standing", "walking", "sprinting"]


def test_zone_time_and_distance_standing_then_running():
    t, x, y = _moving([(1.0, 0.0), (2.0, 5.0)])             # 5 m/s = 18 km/h -> "lopen"
    z = track_zones(t, x, y, smooth_s=0)
    assert abs(z["time_standing_s"] - 1.0) < 0.35
    assert abs(z["time_running_s"] - 2.0) < 0.35
    assert z["time_sprinting_s"] == 0 and z["n_sprints"] == 0
    assert abs(z["dist_running_m"] - 10.0) < 1.5
    total_t = sum(v for k, v in z.items() if k.startswith("time_"))
    total_d = sum(v for k, v in z.items() if k.startswith("dist_"))
    assert abs(total_t - (t[-1] - t[0])) < 1e-6
    assert abs(total_d - (x[-1] - x[0])) < 1e-6                # zone distances add up to the whole distance


def test_sprint_counting():
    for segments, expected in (
        ([(1.0, 0.0), (2.0, 7.0), (1.0, 0.0)], 1),               # 25 km/h for 2 s
        ([(1.0, 0.0), (0.6, 7.0), (1.0, 0.0)], 0),               # too short
        ([(2.0, 7.0), (2.0, 0.0), (2.0, 7.0)], 2),               # two sprints, a stop in between
        ([(3.0, 5.0)], 0),                                       # 18 km/h is not a sprint
    ):
        t, x, y = _moving(segments)
        assert track_zones(t, x, y, smooth_s=0)["n_sprints"] == expected, segments


def test_sprint_does_not_continue_across_a_gap():
    t1, x1, y1 = _moving([(0.7, 7.0)])
    t = np.concatenate([t1, t1 + 0.7 + 1.0])                    # 1 s gap: two 0.7 s sprints, not one 1.4 s sprint
    x = np.concatenate([x1, x1 + 40])
    steps = step_table(t, x, np.zeros_like(t), smooth_s=0)
    assert count_sprints(steps) == 0


def test_zone_table_and_team_summary():
    rows = []
    for tid, team, segs in ((1, "A", [(1.0, 0.0), (2.0, 7.0)]), (2, "B", [(3.0, 1.0)]), (3, "A", [(0.5, 3.0)])):
        t, x, y = _moving(segs)
        rows.append(pd.DataFrame({"frame": range(len(t)), "time_s": t, "track_id": tid, "group": f"player_{team}",
                                  "team": team, "x_m": x, "y_m": y}))
    zones = zone_table(pd.concat(rows), smooth_s=0, min_track_s=1.0)
    assert sorted(zones["track_id"]) == [1, 2]                  # track 3 is too short
    s = team_zone_summary(zones).set_index("team")
    assert s.loc["A", "n_sprints"] == 1 and s.loc["B", "n_sprints"] == 0
    assert s.loc["B", "share_walking"] == 1.0                   # 1 m/s = 3.6 km/h
    assert abs(sum(s.loc["A", f"share_{k}"] for k in ("standing", "walking", "jogging", "running", "sprinting")) - 1) < 0.01


def test_kilometres_sum_and_extrapolation():
    table = pd.DataFrame({"track_id": [1, 2, 3], "group": ["player_A", "player_A", "player_B"],
                          "team": ["A", "A", "B"], "duration_s": [2.0, 2.0, 10.0],
                          "distance_m": [10.0, 14.0, 50.0]})
    per_team, per_player = kilometres(table, clip_s=22.0)
    a = per_team.set_index("team")
    assert abs(a.loc["A", "total_km"] - 0.024) < 1e-9 and abs(a.loc["B", "total_km"] - 0.05) < 1e-9
    assert abs(per_player.set_index("track_id").loc[3, "km_per_90_indicative"] - 5 * 5400 / 1000) < 0.01   # 5 m/s
    assert a.loc["A", "km_per_player_per_90_indicative"] == pytest.approx(((5 + 7) / 2) * 5.4, abs=0.1)
    assert (per_player["km"].sum() * 1000) == pytest.approx(74.0, abs=0.01)


def _tracks(keeper_x=None, a_x=60.0, b_x=60.0, keeper_team="B"):
    rows = []
    for team, x in (("A", a_x), ("B", b_x)):
        for k in range(3):
            rows.append({"track_id": len(rows), "group": f"player_{team}", "team": team, "x_m": x + k, "y_m": 30.0})
    if keeper_x is not None:
        rows.append({"track_id": 99, "group": f"player_{keeper_team}_goal_keeper", "team": keeper_team,
                     "x_m": keeper_x, "y_m": 34.0})
    return pd.DataFrame(rows)


def _frames(team_dx=None):
    """Ball under possession: team_dx = {'A': metres moved in x, ...}; no possession -> ball far away and still."""
    rows, x = [], 50.0
    for team, dx in (team_dx or {}).items():
        for i in range(20):
            rows.append({"possession_team": team, "ball_x_m": x + dx * i / 19, "ball_y_m": 34.0})
        x += dx
    if not rows:
        rows = [{"possession_team": "-", "ball_x_m": 50.0, "ball_y_m": 34.0}] * 5
    f = pd.DataFrame(rows)
    f.insert(0, "frame", range(len(f)))
    return f


def test_direction_keeper_at_right_means_that_team_attacks_left():
    d = attack_direction(_tracks(keeper_x=100.0, keeper_team="B"), _frames())
    assert d["a_attacks"] == "right" and d["goal_x"] == {"A": 105.0, "B": 0.0}   # Inter defends right
    assert "PSG valt aan richting rechter doel (x=105)" in d["text"]["A"]
    d = attack_direction(_tracks(keeper_x=100.0, keeper_team="A"), _frames())
    assert d["a_attacks"] == "left" and d["goal_x"]["A"] == 0.0                  # PSG keeper right -> PSG attacks left
    d = attack_direction(_tracks(keeper_x=4.0, keeper_team="B"), _frames())
    assert d["a_attacks"] == "left"


def test_direction_all_signals_agree_gives_high_confidence():
    d = attack_direction(_tracks(keeper_x=100.0, a_x=55.0, b_x=70.0), _frames({"A": 20.0, "B": -5.0}))
    assert d["a_attacks"] == "right" and d["confidence"] == "hoog" and d["agreement"] == "3 van 3 signalen eens"


def test_direction_conflicting_signals_keeper_wins_but_lower_confidence():
    # keeper + ball say right; outfield positions say left
    d = attack_direction(_tracks(keeper_x=100.0, a_x=80.0, b_x=55.0), _frames({"A": 20.0}))
    assert d["a_attacks"] == "right" and d["confidence"] == "middel" and d["against"] == ["mean_x"]


def test_direction_tie_and_no_info():
    # keeper (weight 2) right vs mean_x + ball (1 + 1) left -> tie: no decision
    d = attack_direction(_tracks(keeper_x=100.0, a_x=80.0, b_x=55.0), _frames({"A": -20.0}))
    assert d["a_attacks"] is None and d["goal_x"] == {"A": None, "B": None} and d["confidence"] == "laag"
    assert "onbekend" in d["text"]["A"]
    d = attack_direction(_tracks(), _frames())
    assert d["a_attacks"] is None and d["confidence"] == "onbekend"
    # keeper near the halfway line says nothing
    assert attack_direction(_tracks(keeper_x=55.0), _frames())["signals"]["keeper"] == 0


def _ball(phases):
    """phases = list of (team, [(x, y), ...]); a '-' frame block separates different teams."""
    rows = []
    for team, pts in phases:
        for x, y in pts:
            rows.append({"possession_team": team, "ball_x_m": x, "ball_y_m": y})
    f = pd.DataFrame(rows)
    f.insert(0, "frame", range(len(f)))
    return f


def test_attack_counts_final_third_and_box():
    goal = {"A": 105.0, "B": 0.0}
    frames = _ball([
        ("A", [(40, 34), (60, 34), (75, 34), (80, 34), (95, 34), (100, 34)]),   # enters final third (>=70) and box (>=88.5)
        ("B", [(50, 34), (30, 34), (20, 34), (10, 34)]),                         # B: final third (<=35) yes, box (<=16.5) at x=10
        ("A", [(50, 34), (55, 34)]),                                             # never near the goal
        ("B", [(60, 34), (50, 34)]),
        ("A", [(90, 5), (95, 5)]),                                               # final third, but wide of the box (y=5)
    ])
    c = attack_counts(frames, goal)
    assert c["A"] == {"phases": 3, "final_third_entries": 2, "attacks": 2, "box_entries": 1}
    assert c["B"] == {"phases": 2, "final_third_entries": 1, "attacks": 1, "box_entries": 1}


def test_attack_counts_counts_entries_once_per_phase_visit():
    goal = {"A": 105.0, "B": 0.0}
    frames = _ball([("A", [(60, 34), (80, 34), (82, 34), (60, 34), (85, 34)])])  # leaves and re-enters: 2 entries, 1 attack
    c = attack_counts(frames, goal)["A"]
    assert c["final_third_entries"] == 2 and c["attacks"] == 1 and c["phases"] == 1
    assert attack_counts(frames, {"A": None, "B": None})["A"]["attacks"] == 0   # unknown direction: nothing counted
    frames.loc[2, "ball_x_m"] = np.nan                                           # missing ball frames are skipped
    assert attack_counts(frames, goal)["A"]["final_third_entries"] == 2


# ---------------------------------------------------------------- run_all (no model)

def _write_synthetic_outputs(out, model="syn"):
    rows = []
    n = 3 * FPS
    for team, x0 in (("A", 40.0), ("B", 60.0)):
        for k in range(4):
            for f in range(n):
                t = f / FPS
                rows.append({"frame": f, "time_s": round(t, 3), "track_id": (1 if team == "A" else 11) + k,
                             "group": f"player_{team}", "team": team, "x_m": x0 + 3 * k + 2.0 * t,
                             "y_m": 15.0 + 10 * k, "conf": 0.9, "box_x1": 0, "box_y1": 0, "box_x2": 1, "box_y2": 1})
    for f in range(n):
        rows.append({"frame": f, "time_s": round(f / FPS, 3), "track_id": 50, "group": "player_B_goal_keeper",
                     "team": "B", "x_m": 100.0, "y_m": 34.0, "conf": 0.9, "box_x1": 0, "box_y1": 0, "box_x2": 1, "box_y2": 1})
    pd.DataFrame(rows).to_csv(out / f"tracks_{model}.csv", index=False)
    frames = pd.DataFrame({"frame": range(n), "time_s": np.arange(n) / FPS,
                           "ball_x_m": np.linspace(50, 95, n), "ball_y_m": 34.0, "ball_state": "detected",
                           "possession_team": "A", "possession_track_id": 1})
    frames.to_csv(out / f"frames_{model}.csv", index=False)


def test_run_all_on_synthetic_csvs(tmp_path, monkeypatch):
    out, data = tmp_path / "output", tmp_path / "data"
    out.mkdir()
    data.mkdir()                                                # no ground truth files: evaluation is skipped
    for mod in (run_all, run_analytics, evaluate_system):
        monkeypatch.setattr(mod, "OUTPUT_DIR", out)
    monkeypatch.setattr(run_all, "DATA_DIR", data)
    called = {}

    def fake_minimap(video, weights, imgsz, start, end, stride, conf, ball_conf, no_video, ball_refine):
        called.update(weights=weights, imgsz=imgsz, stride=stride, no_video=no_video, ball_refine=ball_refine)
        _write_synthetic_outputs(out)
        return "syn"

    monkeypatch.setattr(run_all, "make_minimap", fake_minimap)
    args = run_all.build_parser().parse_args(["--weights", "x/syn.pt", "--imgsz", "640", "--stride", "2",
                                              "--no-video", "--ball-refine"])
    path = run_all.run_all(args)
    assert called == {"weights": "x/syn.pt", "imgsz": 640, "stride": 2, "no_video": True, "ball_refine": True}
    assert path == out / "RAPPORT_syn.md"
    text = path.read_text(encoding="utf-8")
    for needle in ("## Snelheidszones", "## Kilometers", "## Aanvalsrichting en aanvallen", "## Systeemevaluatie",
                   "## Beperkingen", "PSG valt aan richting rechter doel (x=105)", "ground_truth.json ontbreekt"):
        assert needle in text, needle
    import re
    for rel in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):     # every embedded figure exists next to the report
        assert (out / rel).exists(), rel
    for name in ("zones_team_A_syn.png", "heatmap_team_A_syn.png", "report_syn.md", "km_teams_syn.csv"):
        assert (out / "analytics" / name).exists(), name


def test_player_table_zone_table_agree_on_distance():
    t, x, y = _moving([(1.0, 0.0), (2.0, 5.0)])
    tracks = pd.DataFrame({"frame": range(len(t)), "time_s": t, "track_id": 1, "group": "player_A", "team": "A",
                           "x_m": x, "y_m": y})
    table, zones = player_table(tracks, smooth_s=0.5), zone_table(tracks, smooth_s=0.5)
    dist = sum(zones[f"dist_{k}_m"].iloc[0] for k in ("standing", "walking", "jogging", "running", "sprinting"))
    assert abs(dist - table["distance_m"].iloc[0]) < 0.2
