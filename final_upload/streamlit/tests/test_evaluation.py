"""Evaluation metrics and ground-truth helpers on synthetic data."""

import numpy as np
import pytest

from evaluation.gt_format import (add_person, assign_id, can_confirm, empty_frame, load_gt, remove_nearest_person,
                                  save_gt, set_ball)
from evaluation.metrics import click_height, evaluate, format_report, load_predictions, match_persons, tracking_metrics

ROLES = ["A", "A", "B", "K", "R"]
POS = [(10, 10), (20, 30), (40, 20), (60, 50), (80, 40)]


def make_gt(frames=(0, 30), poss="A", ball=(30, 30)):
    gt = {}
    for f in frames:
        gt[f] = {"done": True, "ball_visible": ball is not None, "possession": poss,
                 "ball": {"px": [0, 0], "m": list(ball)} if ball else None,
                 "persons": [{"id": i, "role": r, "px": [0, 0], "m": [x, y]}
                             for i, (r, (x, y)) in enumerate(zip(ROLES, POS))]}
    return gt


def make_pred(frames=(0, 30), shift=0.0, roles=ROLES, ball=(30, 30), poss="A", ids=None):
    """ids: None (track id = index) or {frame: [5 track ids]}."""
    people = {f: [{"track_id": i if ids is None else ids[f][i], "role": roles[i], "x": x + shift, "y": y}
                  for i, (x, y) in enumerate(POS)] for f in frames}
    return people, {f: ball for f in frames}, {f: poss for f in frames}


def test_perfect_match():
    r = evaluate(make_gt(), *make_pred())
    assert r["players"]["position_error_m"]["mean"] == pytest.approx(0)
    assert r["players"]["f1"] == 1 and r["players"]["tp"] == 10
    assert r["teams"]["role_accuracy"] == 1
    assert r["ball"]["detection_rate"] == 1 and r["ball"]["position_error_m"]["mean"] == 0
    assert r["possession"]["accuracy"] == 1
    assert r["tracking"]["id_switches"] == 0 and r["tracking"]["idf1_lite"] == 1


def test_shift_one_metre():
    r = evaluate(make_gt(), *make_pred(shift=1.0))
    assert r["players"]["position_error_m"]["mean"] == pytest.approx(1.0)
    assert r["players"]["position_error_m"]["p90"] == pytest.approx(1.0)


def test_far_prediction_is_fp_and_fn():
    people, ball, poss = make_pred()
    people[0][0]["x"] += 20
    r = evaluate(make_gt(), people, ball, poss)
    assert (r["players"]["tp"], r["players"]["fp"], r["players"]["fn"]) == (9, 1, 1)
    assert r["players"]["precision"] == pytest.approx(0.9)


def test_swapped_team_lowers_accuracy():
    roles = ["B", "A", "B", "K", "R"]       # first player gets the wrong team, in both frames
    r = evaluate(make_gt(), *make_pred(roles=roles))
    assert r["teams"]["role_accuracy"] == pytest.approx(0.8)
    assert r["teams"]["confusion"]["A"]["B"] == 2
    assert r["teams"]["team_accuracy_merged"] == pytest.approx(0.8)


def test_ball_and_possession():
    gt = make_gt()
    gt[30]["ball_visible"], gt[30]["ball"] = False, None
    people, ball, poss = make_pred(ball=(33, 34))
    ball[30] = (1, 1)                        # false ball where GT has none
    poss[30] = "B"
    r = evaluate(gt, people, ball, poss)
    assert r["ball"]["frames_with_ball"] == 1 and r["ball"]["position_error_m"]["mean"] == pytest.approx(5.0)
    assert r["ball"]["false_ball_rate"] == 1.0
    assert r["possession"]["accuracy"] == 0.5
    assert r["possession"]["confusion"]["A"] == {"A": 1, "B": 1}
    people, ball, poss = make_pred(ball=None)
    assert evaluate(make_gt(), people, ball, poss)["ball"]["detection_rate"] == 0


def test_nearest_processed_frame_and_gap():
    gt = make_gt(frames=(1, 31, 100))
    people, ball, poss = make_pred(frames=(0, 2, 30, 32))
    r = evaluate(gt, people, ball, poss)
    assert r["model_frames"]["matched"] == 2 and r["model_frames"]["skipped_no_processed_frame"] == 1
    assert r["players"]["f1"] == 1


def test_id_switches():
    ids = {0: [0, 1, 2, 3, 4], 30: [0, 1, 2, 3, 4], 60: [0, 1, 7, 3, 4]}   # GT person 2 changes track id once
    r = evaluate(make_gt(frames=(0, 30, 60)), *make_pred(frames=(0, 30, 60), ids=ids))
    assert r["tracking"]["id_switches"] == 1
    assert r["tracking"]["idtp"] == 14 and r["tracking"]["idf1_lite"] == pytest.approx(28 / 30)


def test_tracking_skipped_without_ids():
    gt = make_gt()
    gt[0]["persons"][0]["id"] = None
    assert "skipped" in evaluate(gt, *make_pred())["tracking"]
    assert tracking_metrics([], 0, 0)["id_switches"] == 0


def test_match_persons_radius():
    g = [{"m": [0, 0]}, {"m": [10, 0]}]
    p = [{"x": 10.5, "y": 0}, {"x": 0, "y": 2}, {"x": 50, "y": 50}]
    assert sorted((a, b) for a, b, _ in match_persons(g, p, 3.0)) == [(0, 1), (1, 0)]
    assert [(a, b) for a, b, _ in match_persons(g, p, 1.0)] == [(1, 0)]


def test_gt_save_load_and_helpers(tmp_path):
    class H:     # pixel -> metre: divide by 10
        def image_to_pitch(self, pts, idx):
            return np.asarray(pts, float) / 10

    frames = {}
    a = add_person(frames, 0, (100, 200), "A", H())
    assert a["m"] == [10.0, 20.0] and a["id"] == 1
    set_ball(frames, 0, (300, 300), H())
    frames[0]["possession"] = "B"
    assert can_confirm(frames[0]) == (True, "")
    assert not can_confirm(empty_frame())[0]
    # next labelled frame: a click near the old position keeps the id, a far one gets a new id
    near = add_person(frames, 30, (130, 210), "A", H())
    far = add_person(frames, 30, (900, 600), "B", H())
    again = add_person(frames, 30, (110, 200), "A", H())     # id 1 is taken in this frame
    assert near["id"] == 1 and far["id"] == 2 and again["id"] not in (1, 2)
    assert remove_nearest_person(frames[30], (131, 211)) and len(frames[30]["persons"]) == 2
    assert not remove_nearest_person(frames[30], (5000, 5000))
    set_ball(frames, 30, None, H())
    assert frames[30]["ball_visible"] is False and frames[30]["ball"] is None
    path = tmp_path / "gt.json"
    save_gt(path, frames)
    loaded = load_gt(path)
    assert sorted(loaded) == [0, 30] and loaded[0]["persons"][0]["m"] == [10.0, 20.0]
    assert load_gt(tmp_path / "missing.json") == {}
    assert assign_id([0, 0], [], set(), 9) == 9


def test_load_predictions(tmp_path):
    (tmp_path / "t.csv").write_text("frame,time_s,track_id,group,team,x_m,y_m,conf\n"
                                    "0,0,1,player_B_goal_keeper,B,5,6,0.9\n0,0,2,weird,-,1,1,0.5\n")
    (tmp_path / "f.csv").write_text("frame,time_s,ball_x_m,ball_y_m,ball_state,possession_team,possession_track_id\n"
                                    "0,0,,,none,-,\n2,0.07,3,4,detected,A,1\n")
    people, ball, poss = load_predictions(tmp_path / "t.csv", tmp_path / "f.csv")
    assert [p["role"] for p in people[0]] == ["K", "other"] and people[2] == []
    assert ball == {0: None, 2: (3.0, 4.0)} and poss == {0: "-", 2: "A"}


def body_case(click_rel):
    """One frame, two players 50 px tall; the GT click at click_rel of the box height (0 = top, 1 = feet).
    The metre positions are deliberately 4 m off, as body clicks projected with the homography would be."""
    boxes = [(100, 100, 120, 150), (300, 200, 320, 250)]
    gt = {0: {"done": True, "ball_visible": False, "ball": None, "possession": "A",
              "persons": [{"id": i, "role": r, "px": [(b[0] + b[2]) / 2, b[1] + click_rel * (b[3] - b[1])],
                           "m": [10 + 30 * i, 20 - 4]} for i, (r, b) in enumerate(zip("AB", boxes))]}}
    people = {0: [{"track_id": 7 + i, "role": r, "x": 10 + 30 * i, "y": 20, "box": b}
                  for i, (r, b) in enumerate(zip("AB", boxes))]}
    return gt, people, {0: None}, {0: "A"}


def test_body_anchor_matches_in_pixels():
    r = evaluate(*body_case(0.45), anchor="body")
    assert r["players"]["tp"] == 2 and r["players"]["f1"] == 1
    assert r["teams"]["role_accuracy"] == 1
    assert r["players"]["position_error_m"]["n"] == 0          # no metre error for body clicks
    assert r["players"]["match_distance_box_heights"]["mean"] == pytest.approx(0.05)
    assert r["click_height_median"] == pytest.approx(0.45)


def test_feet_anchor_warns_on_body_clicks():
    r = evaluate(*body_case(0.4), radius=5)
    assert r["anchor"] == "feet" and r["players"]["position_error_m"]["mean"] == pytest.approx(4)
    assert "WARNING" in format_report(r, "x")
    r = evaluate(*body_case(1.0))
    assert "WARNING" not in format_report(r, "x")


def test_body_anchor_needs_boxes():
    gt, people, ball, poss = body_case(0.5)
    for p in people[0]:
        p["box"] = None
    with pytest.raises(ValueError):
        evaluate(gt, people, ball, poss, anchor="body")
    assert click_height(gt, people) is None


def test_load_predictions_reads_boxes(tmp_path):
    t, fr = tmp_path / "t.csv", tmp_path / "f.csv"
    t.write_text("frame,time_s,track_id,group,team,x_m,y_m,conf,box_x1,box_y1,box_x2,box_y2\n"
                 "0,0,1,player_A,A,1,2,0.9,10,20,30,60\n", encoding="utf-8")
    fr.write_text("frame,time_s,ball_x_m,ball_y_m,ball_state,possession_team,possession_track_id\n"
                  "0,0,,,none,-,\n", encoding="utf-8")
    people, _, _ = load_predictions(t, fr)
    assert people[0][0]["box"] == (10, 20, 30, 60)
