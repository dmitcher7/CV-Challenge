"""Tracking: ball Kalman filter (gate, gate fix, restart, convergence), interpolation in metres, ByteTrack ids, majority vote."""

import numpy as np

import core.tracking as tracking
from core.tracking import (BALL_GATE, BALL_MAX_MISSED, BallKalman, confirmed, interpolate_ball, remove_duplicates,
                           track_ball, track_people, vote_groups)


def test_kalman_converges_to_a_moving_ball():
    rng = np.random.default_rng(0)
    truth = lambda t: np.array([10 + 0.5 * t, 20 - 0.2 * t])   # 0.5 m/frame
    kf = BallKalman(truth(0))
    for t in range(1, 40):
        kf.predict()
        kf.update(truth(t) + rng.normal(0, 0.3, 2))
    assert np.linalg.norm(kf.x[:2] - truth(39)) < 0.5
    assert abs(kf.x[2] - 0.5) < 0.2 and abs(kf.x[3] + 0.2) < 0.2   # velocity learned (slowly: small Q)


def test_ball_track_gates_outliers():
    # Ball moves 0.3 m/frame along x; every frame also a confident "ball" far away (a boot)
    cands = {}
    for i in range(30):
        ball = np.array([10 + 0.3 * i, 30.0])
        cands[i] = [(ball, 0.6, ("ball", i)), (np.array([80.0, 5.0]), 0.5, ("boot", i))]
    track = track_ball(cands, list(range(30)), max_gap=15)
    assert len(track) == 30
    assert all(np.hypot(*(track[i][0] - [10 + 0.3 * i, 30.0])) < 0.5 for i in range(30))
    assert all(track[i][1][0] == "ball" for i in range(1, 30))   # the boot is never picked after the start


def test_interpolation_in_metres_fills_gaps_up_to_max_gap():
    measured = {0: ((0.0, 0.0), "b", 1), 10: ((10.0, 5.0), "b", 1), 40: ((40.0, 5.0), "b", 1)}
    track = interpolate_ball(measured, list(range(41)), max_gap=15)
    assert track[5][2] and np.allclose(track[5][0], (5.0, 2.5))   # metres, linear, flagged interpolated
    assert not track[0][2] and not track[10][2]
    assert 20 not in track                                         # gap 10..40 is 30 > max_gap: not filled
    assert set(range(11)) <= set(track)


def test_interpolation_never_crosses_segments():
    measured = {0: ((0.0, 0.0), None, 1), 5: ((50.0, 0.0), None, 2)}
    track = interpolate_ball(measured, list(range(6)), max_gap=15)
    assert set(track) == {0, 5}


def test_ball_track_gate_grows_while_ball_is_missing():
    # Ball visible, then 5 frames missing while it keeps moving 1 m/frame, then visible again
    cands = {i: [(np.array([float(i), 0.0]), 0.6, None)] for i in range(10)}
    cands.update({i: [(np.array([float(i), 0.0]), 0.6, None)] for i in range(15, 20)})
    track = track_ball(cands, list(range(20)), max_gap=15)
    assert 17 in track and not track[17][2]
    assert track[12][2]                                            # the gap is filled in metres
    assert abs(track[12][0][0] - 12.0) < 1e-6


def test_gate_constant_is_in_metres():
    assert 0 < BALL_GATE < 10


def _person(box, conf=0.9, name="player_A"):
    return {"box": box, "conf": conf, "name": name}


def test_track_people_keeps_ids_and_separates_players():
    frames = {}
    for i in range(10):
        frames[i] = [_person([100 + 5 * i, 100, 140 + 5 * i, 200]), _person([600 - 5 * i, 300, 640 - 5 * i, 400])]
    track_people(frames, conf=0.3)
    ids_a = {frames[i][0]["track"] for i in range(2, 10)}
    ids_b = {frames[i][1]["track"] for i in range(2, 10)}
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b and None not in ids_a | ids_b


def test_remove_duplicates_keeps_most_confident():
    kept = remove_duplicates([_person([0, 0, 50, 100], 0.6, "player_B"), _person([2, 2, 52, 101], 0.9, "player_A"),
                              _person([300, 0, 350, 100], 0.5)])
    assert len(kept) == 2 and kept[0]["name"] == "player_A"


def test_vote_groups_majority_by_confidence():
    # One track: 8 frames player_A, 2 frames flickering to player_B -> the whole track is player_A
    frames = {i: [{"track": 1, "conf": 0.8, "name": "player_B" if i in (3, 7) else "player_A"}] for i in range(10)}
    result = vote_groups(frames, lambda d: d["name"])
    assert result[1][0] == "player_A" and result[1][1] > 0.7
    assert all(frames[i][0]["group"] == "player_A" for i in frames)


def _kicked_ball(gap=10, conf_after=0.22):
    """0.3 m/frame for 10 frames, then a kick: `gap` frames unseen, then weak detections at 1.4 m/frame."""
    cands = {i: [(np.array([10 + 0.3 * i, 30.0]), 0.6, ("ball", i))] for i in range(10)}
    for i in range(10 + gap, 24 + gap):
        cands[i] = [(np.array([12.7 + 1.4 * (i - 9), 30.0]), conf_after, ("ball", i))]
    return cands


def test_gate_fix_follows_a_kicked_ball_the_old_gate_lost(monkeypatch):
    cands = _kicked_ball()
    ids = list(range(34))
    after = [i for i in cands if i >= 10]
    track = track_ball(cands, ids, max_gap=15)
    assert all(i in track and not track[i][2] for i in after)            # every detection after the kick is used
    assert all(np.allclose(track[i][0], cands[i][0][0]) for i in after)

    monkeypatch.setattr(tracking, "BALL_GATE_GROWTH", 1.0)               # the settings before the gate fix
    monkeypatch.setattr(tracking, "BALL_MAX_MISSED", 15)
    monkeypatch.setattr(tracking, "BALL_RESTART_CONF", 2.0)
    old = track_ball(cands, ids, max_gap=15)
    assert not any(i in old for i in after)                               # too weak to restart: ball lost


def test_restart_needs_a_confirming_candidate():
    # Ball (conf 0.3) moving 0.3 m/frame; in frame 10 it is missing and a very confident outlier appears once
    cands = {i: [(np.array([10 + 0.3 * i, 30.0]), 0.3, ("ball", i))] for i in range(25) if i != 10}
    cands[10] = [(np.array([60.0, 10.0]), 0.9, ("boot", 10))]
    track = track_ball(cands, list(range(25)), max_gap=15)
    assert track[10][2] and abs(track[10][0][0] - 13.0) < 1e-6          # interpolated on the ball path
    assert all(track[i][1][0] == "ball" for i in range(25) if i != 10)   # the outlier never hijacks the track


def test_confirmed_restart_jumps_to_the_new_ball():
    # The ball is kicked out of the gate: confident candidates far away that move on at 1 m/frame
    cands = {i: [(np.array([10 + 0.3 * i, 30.0]), 0.6, ("a", i))] for i in range(10)}
    cands.update({i: [(np.array([40.0 + (i - 10), 10.0]), 0.5, ("b", i))] for i in range(10, 20)})
    track = track_ball(cands, list(range(20)), max_gap=15)
    assert all(i in track and track[i][1] == ("b", i) for i in range(10, 20))   # restarted at once, not 30 frames later
    assert not any(track[i][2] for i in range(10, 20))
    assert confirmed(cands[10][0], 10, cands, [11, 12, 13])
    assert not confirmed(cands[10][0], 10, {10: cands[10]}, [11, 12, 13])


def test_dead_ball_stretches_stay_empty_beyond_max_gap():
    # Ball in the net: no candidates for 25 frames (less than BALL_MAX_MISSED, more than max_gap), then back
    cands = {i: [(np.array([100.0, 34.0]), 0.6, None)] for i in range(10)}
    cands.update({i: [(np.array([100.0, 34.0]), 0.6, None)] for i in range(35, 45)})
    track = track_ball(cands, list(range(45)), max_gap=15)
    assert BALL_MAX_MISSED > 25
    assert not any(i in track for i in range(10, 35))                    # nothing invented in the gap
    assert all(i in track for i in range(35, 45))
    # and a clip that ends without a ball: no extrapolation after the last detection
    assert max(track_ball({i: cands[i] for i in range(10)}, list(range(45)), max_gap=15)) == 9
