"""Possession: nearest team player within the radius, hysteresis needs N frames to switch."""

from core.possession import (POSSESSION_SWITCH, POSSESSION_TIMEOUT, raw_possession, scaled,
                             stable_possession)


def _p(team, x, y, track):
    return {"team": team, "x_m": x, "y_m": y, "track_id": track}


def test_nearest_player_within_radius_has_the_ball():
    people = [_p("A", 10, 10, 1), _p("B", 11, 10, 2), _p("-", 10.2, 10, 3)]   # the referee is closest
    assert raw_possession(people, (10.3, 10)) == ("A", 1)
    assert raw_possession(people, (50, 50)) == (None, None)                    # nobody near
    assert raw_possession(people, None) == (None, None)
    assert raw_possession([], (1, 1)) == (None, None)


def test_hysteresis_needs_switch_frames():
    raw = {i: "A" for i in range(10)}
    raw.update({10 + i: "B" for i in range(POSSESSION_SWITCH)})
    out = stable_possession(raw)
    assert out[9] == "A"
    assert out[10 + POSSESSION_SWITCH - 2] == "A"          # one frame short: still A
    assert out[10 + POSSESSION_SWITCH - 1] == "B"          # N-th frame in a row: switch


def test_short_flicker_does_not_switch():
    raw = {i: "A" for i in range(30)}
    for i in (10, 11, 12):                                 # 3 frames of B < POSSESSION_SWITCH
        raw[i] = "B"
    assert set(stable_possession(raw).values()) == {"A"}


def test_interrupted_run_resets_the_counter():
    raw = {0: "A", 1: "B", 2: "B", 3: "B", 4: "A", 5: "B", 6: "B", 7: "B", 8: "B"}
    assert stable_possession(raw)[8] == "A"                # B never got 5 in a row


def test_timeout_clears_possession():
    raw = {0: "A"}
    raw.update({i: None for i in range(1, POSSESSION_TIMEOUT + 3)})
    out = stable_possession(raw)
    assert out[POSSESSION_TIMEOUT] == "A" and out[POSSESSION_TIMEOUT + 1] is None


def test_scaled_counts_for_stride():
    assert scaled(15, 1) == 15 and scaled(15, 2) == 8 and scaled(5, 10) == 1
