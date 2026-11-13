"""Ball possession: who is closest to the ball, smoothed over time.

raw_possession(): per frame the team of the nearest player within POSSESSION_RADIUS of the ball.
stable_possession(): possession only changes when another team has the ball for POSSESSION_SWITCH
frames in a row (hysteresis), and becomes "no team" after POSSESSION_TIMEOUT frames without anyone near.
"""

from math import ceil

import numpy as np

POSSESSION_RADIUS = 2.0   # metres: the nearest player within this distance "has" the ball
POSSESSION_SWITCH = 5     # frames another team must have the ball before possession changes
POSSESSION_TIMEOUT = 15   # frames without any team near the ball before possession becomes "-"


def scaled(frames, stride):
    """A number of frames -> a number of processed steps when only every stride-th frame is processed."""
    return max(1, ceil(frames / stride))


def raw_possession(people, ball_xy, radius=POSSESSION_RADIUS):
    """(team, track id) of the nearest team player within radius metres of the ball, else (None, None).

    people: [{"team": "A"/"B"/"-", "track_id", "x_m", "y_m"}]; referees ("-") never have the ball.
    """
    players = [p for p in people if p["team"] in ("A", "B")]
    if ball_xy is None or not players:
        return None, None
    dist = [np.hypot(p["x_m"] - ball_xy[0], p["y_m"] - ball_xy[1]) for p in players]
    i = int(np.argmin(dist))
    return (players[i]["team"], players[i]["track_id"]) if dist[i] <= radius else (None, None)


def stable_possession(raw, switch=POSSESSION_SWITCH, timeout=POSSESSION_TIMEOUT):
    """Smooth per-frame possession {frame: team or None} -> {frame: team or None} (hysteresis).

    switch / timeout are counted in processed frames (use scaled() when frames are skipped).
    """
    stable, candidate, run, since_seen, out = None, None, 0, 0, {}
    for idx in sorted(raw):
        team = raw[idx]
        if team is None:
            since_seen += 1
            if since_seen > timeout:
                stable = None
        else:
            since_seen = 0
            if stable is None:
                stable = team
            elif team == stable:
                run = 0
            else:
                run = run + 1 if team == candidate else 1
                candidate = team
                if run >= switch:
                    stable, run = team, 0
        out[idx] = stable
    return out
