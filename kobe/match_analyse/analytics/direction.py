"""Which goal each team attacks, and simple attack counts (final third, penalty box).

Direction = combination of three signals that each vote for "team A attacks the right goal (x=105)" or
"... the left goal (x=0)": (a) where the goalkeepers stand, (b) where the outfield players of both teams
stand on average, (c) the direction the ball travels while a team has possession.
Shots are out of scope: a shot cannot be told apart reliably from a pass in this tracking data.
"""

import numpy as np

from core.pitch import BOX_DEPTH, BOX_WIDTH, LENGTH, WIDTH

HALF = LENGTH / 2
FINAL_THIRD_M = 35.0           # ball within this distance of the attacked goal line = final third
BOX_Y = (WIDTH / 2 - BOX_WIDTH / 2, WIDTH / 2 + BOX_WIDTH / 2)   # penalty area spans these y values
KEEPER_MIN_OFFSET_M = 10.0     # a keeper closer than this to the halfway line says nothing
MEAN_X_MIN_DIFF_M = 2.0        # outfield mean x of the two teams must differ at least this much
BALL_MIN_NET_DX_M = 3.0        # net ball travel under possession must be at least this much
BALL_MAX_STEP_M = 5.0          # ball jumps larger than this between frames are tracking artefacts
PHASE_GAP_FRAMES = 15          # a possession phase survives up to this many frames without possession ("-")
WEIGHTS = {"keeper": 2.0, "mean_x": 1.0, "ball": 1.0}   # keeper is the most direct evidence


def keeper_signal(tracks):
    """+1 if the goalkeepers say team A attacks right (B's keeper stands at the right), -1 left, 0 unknown."""
    is_keeper = tracks["group"].astype(str).str.contains("goal_keeper") & tracks["team"].isin(["A", "B"])
    votes, detail = [], {}
    for team, g in tracks[is_keeper].groupby("team"):
        mx = float(g["x_m"].mean())
        detail[team] = round(mx, 1)
        if abs(mx - HALF) >= KEEPER_MIN_OFFSET_M:
            a_attacks_right = (mx > HALF) == (team == "B")   # B keeper right, or A keeper left
            votes.append(1 if a_attacks_right else -1)
    return int(np.sign(sum(votes))), {"keeper_mean_x": detail}


def mean_x_signal(tracks):
    """+1 if team A's outfield players stand more to the left than B's on average (own-half bias: A then
    defends the left and attacks right), -1 for the reverse, 0 if the difference is too small. Weak evidence."""
    out = tracks[~tracks["group"].astype(str).str.contains("goal_keeper")]
    a, b = out[out["team"] == "A"]["x_m"], out[out["team"] == "B"]["x_m"]
    if a.empty or b.empty:
        return 0, {"outfield_mean_x": {}}
    ma, mb = float(a.mean()), float(b.mean())
    diff = mb - ma
    sign = 0 if abs(diff) < MEAN_X_MIN_DIFF_M else (1 if diff > 0 else -1)
    return sign, {"outfield_mean_x": {"A": round(ma, 1), "B": round(mb, 1)}}


def ball_net_dx(frames):
    """Net x travel (m) of the ball while each team has possession (consecutive frames, same team, ball known)."""
    f = frames.sort_values("frame")
    team = f["possession_team"].to_numpy(dtype=object)
    bx = f["ball_x_m"].to_numpy(float)
    net = {"A": 0.0, "B": 0.0}
    for i in range(1, len(f)):
        if team[i] in net and team[i] == team[i - 1] and not (np.isnan(bx[i]) or np.isnan(bx[i - 1])):
            d = bx[i] - bx[i - 1]
            if abs(d) <= BALL_MAX_STEP_M:
                net[team[i]] += float(d)
    return net


def ball_signal(frames):
    """+1 if the ball moves right under A's possession / left under B's, -1 for the reverse, 0 unclear."""
    net = ball_net_dx(frames)
    score = net["A"] - net["B"]          # A moving right and B moving left both say "A attacks right"
    sign = 0 if abs(score) < BALL_MIN_NET_DX_M else (1 if score > 0 else -1)
    return sign, {"ball_net_dx_m": {k: round(v, 1) for k, v in net.items()}}


def attack_direction(tracks, frames):
    """Decide the attacking direction of both teams from the three signals (weighted vote).

    Returns a dict: a_attacks ("right"/"left"/None), goal_x per team (105 or 0), sign per team (+1 = towards
    x=105), the signals (+1 = A attacks right, -1 = left, 0 = no info), evidence numbers, an agreement note,
    confidence ("hoog" = >= 2 signals and all agree; "middel" = signals conflict but one side wins;
    "laag" = one signal only or a tie; "onbekend" = no signal) and a Dutch sentence per team.
    """
    signals, evidence = {}, {}
    for name, fn, data in (("keeper", keeper_signal, tracks), ("mean_x", mean_x_signal, tracks),
                           ("ball", ball_signal, frames)):
        signals[name], evidence[name] = fn(data)
    active = {k: v for k, v in signals.items() if v != 0}
    score = sum(WEIGHTS[k] * v for k, v in active.items())
    decided = int(np.sign(score))
    agree = [k for k, v in active.items() if v == decided] if decided else []
    against = [k for k, v in active.items() if v == -decided] if decided else []
    if not active:
        conf = "onbekend"
    elif not decided or len(active) == 1:
        conf = "laag"
    elif not against:
        conf = "hoog"
    else:
        conf = "middel"
    goal_x = {"A": None, "B": None} if not decided else \
        {"A": LENGTH if decided > 0 else 0.0, "B": 0.0 if decided > 0 else LENGTH}

    def sentence(name, g):
        if g is None:
            return f"{name}: aanvalsrichting onbekend"
        return f"{name} valt aan richting {'rechter' if g > 0 else 'linker'} doel (x={g:g})"

    return {
        "a_attacks": None if not decided else ("right" if decided > 0 else "left"), "goal_x": goal_x,
        "sign": {"A": decided, "B": -decided}, "signals": signals, "evidence": evidence,
        "agree": agree, "against": against, "agreement": f"{len(agree)} van {len(active)} signalen eens",
        "confidence": conf, "text": {"A": sentence("PSG", goal_x["A"]), "B": sentence("Inter", goal_x["B"])},
    }


def in_final_third(x, goal_x):
    return abs(x - goal_x) <= FINAL_THIRD_M


def in_box(x, y, goal_x):
    return abs(x - goal_x) <= BOX_DEPTH and BOX_Y[0] <= y <= BOX_Y[1]


def possession_phases(frames, max_gap=PHASE_GAP_FRAMES):
    """(frames sorted by frame, phases): each phase = {"team", "rows"} with the row positions of its frames.
    A phase ends when the other team gets the ball or when there is no possession for more than max_gap frames."""
    f = frames.sort_values("frame").reset_index(drop=True)
    team = f["possession_team"].to_numpy(dtype=object)
    phases, cur, last = [], None, -10 ** 9
    for i, t in enumerate(team):
        if t in ("A", "B"):
            if cur is not None and cur["team"] == t and i - last <= max_gap + 1:
                cur["rows"].append(i)
            else:
                cur = {"team": t, "rows": [i]}
                phases.append(cur)
            last = i
    return f, phases


def attack_counts(frames, goal_x, max_gap=PHASE_GAP_FRAMES):
    """Per team: possession phases, final-third entries, attacks (phases with >= 1 final-third entry), box entries.

    An entry is a ball position inside the zone right after one outside it (or at the start of a phase).
    Frames without a known ball position are skipped. goal_x = {"A": x or None, "B": x or None}.
    """
    f, phases = possession_phases(frames, max_gap)
    bx, by = f["ball_x_m"].to_numpy(float), f["ball_y_m"].to_numpy(float)
    out = {t: {"phases": 0, "final_third_entries": 0, "attacks": 0, "box_entries": 0} for t in ("A", "B")}
    for ph in phases:
        t, g = ph["team"], goal_x.get(ph["team"])
        out[t]["phases"] += 1
        if g is None:
            continue
        ft_in = box_in = False
        ft_entries = box_entries = 0
        for i in ph["rows"]:
            if np.isnan(bx[i]) or np.isnan(by[i]):
                continue
            ft, bx_in = in_final_third(bx[i], g), in_box(bx[i], by[i], g)
            ft_entries += int(ft and not ft_in)
            box_entries += int(bx_in and not box_in)
            ft_in, box_in = ft, bx_in
        out[t]["final_third_entries"] += ft_entries
        out[t]["attacks"] += int(ft_entries > 0)
        out[t]["box_entries"] += box_entries
    return out
