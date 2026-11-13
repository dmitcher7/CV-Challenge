"""Speed zones ("statische beweging"), sprint counting and kilometres per track and per team.

Speeds come from the same smoothed positions as stats.py (no smoothing across gaps). Every step between
two consecutive samples has a speed, a duration and a length; time and distance are summed per zone.
"""

import numpy as np
import pandas as pd

from .stats import MAX_GAP_S, MS_TO_KMH, smooth_track, split_segments

# Speed zones in km/h: (key, Dutch label, lower bound inclusive, upper bound exclusive).
# These are common (approximate) football thresholds as used in sports-science and tracking-provider
# reports; there is no single official standard, other sources use e.g. 19.8 or 25.2 km/h for sprinting.
ZONES = (
    ("standing", "stilstaand", 0.0, 2.0),
    ("walking", "wandelen", 2.0, 7.0),
    ("jogging", "joggen", 7.0, 15.0),
    ("running", "lopen", 15.0, 20.0),
    ("sprinting", "sprint", 20.0, np.inf),
)
ZONE_KEYS = [z[0] for z in ZONES]
ZONE_LABELS = {z[0]: z[1] for z in ZONES}
SPRINT_KMH = 20.0       # lower bound of the sprint zone
SPRINT_MIN_S = 1.0      # a sprint = at least this long continuously above SPRINT_KMH
SPEED_SPAN_S = 0.5      # speed of a step = displacement over (at most) this span centred on the step (per-step is too noisy)
MATCH_S = 90 * 60.0     # seconds of a match, only used for the indicative "per 90 minutes" extrapolation


def classify_speed(kmh):
    """Zone key for one speed or an array of speeds (km/h)."""
    arr = np.asarray(kmh, float)
    idx = np.clip(np.searchsorted([z[2] for z in ZONES], arr, side="right") - 1, 0, len(ZONES) - 1)
    if arr.ndim == 0:
        return ZONE_KEYS[int(idx)]
    return np.array(ZONE_KEYS, dtype=object)[idx]


def step_table(time_s, x, y, smooth_s=0.5, max_gap_s=MAX_GAP_S):
    """Per step of one track: DataFrame(dt_s, dist_m, kmh, new_segment) from smoothed positions.
    new_segment marks the first step after a gap (a sprint never continues across it)."""
    t = np.asarray(time_s, float)
    sx, sy = smooth_track(t, x, y, smooth_s, max_gap_s)
    parts = []
    for a, b in split_segments(t, max_gap_s):
        if b - a < 2:
            continue
        ts, xs, ys = t[a:b], sx[a:b], sy[a:b]
        dt = np.diff(ts)
        dist = np.hypot(np.diff(xs), np.diff(ys))
        ok = dt > 0
        if not ok.any():
            continue
        # local speed: displacement over a span centred on the step, shrinking towards the segment ends
        mid = (ts[:-1] + ts[1:]) / 2
        half = np.maximum(np.minimum.reduce([np.full(len(mid), SPEED_SPAN_S / 2), mid - ts[0], ts[-1] - mid]), dt / 2)
        lo, hi = mid - half, mid + half
        span = np.hypot(np.interp(hi, ts, xs) - np.interp(lo, ts, xs), np.interp(hi, ts, ys) - np.interp(lo, ts, ys))
        kmh = span / np.where(ok, hi - lo, 1.0) * MS_TO_KMH
        seg = pd.DataFrame({"dt_s": dt[ok], "dist_m": dist[ok], "kmh": kmh[ok]})
        seg["new_segment"] = False
        seg.iloc[0, seg.columns.get_loc("new_segment")] = True
        parts.append(seg)
    if not parts:
        return pd.DataFrame({"dt_s": [], "dist_m": [], "kmh": [], "new_segment": []})
    return pd.concat(parts, ignore_index=True)


def count_sprints(steps, sprint_kmh=SPRINT_KMH, min_s=SPRINT_MIN_S):
    """Number of runs of consecutive steps above sprint_kmh lasting at least min_s seconds."""
    n, run = 0, 0.0
    for dt, kmh, new in zip(steps["dt_s"], steps["kmh"], steps["new_segment"]):
        if new:                                   # a gap ends any running sprint
            n += int(run >= min_s - 1e-9)
            run = 0.0
        if kmh >= sprint_kmh:
            run += dt
        else:
            n += int(run >= min_s - 1e-9)
            run = 0.0
    return n + int(run >= min_s - 1e-9)


def track_zones(time_s, x, y, smooth_s=0.5, max_gap_s=MAX_GAP_S):
    """Time (s) and distance (m) per zone plus the sprint count of one track."""
    steps = step_table(time_s, x, y, smooth_s, max_gap_s)
    zone = np.asarray(classify_speed(steps["kmh"].to_numpy()), dtype=object).reshape(-1)
    out = {}
    for key in ZONE_KEYS:
        m = zone == key
        out[f"time_{key}_s"] = float(steps["dt_s"].to_numpy()[m].sum())
        out[f"dist_{key}_m"] = float(steps["dist_m"].to_numpy()[m].sum())
    out["n_sprints"] = count_sprints(steps)
    return out


def zone_table(tracks, smooth_s=0.5, min_track_s=1.0, max_gap_s=MAX_GAP_S):
    """One row per track (duration >= min_track_s, same filter as stats.player_table) with zone times/distances."""
    cols = ["track_id", "group", "team"] + [f"time_{k}_s" for k in ZONE_KEYS] \
        + [f"dist_{k}_m" for k in ZONE_KEYS] + ["n_sprints"]
    rows = []
    for tid, g in tracks.sort_values("time_s").groupby("track_id"):
        if float(g["time_s"].iloc[-1] - g["time_s"].iloc[0]) < min_track_s:
            continue
        z = track_zones(g["time_s"], g["x_m"], g["y_m"], smooth_s, max_gap_s)
        rows.append({"track_id": int(tid), "group": g["group"].mode().iloc[0],
                     "team": g["team"].mode().iloc[0], **{k: (v if k == "n_sprints" else round(v, 2))
                                                          for k, v in z.items()}})
    return pd.DataFrame(rows, columns=cols)


def team_zone_summary(zones):
    """Per team (A, B): tracks, time and distance per zone, share of time per zone, sprint count."""
    rows = []
    for team in ("A", "B"):
        t = zones[zones["team"] == team]
        total_t = sum(t[f"time_{k}_s"].sum() for k in ZONE_KEYS)
        row = {"team": team, "n_tracks": len(t), "n_sprints": int(t["n_sprints"].sum())}
        for k in ZONE_KEYS:
            row[f"time_{k}_s"] = round(float(t[f"time_{k}_s"].sum()), 1)
            row[f"dist_{k}_m"] = round(float(t[f"dist_{k}_m"].sum()), 1)
            row[f"share_{k}"] = round(float(t[f"time_{k}_s"].sum() / total_t), 3) if total_t > 0 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def kilometres(table, clip_s):
    """Total km per team (sum over kept tracks) and per player track, with a per-90-minutes extrapolation.

    Lower bound: ID switches split one player over several tracks, tracks shorter than min_track_s are
    dropped and a player is only counted while he is tracked. The extrapolation scales each track's
    distance rate to 90 minutes and is INDICATIVE only: a 22 s clip says little about a whole match.
    Returns (per_team DataFrame, per_player DataFrame).
    """
    t = table[table["team"].isin(["A", "B"])].copy()
    t["km"] = t["distance_m"] / 1000.0
    dur = t["duration_s"].to_numpy(float)
    t["km_per_90_indicative"] = np.where(dur > 0, t["distance_m"] / np.where(dur > 0, dur, 1) * MATCH_S / 1000.0, 0.0)
    rows = []
    for team in ("A", "B"):
        g = t[t["team"] == team]
        rows.append({
            "team": team, "n_tracks": len(g), "total_km": round(float(g["km"].sum()), 3),
            "mean_km_per_track": round(float(g["km"].mean()), 3) if len(g) else 0.0,
            "km_per_player_per_90_indicative": round(float(g["km_per_90_indicative"].mean()), 1) if len(g) else 0.0,
            "clip_s": round(float(clip_s), 1),
        })
    players = t[["track_id", "team", "group", "duration_s", "km", "km_per_90_indicative"]].round(3)
    return pd.DataFrame(rows), players.reset_index(drop=True)
