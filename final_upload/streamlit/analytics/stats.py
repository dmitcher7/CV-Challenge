"""Per-track smoothing, distance and speed, plus team and possession summaries.

Homography noise makes raw positions jitter (tens of cm per frame), so summing raw
steps inflates distance and speed. We smooth in time first. Tracks are split into
segments at time gaps larger than MAX_GAP_S; nothing is smoothed or measured across a gap.
"""

import numpy as np
import pandas as pd

MAX_SPEED_KMH = 36.0   # physically plausible sprint cap for a footballer
MAX_GAP_S = 0.5        # larger time gaps split a track into separate segments
MS_TO_KMH = 3.6


def split_segments(t, max_gap_s=MAX_GAP_S):
    """Index ranges (start, stop) of runs in sorted times t without a gap > max_gap_s."""
    t = np.asarray(t, float)
    if len(t) == 0:
        return []
    cuts = np.where(np.diff(t) > max_gap_s + 1e-9)[0] + 1
    bounds = np.concatenate([[0], cuts, [len(t)]])
    return list(zip(bounds[:-1], bounds[1:]))


def _smooth_segment(t, xy, smooth_s):
    """Centred moving average over smooth_s seconds; the window shrinks symmetrically
    towards the segment ends (keeps a constant-velocity track exactly linear)."""
    out = np.empty_like(xy)
    half = smooth_s / 2.0
    for i, ti in enumerate(t):
        h = min(half, ti - t[0], t[-1] - ti)
        mask = np.abs(t - ti) <= h + 1e-9
        out[i] = xy[mask].mean(axis=0)
    return out


def smooth_track(time_s, x, y, smooth_s=0.5, max_gap_s=MAX_GAP_S):
    """Smoothed (x, y) for one track; arrays keep the input order (must be sorted by time)."""
    t = np.asarray(time_s, float)
    xy = np.stack([np.asarray(x, float), np.asarray(y, float)], axis=1)
    out = xy.copy()
    if smooth_s > 0:
        for a, b in split_segments(t, max_gap_s):
            out[a:b] = _smooth_segment(t[a:b], xy[a:b], smooth_s)
    return out[:, 0], out[:, 1]


def track_stats(time_s, x, y, smooth_s=0.5, max_gap_s=MAX_GAP_S, max_kmh=MAX_SPEED_KMH):
    """Distance (m), mean and max speed (km/h) of one track from smoothed positions.

    mean = distance / time covered by segments (gaps excluded). max = peak speed over a >= 0.5 s span,
    capped at max_kmh; `clipped` says whether the cap was hit (likely an artefact, not a sprint).
    """
    t = np.asarray(time_s, float)
    sx, sy = smooth_track(t, x, y, smooth_s, max_gap_s)
    dist, covered, speeds = 0.0, 0.0, []
    for a, b in split_segments(t, max_gap_s):
        if b - a < 2:
            continue
        step = np.hypot(np.diff(sx[a:b]), np.diff(sy[a:b]))
        dt = np.diff(t[a:b])
        ok = dt > 0
        dist += step[ok].sum()
        covered += dt[ok].sum()
        # peak speed over a >= 0.5 s span: single-step speeds on a smoothed track stay noisy
        ts = t[a:b]
        j = np.searchsorted(ts, ts + 0.5 - 1e-9)
        valid = j < len(ts)
        if valid.any():
            i = np.where(valid)[0]
            span = np.hypot(sx[a:b][j[i]] - sx[a:b][i], sy[a:b][j[i]] - sy[a:b][i])
            speeds.extend((span / (ts[j[i]] - ts[i]) * MS_TO_KMH).tolist())
        else:
            speeds.extend((step[ok] / dt[ok] * MS_TO_KMH).tolist())
    max_raw = float(np.max(speeds)) if speeds else 0.0
    return {
        "distance_m": float(dist),
        "mean_kmh": float(dist / covered * MS_TO_KMH) if covered > 0 else 0.0,
        "max_kmh": min(max_raw, max_kmh),
        "clipped": max_raw > max_kmh,
    }


def player_table(tracks, smooth_s=0.5, min_track_s=1.0, max_gap_s=MAX_GAP_S, max_kmh=MAX_SPEED_KMH):
    """One row per track with duration >= min_track_s (all groups, referee included)."""
    rows = []
    for tid, g in tracks.sort_values("time_s").groupby("track_id"):
        duration = float(g["time_s"].iloc[-1] - g["time_s"].iloc[0])
        if duration < min_track_s:
            continue
        s = track_stats(g["time_s"], g["x_m"], g["y_m"], smooth_s, max_gap_s, max_kmh)
        rows.append({
            "track_id": int(tid), "group": g["group"].mode().iloc[0], "team": g["team"].mode().iloc[0],
            "n_frames": len(g), "duration_s": round(duration, 2),
            "distance_m": round(s["distance_m"], 1), "mean_kmh": round(s["mean_kmh"], 2),
            "max_kmh": round(s["max_kmh"], 1), "speed_clipped": s["clipped"],
        })
    cols = ["track_id", "group", "team", "n_frames", "duration_s", "distance_m", "mean_kmh",
            "max_kmh", "speed_clipped"]
    return pd.DataFrame(rows, columns=cols)


def team_summary(table):
    """Per team (A, B): number of tracks, total/mean distance, mean speed, max speed."""
    rows = []
    for team in ("A", "B"):
        t = table[table["team"] == team]
        rows.append({
            "team": team, "n_tracks": len(t), "total_distance_m": round(t["distance_m"].sum(), 1),
            "mean_distance_m": round(t["distance_m"].mean(), 1) if len(t) else 0.0,
            "mean_kmh": round(t["mean_kmh"].mean(), 2) if len(t) else 0.0,
            "max_kmh": round(t["max_kmh"].max(), 1) if len(t) else 0.0,
        })
    return pd.DataFrame(rows)


def possession_summary(frames):
    """Share per team over frames with possession, and the number of possession changes."""
    team = frames["possession_team"].dropna()
    team = team[team.isin(["A", "B"])]
    n = len(team)
    changes = int((team.values[1:] != team.values[:-1]).sum()) if n > 1 else 0
    return {
        "frames_total": len(frames), "frames_with_possession": n,
        "share_A": float((team == "A").sum() / n) if n else 0.0,
        "share_B": float((team == "B").sum() / n) if n else 0.0,
        "changes": changes,
    }
