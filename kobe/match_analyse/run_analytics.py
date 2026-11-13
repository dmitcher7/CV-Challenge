"""Analytics on the tracking output: distance, speed, speed zones, kilometres, attacking direction,
heatmaps, average positions, possession.

    python run_analytics.py --model best [--smooth-s 0.5] [--min-track-s 1.0]

Reads output/tracks_<model>.csv and output/frames_<model>.csv (from run_minimap.py),
writes everything to output/analytics/.
"""

import argparse

import pandas as pd

from analytics import (MAX_SPEED_KMH, average_positions, draw_average_positions, draw_heatmap, draw_zone_bars,
                       heatmap, player_table, possession_summary, smooth_track, team_summary)
from analytics.extra import extra_stats
from analytics.report import TEAM_NAMES, extra_markdown, md_table
from core.paths import OUTPUT_DIR


def build_report(model, table, teams, poss, tracks, args, extra=None, figures=None):
    clip = tracks["time_s"].max() - tracks["time_s"].min()
    extra_text = ("\n" + extra_markdown(extra, figures) + "\n") if extra else ""
    return f"""# Match analytics ({model})

Clip: {clip:.1f} s, {tracks['frame'].nunique()} processed frames, {len(table)} tracks of at least {args.min_track_s} s.
Positions smoothed with a {args.smooth_s} s centred moving average (no smoothing across gaps > 0.5 s).

## Per team
{md_table(teams)}

## Per track
{md_table(table.drop(columns=['speed_clipped']))}

## Possession
- Frames with possession: {poss['frames_with_possession']} of {poss['frames_total']}
- Share team A: {poss['share_A']:.0%}, team B: {poss['share_B']:.0%}
- Possession changes: {poss['changes']}
{extra_text}
## Figures
- heatmap_team_A_{model}.png, heatmap_team_B_{model}.png, heatmap_ball_{model}.png (team heatmaps with the attacking direction as arrow)
- average_positions_{model}.png
- zones_team_A_{model}.png, zones_team_B_{model}.png

## Beperkingen
- Homography noise makes positions jitter; we smooth, but a raw (unsmoothed) speed would be inflated.
- Speeds are estimates, not true speeds: max speed is capped at {MAX_SPEED_KMH:.0f} km/h
  ({int(table['speed_clipped'].sum())} track(s) hit the cap, which points to a tracking/homography artefact).
- The clip is short, so distances cover seconds, not a match; average positions are not a real formation.
- ID switches split a player's distance over several track ids, so per-track distance is a lower bound
  per player and the number of tracks can exceed 22.
- Speed zones, sprints, kilometres and attack counts inherit these limits; the attacking direction is a vote of
  three signals (see above) and shots are not counted.
"""


def analyze(model, tracks, frames, out, smooth_s=0.5, min_track_s=1.0):
    """Compute everything, write tables, figures and report_<model>.md into out. Returns a dict with the results."""
    out.mkdir(parents=True, exist_ok=True)
    table = player_table(tracks, smooth_s, min_track_s)
    teams = team_summary(table)
    poss = possession_summary(frames)
    extra = extra_stats(tracks, frames, table, smooth_s, min_track_s)
    table.to_csv(out / f"players_{model}.csv", index=False)
    teams.to_csv(out / f"teams_{model}.csv", index=False)
    extra["zones"].to_csv(out / f"zones_players_{model}.csv", index=False)
    extra["zone_teams"].to_csv(out / f"zones_teams_{model}.csv", index=False)
    extra["km_teams"].to_csv(out / f"km_teams_{model}.csv", index=False)
    extra["km_players"].to_csv(out / f"km_players_{model}.csv", index=False)

    kept = tracks[tracks["track_id"].isin(table["track_id"])].sort_values(["track_id", "time_s"])
    sign = extra["direction"]["sign"]
    figures = {}
    for team in ("A", "B"):
        xs, ys = [], []
        for _, t in kept[kept["team"] == team].groupby("track_id"):
            sx, sy = smooth_track(t["time_s"], t["x_m"], t["y_m"], smooth_s)
            xs.extend(sx)
            ys.extend(sy)
        draw_heatmap(heatmap(xs, ys), f"Heatmap team {team} ({model})",
                     out / f"heatmap_team_{team}_{model}.png", attack_direction=sign[team])
        row = extra["zone_teams"][extra["zone_teams"]["team"] == team].iloc[0]
        zones = extra["zones"][extra["zones"]["team"] == team]
        if len(zones):
            name = f"zones_team_{team}_{model}.png"
            draw_zone_bars(zones, row, TEAM_NAMES[team], f"Snelheidszones {TEAM_NAMES[team]} ({model})", out / name)
            figures[f"zones_{team}"] = name
    ball = frames.dropna(subset=["ball_x_m", "ball_y_m"])
    draw_heatmap(heatmap(ball["ball_x_m"], ball["ball_y_m"]), f"Heatmap ball ({model})",
                 out / f"heatmap_ball_{model}.png")
    draw_average_positions({t: average_positions(tracks, table, t) for t in ("A", "B")},
                           f"Average positions ({model})", out / f"average_positions_{model}.png")

    args = argparse.Namespace(smooth_s=smooth_s, min_track_s=min_track_s)
    report = build_report(model, table, teams, poss, tracks, args, extra, figures)
    (out / f"report_{model}.md").write_text(report, encoding="utf-8")
    return {"table": table, "teams": teams, "poss": poss, "extra": extra, "report": report,
            "figures": figures, "out": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="best", help="csv suffix (tracks_<model>.csv)")
    ap.add_argument("--smooth-s", type=float, default=0.5, help="smoothing window in seconds")
    ap.add_argument("--min-track-s", type=float, default=1.0, help="drop tracks shorter than this")
    args = ap.parse_args()

    tracks = pd.read_csv(OUTPUT_DIR / f"tracks_{args.model}.csv")
    frames = pd.read_csv(OUTPUT_DIR / f"frames_{args.model}.csv")
    res = analyze(args.model, tracks, frames, OUTPUT_DIR / "analytics", args.smooth_s, args.min_track_s)
    print(res["report"])
    print(f"Written to {res['out']}")


if __name__ == "__main__":
    main()
