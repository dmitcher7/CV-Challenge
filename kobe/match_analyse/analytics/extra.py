"""The extra statistics in one call: speed zones, kilometres, attacking direction and attack counts.

Used by run_analytics.py, run_all.py and the Streamlit app, so they all show the same numbers.
"""

from .direction import attack_counts, attack_direction
from .zones import kilometres, team_zone_summary, zone_table


def extra_stats(tracks, frames, table, smooth_s=0.5, min_track_s=1.0):
    """tracks/frames = the csv DataFrames, table = stats.player_table(tracks, ...).

    Returns a dict: zones (per track), zone_teams, km_teams, km_players, direction (see
    direction.attack_direction), attacks (see direction.attack_counts), clip_s.
    """
    clip_s = float(tracks["time_s"].max() - tracks["time_s"].min()) if len(tracks) else 0.0
    zones = zone_table(tracks, smooth_s, min_track_s)
    km_teams, km_players = kilometres(table, clip_s)
    direction = attack_direction(tracks, frames)
    return {
        "zones": zones, "zone_teams": team_zone_summary(zones), "km_teams": km_teams,
        "km_players": km_players, "direction": direction,
        "attacks": attack_counts(frames, direction["goal_x"]), "clip_s": clip_s,
    }
