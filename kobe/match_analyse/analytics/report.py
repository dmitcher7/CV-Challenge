"""Markdown text for the reports: report_<model>.md (analytics) and RAPPORT_<model>.md (run_all.py)."""

import pandas as pd

from .zones import SPRINT_KMH, SPRINT_MIN_S, ZONES, ZONE_KEYS

TEAM_NAMES = {"A": "PSG (A)", "B": "Inter (B)"}
ZONE_RANGE = {"standing": "< 2 km/h", "walking": "2-7 km/h", "jogging": "7-15 km/h",
              "running": "15-20 km/h", "sprinting": "> 20 km/h"}


def md_table(df):
    """DataFrame -> markdown table (no tabulate dependency)."""
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def zone_team_view(zone_teams):
    """Readable team table: per zone 'seconds (share of time)', distance in m, sprints."""
    rows = []
    for _, r in zone_teams.iterrows():
        row = {"team": TEAM_NAMES[r["team"]], "tracks": int(r["n_tracks"])}
        for key, label, *_ in ZONES:
            row[f"{label} ({ZONE_RANGE[key]})"] = f"{r[f'time_{key}_s']:.0f} s ({r[f'share_{key}']:.0%})"
        row["afstand sprint (m)"] = r["dist_sprinting_m"]
        row["sprints"] = int(r["n_sprints"])
        rows.append(row)
    return pd.DataFrame(rows)


def extra_markdown(extra, figures=None):
    """Dutch markdown sections for the extra statistics. figures = dict name -> relative png path (optional)."""
    figures = figures or {}
    d, a = extra["direction"], extra["attacks"]
    ev = d["evidence"]
    km = extra["km_teams"].copy()
    km["team"] = km["team"].map(TEAM_NAMES)
    km = km.rename(columns={"n_tracks": "tracks", "total_km": "totaal km (ondergrens)",
                            "mean_km_per_track": "gem. km per track",
                            "km_per_player_per_90_indicative": "km per speler per 90 min (indicatief)",
                            "clip_s": "cliplengte (s)"})
    att = pd.DataFrame([{"team": TEAM_NAMES[t], "balbezitfases": a[t]["phases"],
                         "ingangen laatste derde": a[t]["final_third_entries"], "aanvallen": a[t]["attacks"],
                         "ingangen 16 m-gebied": a[t]["box_entries"]} for t in ("A", "B")])
    lines = [
        "## Snelheidszones",
        f"Zones (bij benadering de gangbare voetbaldrempels; er is geen officiële standaard): "
        + ", ".join(f"{z[1]} {ZONE_RANGE[z[0]]}" for z in ZONES)
        + f". Een sprint = minstens {SPRINT_MIN_S:g} s aaneengesloten boven {SPRINT_KMH:g} km/h. "
        "Snelheid uit afgevlakte posities per stap.",
        "",
        md_table(zone_team_view(extra["zone_teams"])),
    ]
    for t in ("A", "B"):
        if f"zones_{t}" in figures:
            lines += ["", f"![snelheidszones {TEAM_NAMES[t]}]({figures[f'zones_{t}']})"]
    z = extra["zones"]
    if len(z):
        lines += ["", "Per track (tijd in s per zone):", "",
                  md_table(z[["track_id", "team"] + [f"time_{k}_s" for k in ZONE_KEYS] + ["n_sprints"]])]
    lines += [
        "", "## Kilometers",
        md_table(km),
        "",
        "Dit is een **ondergrens**: een ID-switch splitst een speler over meerdere tracks, tracks korter dan de minimumduur "
        "vallen weg en een speler telt alleen mee zolang hij getrackt wordt. De waarde per 90 minuten schaalt de afstandsratio "
        f"van een clip van {extra['clip_s']:.0f} s op en is **alleen indicatief**: een korte clip zegt weinig over een hele wedstrijd.",
        "",
        "## Aanvalsrichting en aanvallen",
        f"- **{d['text']['A']}**; {d['text']['B']}.",
        f"- Betrouwbaarheid: **{d['confidence']}** ({d['agreement']}).",
        f"- Keepers (gem. x in m): {ev['keeper']['keeper_mean_x'] or 'geen keeper gezien'} -> signaal {d['signals']['keeper']:+d}",
        f"- Gemiddelde x veldspelers (m): {ev['mean_x']['outfield_mean_x'] or 'onbekend'} -> signaal {d['signals']['mean_x']:+d} (zwak bewijs)",
        f"- Netto balverplaatsing in x bij balbezit (m): {ev['ball']['ball_net_dx_m']} -> signaal {d['signals']['ball']:+d}",
        "- Signaal +1 = PSG valt naar rechts aan (x=105), -1 = naar links (x=0), 0 = geen uitspraak; de keeper telt dubbel.",
        "",
        md_table(att),
        "",
        "Aanval = balbezitsfase waarin de bal het laatste derde (binnen 35 m van de aangevallen doellijn) binnenkomt; "
        "ingang 16 m-gebied = bal in het strafschopgebied (16,5 m x 40,3 m). **Schoten** zijn bewust niet geteld: ze zijn in "
        "deze trackingdata niet betrouwbaar van passes te onderscheiden.",
    ]
    return "\n".join(lines)
