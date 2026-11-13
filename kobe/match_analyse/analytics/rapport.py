"""The full Dutch RAPPORT_<model>.md written by run_all.py."""

from .report import TEAM_NAMES, extra_markdown, md_table


def eval_summary(res):
    """Short Dutch bullet list of the key numbers of one evaluate() result (see evaluation/metrics.py)."""
    p, t, b, pos, tr = res["players"], res["teams"], res["ball"], res["possession"], res.get("tracking", {})
    lines = [f"- Gelabelde frames: {res['model_frames']['labelled']}, vergeleken: {res['model_frames']['matched']}",
             f"- Spelers: precision {p['precision']:.3f}, recall {p['recall']:.3f}, F1 {p['f1']:.3f} "
             f"(TP {p['tp']}, FP {p['fp']}, FN {p['fn']})"]
    e = p["position_error_m"]
    if e["n"]:
        lines.append(f"- Positiefout spelers: mediaan {e['median']:.2f} m, gemiddeld {e['mean']:.2f} m, p90 {e['p90']:.2f} m")
    lines.append(f"- Teams: nauwkeurigheid {t['team_accuracy_merged']:.3f}, rol {t['role_accuracy']:.3f} (n={t['n']})")
    be = b["position_error_m"]
    ball = f"- Bal: detectiegraad {b['detection_rate']:.2f}"
    if be["n"]:
        ball += f", positiefout mediaan {be['median']:.2f} m (n={be['n']}), binnen 3 m: {b['within_3m']:.2f}"
    lines.append(ball)
    lines.append(f"- Balbezit: nauwkeurigheid {pos['accuracy']:.2f} (n={pos['n']})")
    if tr:
        lines.append(f"- Tracking: {tr['id_switches']} ID-switches (grove schatting), IDF1-lite {tr['idf1_lite']:.2f}")
    return "\n".join(lines)


def rapport_markdown(model, clip_s, n_frames, table, teams, poss, extra, evals, figures, video_name=None, params=""):
    """Dutch report text. Figure paths in `figures` are relative to analytics/ (zones_*), the report lives in output/.
    evals = list of (label, result dict or None, md filename or None, reason when skipped)."""
    d, att, km = extra["direction"], extra["attacks"], extra["km_teams"].set_index("team")
    teams_view = teams.copy()
    teams_view["team"] = teams_view["team"].map(TEAM_NAMES)
    eval_lines = []
    for label, res, md, reason in evals:
        eval_lines += [f"### {label}", ""]
        if res:
            eval_lines += [eval_summary(res), "", f"Volledig rapport: [{md}](evaluation/{md})", ""]
        else:
            eval_lines += [f"Overgeslagen: {reason}", ""]
    lines = [
        f"# Rapport wedstrijdanalyse ({model})", "",
        "## Samenvatting",
        f"- Clip: {clip_s:.1f} s, {n_frames} verwerkte frames, {len(table)} tracks van minstens 1 s.",
        f"- Balbezit: PSG {poss['share_A']:.0%}, Inter {poss['share_B']:.0%}, {poss['changes']} wissels.",
        f"- {d['text']['A']} (betrouwbaarheid {d['confidence']}).",
        f"- Aanvallen (bal in laatste derde): PSG {att['A']['attacks']}, Inter {att['B']['attacks']}; "
        f"ingangen 16 m-gebied: PSG {att['A']['box_entries']}, Inter {att['B']['box_entries']}.",
        f"- Totaal gelopen (ondergrens): PSG {km.loc['A', 'total_km']:.2f} km, Inter {km.loc['B', 'total_km']:.2f} km.",
        "",
        f"Gegenereerd met `run_all.py` ({params})." if params else "",
        "", "## Balbezit",
        f"- Frames met balbezit: {poss['frames_with_possession']} van {poss['frames_total']}",
        f"- Aandeel PSG (A): {poss['share_A']:.0%}, Inter (B): {poss['share_B']:.0%}; wissels: {poss['changes']}",
        "", "## Afstand en snelheid per team", "", md_table(teams_view),
        "", f"Per track: zie `analytics/players_{model}.csv`.",
        "", extra_markdown(extra, {k: f"analytics/{v}" for k, v in figures.items()}),
        "", "## Heatmaps en posities",
        "De pijl op de teamheatmaps toont de aanvalsrichting.", "",
        f"![heatmap PSG](analytics/heatmap_team_A_{model}.png)", "",
        f"![heatmap Inter](analytics/heatmap_team_B_{model}.png)", "",
        f"![heatmap bal](analytics/heatmap_ball_{model}.png)", "",
        f"![gemiddelde posities](analytics/average_positions_{model}.png)", "",
    ]
    if video_name:
        lines += ["## Video", f"Minimapvideo: [{video_name}]({video_name})", ""]
    lines += ["## Systeemevaluatie (tegen eigen ground truth)", ""] + eval_lines
    lines += [
        "## Beperkingen",
        "- Eén korte clip (±22 s) van één wedstrijd: geen algemene uitspraken; afstanden en zones gaan over seconden.",
        "- Homografie geldt alleen voor punten op het veld; positieruis wordt afgevlakt maar blijft een foutbron.",
        "- ID-switches splitsen spelers over meerdere tracks en korte tracks vallen weg: kilometers zijn een ondergrens.",
        "- Waarden per 90 minuten zijn een ruwe, indicatieve extrapolatie van 22 s.",
        "- Aanvalsrichting = gewogen stem van keeperpositie, gemiddelde positie veldspelers en balrichting; zie de betrouwbaarheid.",
        "- Schoten worden niet geteld (te onbetrouwbaar in trackingdata); aanvallen en ingangen laatste derde/16 m-gebied "
        "zijn eenvoudige regels op de balpositie.",
        "- De evaluatie gebruikt maar enkele gelabelde frames, dus de cijfers zijn grof.",
        "",
    ]
    return "\n".join(lines)
