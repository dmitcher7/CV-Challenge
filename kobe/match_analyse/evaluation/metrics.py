"""Evaluation metrics: ground truth (data/ground_truth.json) versus the pipeline CSVs.

Per labelled frame we take the nearest processed frame (the pipeline may run with a stride); a
labelled frame further away than max_gap frames is skipped. Persons are matched with the Hungarian
algorithm on the metre distance; a pair further apart than `radius` is not a match.

Two anchors for the person clicks (`anchor`):
- "feet" (default): the GT click is on the feet, so it is compared in METRES with the pipeline position
  (feet = bottom-centre of the box, projected with the homography). This gives the real minimap error.
- "body": the GT click is somewhere on the body (e.g. the torso). A click above the feet is not on the
  pitch plane, so its metre position is wrong by metres. Then persons are matched in IMAGE PIXELS:
  distance from the click to the box centre, divided by the box height (so near and far players count
  alike); a pair with a normalised distance above BODY_RADIUS is not a match. Detection, team and
  tracking metrics stay valid; a metre position error is not reported.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROLE_OF_GROUP = {"player_A": "A", "player_B": "B", "player_B_goal_keeper": "K", "referee": "R"}
ROLE_ORDER = ["A", "B", "K", "R", "other"]
DEFAULT_RADIUS = 3.0       # m, anchor "feet"
BODY_RADIUS = 0.6          # box heights, anchor "body": click within ~0.6 box height of the box centre
FEET_CLICK_MIN = 0.75      # median click height in the box (0 = top, 1 = bottom) below this -> not feet clicks


def load_predictions(tracks_csv, frames_csv):
    """-> (people, ball, possession), each keyed by frame idx.

    people[f]     list of {"track_id", "role", "x", "y", "box"}   (box = (x1, y1, x2, y2) px, or None
                  for an older CSV without box columns)
    ball[f]       (x, y) or None
    possession[f] "A" / "B" / "-"
    """
    people, ball, possession = defaultdict(list), {}, {}
    with open(tracks_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            people[int(r["frame"])].append({"track_id": int(r["track_id"]),
                                            "role": ROLE_OF_GROUP.get(r["group"], "other"),
                                            "x": float(r["x_m"]), "y": float(r["y_m"]),
                                            "box": _box(r)})
    with open(frames_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            idx = int(r["frame"])
            people.setdefault(idx, [])
            has_ball = r["ball_state"] != "none" and r["ball_x_m"] != ""
            ball[idx] = (float(r["ball_x_m"]), float(r["ball_y_m"])) if has_ball else None
            possession[idx] = r["possession_team"] or "-"
    return dict(people), ball, possession


def _box(row):
    try:
        return tuple(float(row[k]) for k in ("box_x1", "box_y1", "box_x2", "box_y2"))
    except (KeyError, TypeError, ValueError):
        return None


def nearest_frame(processed, idx):
    """Nearest processed frame index (sorted list) to idx."""
    return min(processed, key=lambda p: abs(p - idx))


def match_persons(gt_persons, preds, radius=DEFAULT_RADIUS, anchor="feet"):
    """Hungarian matching. Returns [(gt index, pred index, distance)] with distance <= radius.

    anchor "feet": distance in metres. anchor "body": pixel distance click -> box centre / box height.
    """
    if not gt_persons or not preds:
        return []
    if anchor == "body":
        g = np.array([p["px"] for p in gt_persons], float)
        b = np.array([p["box"] for p in preds], float)
        centre = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)
        height = np.maximum(b[:, 3] - b[:, 1], 1.0)
        dist = np.linalg.norm(g[:, None, :] - centre[None, :, :], axis=2) / height[None, :]
    else:
        g = np.array([p["m"] for p in gt_persons], float)
        q = np.array([[p["x"], p["y"]] for p in preds], float)
        dist = np.linalg.norm(g[:, None, :] - q[None, :, :], axis=2)
    cost = np.where(dist <= radius, dist, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c), float(dist[r, c])) for r, c in zip(rows, cols) if dist[r, c] <= radius]


def _stats(values):
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "p90": None}
    v = np.asarray(values, float)
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90))}


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def click_height(gt, people, max_gap=0):
    """Median relative height of the GT person clicks inside the predicted box they fall in
    (0 = top of the box, 1 = bottom = feet), or None when there are no boxes. Used to warn when the
    persons were not clicked on the feet."""
    processed = sorted(people)
    rel = []
    for i, rec in gt.items():
        if not processed:
            break
        p = nearest_frame(processed, i)
        if abs(p - i) > max_gap:
            continue
        boxes = [q["box"] for q in people[p] if q.get("box")]
        for g in rec.get("persons", []):
            x, y = g["px"]
            inside = [b for b in boxes if b[0] - 3 <= x <= b[2] + 3 and b[1] - 15 <= y <= b[3] + 15]
            if inside:
                b = min(inside, key=lambda b: b[3] - b[1])
                rel.append((y - b[1]) / max(b[3] - b[1], 1.0))
    return float(np.median(rel)) if rel else None


def tracking_metrics(pairs_per_frame, n_gt, n_pred):
    """ID switches + IDF1-lite.

    pairs_per_frame: [(frame, [(gt_id, track_id), ...])] for the labelled frames, matched per frame.
    n_gt / n_pred: total GT persons / predicted persons on those frames.
    IDF1-lite: gt ids and track ids are matched one-to-one for the whole clip so that most frame-level
    matches agree (Hungarian on co-occurrence counts); IDTP = those agreements.
    """
    seq = defaultdict(list)
    co = Counter()
    for frame, pairs in sorted(pairs_per_frame):
        for gid, tid in pairs:
            seq[gid].append(tid)
            co[(gid, tid)] += 1
    switches = sum(a != b for s in seq.values() for a, b in zip(s[:-1], s[1:]))
    gids, tids = sorted({g for g, _ in co}), sorted({t for _, t in co})
    idtp = 0
    if gids and tids:
        m = np.zeros((len(gids), len(tids)))
        for (g, t), c in co.items():
            m[gids.index(g), tids.index(t)] = c
        r, c = linear_sum_assignment(-m)
        idtp = int(m[r, c].sum())
    idfn, idfp = n_gt - idtp, n_pred - idtp
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if 2 * idtp + idfp + idfn else None
    return {"id_switches": int(switches), "n_gt_ids": len(gids), "idtp": idtp, "idfp": int(idfp),
            "idfn": int(idfn), "idf1_lite": idf1}


def evaluate(gt, people, ball, possession, radius=None, max_gap=None, anchor="feet"):
    """Compare ground truth {frame: record} with the predictions of load_predictions(). Returns a dict.

    radius: metres for anchor "feet" (default DEFAULT_RADIUS), box heights for "body" (default BODY_RADIUS).
    """
    if anchor not in ("feet", "body"):
        raise ValueError(f"anchor must be 'feet' or 'body', not {anchor!r}")
    if radius is None:
        radius = BODY_RADIUS if anchor == "body" else DEFAULT_RADIUS
    if anchor == "body" and any(q.get("box") is None for ps in people.values() for q in ps):
        raise ValueError("anchor 'body' needs the box columns in tracks_<model>.csv: run run_minimap.py again")
    processed = sorted(set(people) | set(ball))
    if not processed:
        raise ValueError("the pipeline output has no frames")
    if max_gap is None:   # the stride of the pipeline run
        max_gap = int(np.median(np.diff(processed))) if len(processed) > 1 else 0
    labelled = sorted(i for i, rec in gt.items() if rec.get("done", True))

    used, skipped = [], []         # (gt frame, pipeline frame)
    for i in labelled:
        p = nearest_frame(processed, i)
        (used if abs(p - i) <= max_gap else skipped).append((i, p))

    errors, tp, fp, fn = [], 0, 0, 0
    confusion = defaultdict(Counter)
    ball_err, ball_gt, ball_found, ball_none, ball_false = [], 0, 0, 0, 0
    poss_conf = defaultdict(Counter)
    id_frames, n_gt_ids, n_pred_ids, ids_present = [], 0, 0, True
    n_gt_persons = 0

    for i, p in used:
        rec = gt[i]
        preds = people.get(p, [])
        gps = rec.get("persons", [])
        pairs = match_persons(gps, preds, radius, anchor)
        tp += len(pairs)
        fn += len(gps) - len(pairs)
        fp += len(preds) - len(pairs)
        n_gt_persons += len(gps)
        n_pred_ids += len(preds)
        for a, b, d in pairs:
            errors.append(d)
            confusion[gps[a]["role"]][preds[b]["role"]] += 1
        if any(g.get("id") is None for g in gps):
            ids_present = False
        id_frames.append((i, [(gps[a]["id"], preds[b]["track_id"]) for a, b, _ in pairs]))

        if rec.get("ball_visible") is not None:
            gb, pb = rec.get("ball"), ball.get(p)
            if rec["ball_visible"] and gb:
                ball_gt += 1
                if pb:
                    ball_found += 1
                    ball_err.append(float(np.hypot(pb[0] - gb["m"][0], pb[1] - gb["m"][1])))
            else:
                ball_none += 1
                ball_false += bool(pb)
        if rec.get("possession") is not None:
            poss_conf[rec["possession"]][possession.get(p, "-")] += 1

    n_matched = sum(sum(c.values()) for c in confusion.values())
    correct = sum(confusion[r][r] for r in confusion)
    merge = {"A": "A", "B": "B", "K": "B", "R": "-", "other": "-"}   # keeper belongs to team B
    correct_team = sum(c for g, row in confusion.items() for pr, c in row.items() if merge[g] == merge[pr])
    poss_n = sum(sum(c.values()) for c in poss_conf.values())
    poss_ok = sum(poss_conf[k][k] for k in poss_conf)

    has_boxes = any(q.get("box") for ps in people.values() for q in ps)
    ch = click_height({i: gt[i] for i, _ in used}, people, max_gap) if has_boxes else None
    result = {
        "anchor": anchor,
        "click_height_median": ch,
        "model_frames": {"labelled": len(labelled), "matched": len(used), "skipped_no_processed_frame": len(skipped),
                         "max_gap": max_gap, "exact": sum(i == p for i, p in used),
                         "max_offset": max((abs(i - p) for i, p in used), default=0)},
        "radius_m": radius,
        "players": {"position_error_m": _stats(errors) if anchor == "feet" else _stats([]),
                    "match_distance_box_heights": _stats(errors) if anchor == "body" else _stats([]),
                    **_prf(tp, fp, fn)},
        "teams": {"n": n_matched, "role_accuracy": correct / n_matched if n_matched else None,
                  "team_accuracy_merged": correct_team / n_matched if n_matched else None,
                  "confusion": {g: {pr: int(confusion[g][pr]) for pr in ROLE_ORDER if confusion[g][pr]}
                                for g in ["A", "B", "K", "R"] if g in confusion}},
        "ball": {"frames_with_ball": ball_gt, "detection_rate": ball_found / ball_gt if ball_gt else None,
                 "position_error_m": _stats(ball_err),
                 "within_3m": float(np.mean(np.array(ball_err) <= 3)) if ball_err else None,
                 "frames_without_ball": ball_none,
                 "false_ball_rate": ball_false / ball_none if ball_none else None},
        "possession": {"n": poss_n, "accuracy": poss_ok / poss_n if poss_n else None,
                       "confusion": {g: dict(poss_conf[g]) for g in ("A", "B", "-") if g in poss_conf}},
    }
    if ids_present and n_gt_persons:
        result["tracking"] = tracking_metrics(id_frames, n_gt_persons, n_pred_ids)
    else:
        result["tracking"] = {"skipped": "ground truth has no person ids (label with the tool to carry ids)"}
    return result


def format_report(res, model):
    """Markdown report of evaluate()."""
    def f(v, nd=3):
        return "n/a" if v is None else f"{v:.{nd}f}"

    mf, pl, te, ba, po, tr = (res[k] for k in ("model_frames", "players", "teams", "ball", "possession", "tracking"))
    pe = pl["position_error_m"]
    L = [f"# System evaluation: model `{model}`", "",
         f"Labelled frames: {mf['labelled']}, compared: {mf['matched']} "
         f"({mf['exact']} exactly on a processed frame, max offset {mf['max_offset']} frames, "
         f"{mf['skipped_no_processed_frame']} skipped: no processed frame within {mf['max_gap']} frames).", "",
    ]
    anchor, ch = res.get("anchor", "feet"), res.get("click_height_median")
    if ch is not None:
        L.append(f"Person clicks: median height in the box {ch:.2f} (0 = head, 1 = feet).")
    if anchor == "feet" and ch is not None and ch < FEET_CLICK_MIN:
        L += ["", "> **WARNING:** the persons were not clicked on the feet. A click above the feet is not on the",
              "> pitch plane, so its metre position is off by metres and the player/team numbers below are",
              "> unreliable. Run again with `--anchor body`, or relabel the feet.", ""]
    else:
        L.append("")
    if anchor == "body":
        md = pl["match_distance_box_heights"]
        L += [f"## Players (anchor body: click vs box centre in pixels, match radius {res['radius_m']} box heights, "
              "Hungarian)", "",
              "- no position error in metres: body clicks are not on the pitch plane (label the feet for that)",
              f"- match distance (box heights): mean {f(md['mean'])}, median {f(md['median'])} (n={md['n']})"]
    else:
        L += [f"## Players (anchor feet: match radius {res['radius_m']} m, Hungarian)", "",
              f"- position error (m): mean {f(pe['mean'])}, median {f(pe['median'])}, p90 {f(pe['p90'])} (n={pe['n']})"]
    L += [
         f"- TP {pl['tp']}, FP {pl['fp']}, FN {pl['fn']}: precision {f(pl['precision'])}, "
         f"recall {f(pl['recall'])}, F1 {f(pl['f1'])}", "",
         "## Teams / roles (matched players)", "",
         f"- role accuracy (A/B/keeper/referee): {f(te['role_accuracy'])}; "
         f"team accuracy (keeper counts as B, referee as none): {f(te['team_accuracy_merged'])} (n={te['n']})", "",
         "| GT \\ predicted | " + " | ".join(ROLE_ORDER) + " |", "|---|" + "---|" * len(ROLE_ORDER)]
    for g, row in te["confusion"].items():
        L.append(f"| {g} | " + " | ".join(str(row.get(r, 0)) for r in ROLE_ORDER) + " |")
    L += ["", "## Ball", "",
          f"- frames with a ball in GT: {ba['frames_with_ball']}; detection rate {f(ba['detection_rate'])}",
          f"- position error (m): mean {f(ba['position_error_m']['mean'])}, median {f(ba['position_error_m']['median'])}, "
          f"p90 {f(ba['position_error_m']['p90'])}; within 3 m: {f(ba['within_3m'])}",
          f"- frames without a ball in GT: {ba['frames_without_ball']}; false ball rate {f(ba['false_ball_rate'])}", "",
          "## Possession", "", f"- accuracy {f(po['accuracy'])} (n={po['n']})", "",
          "| GT \\ predicted | A | B | - |", "|---|---|---|---|"]
    for g, row in po["confusion"].items():
        L.append(f"| {g} | " + " | ".join(str(row.get(r, 0)) for r in ("A", "B", "-")) + " |")
    L += ["", "## Tracking", ""]
    if "skipped" in tr:
        L.append(f"- skipped: {tr['skipped']}")
    else:
        L += [f"- ID switches (change of matched track id of a GT person between consecutive labelled frames): "
              f"{tr['id_switches']} ({tr['n_gt_ids']} GT persons)",
              f"- IDF1-lite {f(tr['idf1_lite'])} (IDTP {tr['idtp']}, IDFP {tr['idfp']}, IDFN {tr['idfn']})",
              "- note: labelled frames are far apart, so this is a coarse estimate of the real ID switches."]
    return "\n".join(L) + "\n"
