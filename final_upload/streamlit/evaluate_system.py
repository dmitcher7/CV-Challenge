"""Evaluate the whole system against hand-labelled ground truth (course lesson 5.10, step 4).

    python evaluate_system.py --model psg_vs_inter_CV
    python evaluate_system.py --gt ..\\data\\ground_truth.example.json

Reads ../data/ground_truth.json (made with label_ground_truth.py) and output/tracks_<model>.csv +
output/frames_<model>.csv (made by the pipeline), and writes output/evaluation/system_eval_<model>_<anchor>.md and .json:
player position error / precision / recall / F1, team correctness, ball error and detection rate,
possession accuracy and (when the labels carry person ids) ID switches + IDF1-lite.
"""

import argparse
import json
import sys
from pathlib import Path

from core.paths import DATA_DIR, DEFAULT_WEIGHTS, OUTPUT_DIR
from evaluation.gt_format import load_gt
from evaluation.metrics import evaluate, format_report, load_predictions


def evaluate_files(model=DEFAULT_WEIGHTS.stem, gt_path=None, anchor="feet", radius=None, max_gap=None):
    """Evaluate output/tracks_<model>.csv + frames_<model>.csv against a ground-truth file and write the
    reports (also used by run_all.py). Returns (result dict, report text, path of the .md); exits via
    FileNotFoundError when an input is missing."""
    gt_path = Path(gt_path) if gt_path else DATA_DIR / "ground_truth.json"
    tracks, frames = OUTPUT_DIR / f"tracks_{model}.csv", OUTPUT_DIR / f"frames_{model}.csv"
    for p in (gt_path, tracks, frames):
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")
    gt = load_gt(gt_path)
    people, ball, possession = load_predictions(tracks, frames)
    res = evaluate(gt, people, ball, possession, radius=radius, max_gap=max_gap, anchor=anchor)

    out = OUTPUT_DIR / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"system_eval_{model}_{anchor}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    report = format_report(res, model)
    md = out / f"system_eval_{model}_{anchor}.md"
    md.write_text(report, encoding="utf-8")
    return res, report, md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_WEIGHTS.stem, help="name in output/tracks_<model>.csv")
    parser.add_argument("--gt", default=str(DATA_DIR / "ground_truth.json"))
    parser.add_argument("--anchor", choices=["feet", "body"], default="feet",
                        help="where the persons were clicked: feet (compare in metres) or body "
                             "(compare with the box centre in pixels; no metre error)")
    parser.add_argument("--radius", type=float, default=None,
                        help="match radius: metres for --anchor feet (default 3), box heights for body (default 0.6)")
    parser.add_argument("--max-gap", type=int, default=None,
                        help="max distance (frames) to the nearest processed frame; default = pipeline stride")
    args = parser.parse_args()
    try:
        _, report, md = evaluate_files(args.model, args.gt, args.anchor, args.radius, args.max_gap)
    except FileNotFoundError as e:
        sys.exit(str(e))
    print(report)
    print(f"-> {md}")


if __name__ == "__main__":
    main()
