"""One command for everything: minimap -> analytics (incl. extra statistics) -> evaluation -> report.

    venv\\Scripts\\python kobe\\match_analyse\\run_all.py --no-video [--weights models/best.pt] [--imgsz 1280]
                                                         [--stride 1] [--ball-refine]

1. run_minimap.make_minimap        detections come from cache/ when present (YOLO only runs for new settings)
2. run_analytics.analyze           tables, heatmaps, speed zones, kilometres, attacking direction
3. evaluate_system.evaluate_files  data/ground_truth.json (anchor body) and data/ground_truth_feet.json (anchor feet),
                                   each skipped when the file is missing
4. output/RAPPORT_<model>.md       Dutch report linking all outputs
The separate scripts keep working on their own.
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from analytics.rapport import rapport_markdown
from core.paths import DATA_DIR, DEFAULT_WEIGHTS, OUTPUT_DIR, VIDEO
from core.pipeline import DEFAULT_BALL_CONF, DEFAULT_IMGSZ
from evaluate_system import evaluate_files
from run_analytics import analyze
from run_minimap import make_minimap

# (label, ground-truth file, anchor): where the persons were clicked
EVALUATIONS = (
    ("Spelers op het lichaam geklikt (anchor body)", "ground_truth.json", "body"),
    ("Spelers op de voeten geklikt (anchor feet)", "ground_truth_feet.json", "feet"),
)


def step(n, total, text):
    print(f"\n=== [{n}/{total}] {text} ===", flush=True)


def run_all(args):
    """Run all steps for the argparse namespace of build_parser(). Returns the path of the report."""
    t0 = time.time()
    step(1, 4, "Minimap (detectie uit cache indien aanwezig)")
    model = make_minimap(args.video, args.weights, args.imgsz, args.start, args.end, args.stride, args.conf,
                         args.ball_conf, args.no_video, args.ball_refine)

    step(2, 4, "Analyses (afstand, snelheidszones, kilometers, aanvalsrichting, heatmaps)")
    tracks = pd.read_csv(OUTPUT_DIR / f"tracks_{model}.csv")
    frames = pd.read_csv(OUTPUT_DIR / f"frames_{model}.csv")
    res = analyze(model, tracks, frames, OUTPUT_DIR / "analytics", args.smooth_s, args.min_track_s)
    d = res["extra"]["direction"]
    print(f"{d['text']['A']}; {d['text']['B']} (betrouwbaarheid {d['confidence']}, {d['agreement']})")
    print(f"-> {res['out']}")

    step(3, 4, "Systeemevaluatie")
    evals = []
    for label, gt_name, anchor in EVALUATIONS:
        gt = DATA_DIR / gt_name
        if not gt.exists():
            print(f"overgeslagen: {gt} ontbreekt")
            evals.append((label, None, None, f"{gt_name} ontbreekt"))
            continue
        try:
            ev, _, md = evaluate_files(model, gt, anchor)
        except Exception as e:                      # one broken label file must not stop the report
            print(f"evaluatie {anchor} mislukt: {e}")
            evals.append((label, None, None, f"mislukt: {e}"))
            continue
        print(f"{anchor}: F1 spelers {ev['players']['f1']:.3f}, balbezit {ev['possession']['accuracy']:.2f} -> {md}")
        evals.append((label, ev, md.name, ""))

    step(4, 4, "Rapport schrijven")
    video = OUTPUT_DIR / f"minimap_{model}.mp4"
    params = f"gewichten {Path(args.weights).name}, imgsz {args.imgsz}, stride {args.stride}" \
             + (", bal-herdetectie" if args.ball_refine else "")
    text = rapport_markdown(model, res["extra"]["clip_s"], tracks["frame"].nunique(), res["table"], res["teams"],
                            res["poss"], res["extra"], evals, res["figures"],
                            video.name if video.exists() and not args.no_video else None, params)
    path = OUTPUT_DIR / f"RAPPORT_{model}.md"
    path.write_text(text, encoding="utf-8")
    print(f"-> {path}\nKlaar in {time.time() - t0:.0f} s.")
    return path


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default=str(VIDEO))
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                   help="1280 finds the small ball more often; 960 = training size, faster on CPU")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None, help="last frame (inclusive), default: end of the video")
    p.add_argument("--stride", type=int, default=1, help="process every n-th frame")
    p.add_argument("--conf", type=float, default=0.3, help="confidence to start a person track")
    p.add_argument("--ball-conf", type=float, default=DEFAULT_BALL_CONF, help="confidence threshold for ball candidates")
    p.add_argument("--no-video", action="store_true", help="skip the minimap video (csv files only)")
    p.add_argument("--ball-refine", action="store_true", help="re-detect the ball in crops (slow once, then cached)")
    p.add_argument("--smooth-s", type=float, default=0.5, help="smoothing window in seconds")
    p.add_argument("--min-track-s", type=float, default=1.0, help="drop tracks shorter than this")
    return p


if __name__ == "__main__":
    run_all(build_parser().parse_args())
