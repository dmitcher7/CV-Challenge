"""Job 3: confidence threshold per class from the F1-confidence curve (course lesson 5.4 / 3.8).

Uses the curves saved by pr_curves.py (run that first). For every model and class: the confidence with the highest
F1 on the test set, with precision and recall there. Writes thresholds/thresholds.csv and f1_<run>.png.
Note: the pipeline caches YOLO at conf 0.05 (core/detection.py DETECT_CONF), so a threshold below 0.05 acts as 0.05.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpu.common import Ctx, JobError, best_threshold, log, write_csv  # noqa: E402
from gpu.pr_curves import load_curves  # noqa: E402

PIPELINE_MIN_CONF = 0.05


def run(ctx: Ctx):
    pr = ctx.info.get("pr")
    if not pr or not pr.get("runs"):
        raise JobError("Geen PR-resultaten: draai eerst pr_curves.py (of run_all.py).")
    out = ctx.sub("thresholds")
    rows = []
    for name, run_ in pr["runs"].items():
        # own models: per real class ("classes") and ball/person ("merged"); COCO: only "merged"
        curves = load_curves(run_["curves"])
        for cls, c in curves.items():
            best = best_threshold(c["px"], c["f1"], c["p"], c["r"])
            rows.append({"model": run_["model"], "view": run_["view"], "imgsz": run_["imgsz"], "class": cls,
                         "best_conf": best["conf"], "f1": best["f1"], "precision": best["precision"],
                         "recall": best["recall"],
                         "below_pipeline_min": best["conf"] is not None and best["conf"] < PIPELINE_MIN_CONF})
        plot_f1(curves, f"{run_['model']} ({run_['view']}, imgsz {run_['imgsz']})", out / f"f1_{name}.png")
    write_csv(out / "thresholds.csv", rows)
    ctx.info["thresholds"] = rows
    ctx.save_json("thresholds.json", rows)
    log(f"drempels: {len(rows)} rijen -> {out / 'thresholds.csv'}")
    return rows


def plot_f1(curves, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for cls, c in curves.items():
        best = best_threshold(c["px"], c["f1"])
        if best["conf"] is None:
            ax.plot(c["px"], c["f1"], lw=1.8, label=f"{cls}: nooit gevonden (F1 = 0)")
            continue
        line, = ax.plot(c["px"], c["f1"], lw=1.8, label=f"{cls}: max F1 {best['f1']:.2f} bij conf {best['conf']:.2f}")
        ax.plot([best["conf"]], [best["f1"]], "o", color=line.get_color())
    ax.axvline(PIPELINE_MIN_CONF, color="grey", ls=":", lw=1)
    ax.set(xlabel="Confidence-drempel", ylabel="F1", xlim=(0, 1), ylim=(0, 1.02), title=f"F1 vs confidence: {title}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    import json
    from gpu.common import common_args, ctx_from_args
    parser = common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    if not args.out:
        sys.exit("geef --out <resultatenmap van een eerdere run> (met pr.json)")
    ctx = ctx_from_args(args)
    ctx.info["pr"] = json.loads((ctx.out / "pr.json").read_text(encoding="utf-8"))
    run(ctx)


if __name__ == "__main__":
    main()
