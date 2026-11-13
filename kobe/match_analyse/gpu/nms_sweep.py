"""Job 4: NMS-IoU sweep + the NMS-free head of YOLO26, and the effect on the whole pipeline (lesson 5.4).

A. val() of best.pt on the test set for iou in IOUS (classic NMS on the one2many head, ultralytics default)
   plus nms=False (the end-to-end one2one head; iou has no effect there). Per setting: mAP50, mAP50-95,
   AP50 and best F1 per class. -> nms/sweep_val.csv + sweep_val.png
B. For the default (iou 0.7, NMS) and the best 2 other settings (score = (mAP50 + ball AP50) / 2): YOLO over the
   clip with that setting (core.detection.detect(..., iou, nms)), core.pipeline.run and the system evaluation
   (evaluation.metrics) against data/ground_truth.json (anchor body) and data/ground_truth_feet.json (feet).
   -> pipeline_effect.csv (results folder; the fine-tuned model is added by run_finetuned), and every detection
   cache in cache_export/ (portable key, see core/detection.cache_path): copied into cache/ on the owner's laptop
   the pipeline reuses it and YOLO does not run again.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpu.common import GT_BODY, GT_FEET, Ctx, JobError, best_threshold, log, resolved_yaml, write_csv  # noqa: E402
from gpu.pr_curves import extract, val_model  # noqa: E402

IOUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
QUICK_IOUS = [0.5, 0.7]
DEFAULT = (0.7, None)
QUICK_STRIDE = 30            # quick: every 30th frame = the 22 ground-truth frames


def label(iou, nms):
    return "nms=False (NMS-vrij)" if nms is False else f"NMS iou={iou}"


def sweep_val(ctx: Ctx, weights, out):
    test_yaml = resolved_yaml(ctx.test_data, out / "data_test.yaml", splits=(ctx.test_split,))
    settings = [(i, None) for i in (QUICK_IOUS if ctx.quick else IOUS)] + [(0.7, False)]
    rows = []
    for iou, nms in settings:
        log(f"val sweep: {label(iou, nms)} ...")
        m = val_model(weights, test_yaml, ctx.test_split, ctx.imgsz, ctx.device, out / "val_runs",
                      f"iou{iou}_nms{nms}", iou=iou, nms=nms, plots=False)
        per_class, curves = extract(m)
        row = {"setting": label(iou, nms), "iou": None if nms is False else iou, "nms": "False" if nms is False else "ja"}
        for r in per_class:
            if r["class"] == "all":
                row["mAP50"], row["mAP50-95"] = r["mAP50"], r["mAP50-95"]
            else:
                row[f"AP50_{r['class']}"] = r["mAP50"]
                c = curves[r["class"]]
                row[f"bestF1_{r['class']}"] = best_threshold(c["px"], c["f1"])["f1"]
        row["_iou"], row["_nms"] = iou, nms
        rows.append(row)
    cols = ["setting", "iou", "nms", "mAP50", "mAP50-95"] + sorted({k for r in rows for k in r
                                                                    if k.startswith(("AP50_", "bestF1_"))})
    write_csv(out / "sweep_val.csv", rows, cols)
    plot_sweep(rows, out / "sweep_val.png")
    return rows, cols


def score(row):
    return ((row.get("mAP50") or 0) + (row.get("AP50_ball") or 0)) / 2


def pick_settings(rows, n=2):
    """The default plus the n best other settings (by score)."""
    others = sorted((r for r in rows if (r["_iou"], r["_nms"]) != DEFAULT), key=score, reverse=True)[:n]
    return [DEFAULT] + [(r["_iou"], r["_nms"]) for r in others]


PIPELINE_COLS = ["model", "setting", "imgsz", "frames", "seconds", "ball_detected", "ball_interpolated", "ball_none",
                 "body_ball_found", "body_false_ball", "body_players_f1", "body_role_acc", "body_possession_acc",
                 "feet_ball_found", "feet_false_ball", "feet_players_f1", "feet_role_acc", "feet_possession_acc",
                 "possession_A_pct", "possession_B_pct", "cache_file"]


def sweep_pipeline(ctx: Ctx, weights, settings, out, model_label="best.pt", imgsz=None):
    """Whole clip through core.pipeline.run for every (iou, nms) setting, evaluated against both GT files.
    Every detection cache is copied to <results>/cache_export/ (portable file name) for the owner's cache/."""
    import shutil

    from core.detection import cache_path, load_model
    from core.pipeline import possession_share, run, save_frames_csv, save_tracks_csv
    from core.video import video_info
    from evaluation.gt_format import load_gt
    from evaluation.metrics import evaluate, load_predictions
    from gpu.common import CACHE_EXPORT

    imgsz = imgsz or ctx.imgsz
    info = video_info(ctx.video)
    if info is None:
        raise JobError(f"video niet te openen: {ctx.video}")
    stride = QUICK_STRIDE if ctx.quick else 1
    end = info["frames"] - 1
    gts = {"body": load_gt(GT_BODY), "feet": load_gt(GT_FEET)}
    export = ctx.sub(CACHE_EXPORT)
    rows = []
    for iou, nms in settings:
        name = label(iou, nms)
        log(f"pijplijn {model_label}: {name}, imgsz {imgsz}, frames 0-{end} stride {stride} ...")
        t0 = time.time()
        model = load_model(weights)            # fresh model per setting: the predictor keeps its head mode
        model.overrides["device"] = ctx.device

        def progress(done, total):
            if done == total or done % 100 == 0:
                log(f"  gedetecteerd {done}/{total}")
        result = run(ctx.video, weights, imgsz=imgsz, iou=iou, start=0, end=end, stride=stride, model=model,
                     progress=progress, nms=nms)
        cache = cache_path(ctx.video, weights, imgsz, iou, 0, end, stride, nms=nms)
        exported = cache_path(ctx.video, weights, imgsz, iou, 0, end, stride, nms=nms, portable=True).name
        if cache.exists():           # portable name: the owner copies it into cache/ and YOLO is skipped there
            shutil.copy2(cache, export / exported)
        d = out / f"pipeline_{slug_model(model_label)}_{imgsz}_iou{iou}_nms{nms}"
        d.mkdir(exist_ok=True)
        tracks, frames = save_tracks_csv(result, d / "tracks.csv"), save_frames_csv(result, d / "frames.csv")
        people, ball, poss = load_predictions(tracks, frames)
        n = len(result.ids)
        states = [result.ball[i]["state"] if i in result.ball else "none" for i in result.ids]
        share = possession_share(result)
        row = {"model": model_label, "setting": name, "imgsz": imgsz, "_iou": iou, "_nms": nms, "frames": n,
               "seconds": round(time.time() - t0, 1),
               "ball_detected": states.count("detected"), "ball_interpolated": states.count("interpolated"),
               "ball_none": states.count("none"),
               "possession_A_pct": share.get("A"), "possession_B_pct": share.get("B"),
               "cache_file": exported if cache.exists() else None}
        for anchor, gt in gts.items():
            if not gt:
                continue
            res = evaluate(gt, people, ball, poss, anchor=anchor)
            b, t, p, pl = res["ball"], res["teams"], res["possession"], res["players"]
            found = round((b["detection_rate"] or 0) * b["frames_with_ball"])
            false = round((b["false_ball_rate"] or 0) * b["frames_without_ball"])
            row.update({f"{anchor}_ball_found": f"{found}/{b['frames_with_ball']}",
                        f"{anchor}_false_ball": f"{false}/{b['frames_without_ball']}",
                        f"_{anchor}_found": found, f"_{anchor}_false": false,
                        f"{anchor}_players_f1": pl["f1"], f"{anchor}_role_acc": t["role_accuracy"],
                        f"{anchor}_possession_acc": p["accuracy"]})
        rows.append(row)
    return rows


def slug_model(model_label):
    return model_label.replace(" ", "_").replace(".", "").replace("@", "_")


def write_pipeline_effect(ctx: Ctx):
    """pipeline_effect.csv in the results folder: all pipeline rows of this run (best.pt sweep + fine-tune)."""
    rows = list(ctx.info.get("nms", {}).get("pipeline", [])) + list(ctx.info.get("pipeline_ft", []))
    return write_csv(ctx.out / "pipeline_effect.csv", rows, PIPELINE_COLS)


def plot_sweep(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nms_rows = [r for r in rows if r["_nms"] is None]
    free = next((r for r in rows if r["_nms"] is False), None)
    xs = [r["_iou"] for r in nms_rows]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for key, name in [("mAP50", "mAP50 (alle klassen)"), ("mAP50-95", "mAP50-95"), ("AP50_ball", "AP50 bal"),
                      ("bestF1_ball", "beste F1 bal")]:
        if not any(r.get(key) is not None for r in rows):
            continue
        line, = ax.plot(xs, [r.get(key) for r in nms_rows], "o-", label=name)
        if free and free.get(key) is not None:
            ax.axhline(free[key], color=line.get_color(), ls="--", lw=1)
    ax.set(xlabel="NMS IoU-drempel", ylabel="score", ylim=(0, 1.02),
           title="NMS-IoU-sweep (best.pt, testset); stippellijn = nms=False (NMS-vrije kop)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_job(ctx: Ctx, weights=None):
    weights = Path(weights or ctx.best)
    out = ctx.sub("nms")
    val_rows, val_cols = sweep_val(ctx, weights, out)
    settings = pick_settings(val_rows)
    log("pipeline-instellingen: " + ", ".join(label(*s) for s in settings))
    pipe_rows = sweep_pipeline(ctx, weights, settings, out, "best.pt")
    res = {"val": [{k: v for k, v in r.items() if not k.startswith("_")} for r in val_rows], "val_cols": val_cols,
           "pipeline": pipe_rows, "pipeline_cols": PIPELINE_COLS[1:-1], "weights": str(weights)}
    ctx.info["nms"] = res
    ctx.save_json("nms.json", res)
    write_pipeline_effect(ctx)
    return res


def run_finetuned(ctx: Ctx):
    """Default setting (NMS iou 0.7) of the fine-tuned model over the whole clip, at the pipeline imgsz and (when
    different) at the fine-tune imgsz, plus best.pt at that imgsz; added to pipeline_effect.csv."""
    if not Path(ctx.ft).exists():
        raise JobError("geen fine-tuned model")
    out = ctx.sub("nms")
    rows = sweep_pipeline(ctx, ctx.ft, [DEFAULT], out, "ball-finetune")
    ft_imgsz = ctx.info.get("finetune", {}).get("imgsz")
    if ft_imgsz and ft_imgsz != ctx.imgsz and not ctx.quick:
        rows += sweep_pipeline(ctx, ctx.ft, [DEFAULT], out, f"ball-finetune@{ft_imgsz}", imgsz=ft_imgsz)
        rows += sweep_pipeline(ctx, ctx.best, [DEFAULT], out, f"best.pt@{ft_imgsz}", imgsz=ft_imgsz)
    ctx.info["pipeline_ft"] = rows
    ctx.save_json("pipeline_ft.json", rows)
    write_pipeline_effect(ctx)
    return rows


def main():
    from gpu.common import common_args, ctx_from_args
    ctx = ctx_from_args(common_args(argparse.ArgumentParser(description=__doc__)).parse_args())
    try:
        run_job(ctx)
    except JobError as e:
        sys.exit(str(e))
    print(f"-> {ctx.out / 'nms'}")


if __name__ == "__main__":
    main()
