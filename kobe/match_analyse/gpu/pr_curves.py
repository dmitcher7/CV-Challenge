"""Job 2: PR curves of the COCO baseline, our best.pt and the ball fine-tune (course lessons 3.8, 5.6, 5.10).

    python kobe\\match_analyse\\gpu\\pr_curves.py --test-data <data.yaml> --test-split test [--quick]

Two views of the same test images:
  * "classes": the 5 classes of our dataset (ball, player_A, player_B, keeper, referee) -> only our own models.
  * "merged":  ball + person (all players, keeper and referee together). COCO cannot tell teams apart, so
               this is the fair comparison with the COCO model ('sports ball' -> ball, 'person' -> person).
               The test labels are remapped to {0 ball, 1 person}; the predictions of every model (COCO and ours)
               are remapped the same way by RemapValidator after NMS, so all models go through identical code.
Per model and view: metrics_<run>.csv (P, R, AP50, AP50-95 per class), the ultralytics plots
(BoxPR_curve.png, BoxF1_curve.png, ...) in pr/<run>/ and the raw curves in curves_<run>.npz.
Combined: pr_ball_compare.png and pr_person_compare.png (all models overlaid, AP50 in the legend).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpu.common import (MERGED_NAMES, Ctx, JobError, build_merged_dataset, load_yaml, log, merged_map,  # noqa: E402
                        resolved_yaml, write_csv)

VAL_CONF = 0.001          # ultralytics default for val: the whole PR curve
VAL_BATCH = 4             # safe for 8 GB at imgsz 1280


def remap_validator(pred_map, names):
    """DetectionValidator subclass that maps predicted classes via pred_map (unmapped classes are dropped)
    and reports with the dataset's class names instead of the model's."""
    import torch
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils.metrics import ConfusionMatrix

    class RemapValidator(DetectionValidator):
        def init_metrics(self, model):
            super().init_metrics(model)
            self.names = dict(names)
            self.nc = len(names)
            self.metrics.names = self.names
            self.confusion_matrix = ConfusionMatrix(names=self.names,
                                                    save_matches=self.args.plots and self.args.visualize)

        def _prepare_pred(self, pred):
            pred = super()._prepare_pred(pred)
            cls = pred["cls"].long()
            lut = torch.full((max(max(pred_map), int(cls.max()) if len(cls) else 0) + 1,), -1,
                             dtype=torch.long, device=cls.device)
            for k, v in pred_map.items():
                lut[k] = v
            new = lut[cls] if len(cls) else cls
            keep = new >= 0
            out = {k: (v[keep] if isinstance(v, torch.Tensor) and len(v) == len(keep) else v) for k, v in pred.items()}
            out["cls"] = new[keep].to(pred["cls"].dtype)
            return out

    return RemapValidator


def val_model(weights, data, split, imgsz, device, project, name, pred_map=None, names=None, iou=0.7, nms=None,
              plots=True, batch=VAL_BATCH):
    """model.val() -> (metrics object). pred_map/names: remap predictions (see remap_validator)."""
    from ultralytics import YOLO
    model = YOLO(str(weights), task="detect")
    kwargs = dict(data=str(data), split=split, imgsz=imgsz, device=device, conf=VAL_CONF, iou=iou, batch=batch,
                  plots=plots, project=str(project), name=name, exist_ok=True, verbose=False, workers=0)
    if nms is not None:
        kwargs["nms"] = nms
    validator = remap_validator(pred_map, names) if pred_map is not None else None
    return model.val(validator=validator, **kwargs)


def extract(metrics):
    """(rows, curves) of a DetMetrics: rows = per class P/R/AP50/AP50-95 (+ 'all'); curves[class] = arrays.

    Verified in ultralytics 8.4.182 (utils/metrics.py, class Metric): box.p / r / ap50 / ap and the curves
    box.px (1000 points, x axis), box.prec_values (precision at recall = px), box.f1_curve / p_curve / r_curve
    (vs confidence = px); every per-class array is ordered like box.ap_class_index.
    """
    box, names = metrics.box, metrics.names
    nt = getattr(metrics, "nt_per_class", None)
    rows, curves = [], {}
    for i, c in enumerate(box.ap_class_index):
        name = names[int(c)]
        rows.append({"class": name, "instances": int(nt[int(c)]) if nt is not None else None,
                     "precision": float(box.p[i]), "recall": float(box.r[i]),
                     "mAP50": float(box.ap50[i]), "mAP50-95": float(box.ap[i])})
        curves[name] = {"px": np.asarray(box.px), "prec": np.asarray(box.prec_values[i]),
                        "f1": np.asarray(box.f1_curve[i]), "p": np.asarray(box.p_curve[i]),
                        "r": np.asarray(box.r_curve[i])}
    mp, mr, map50, mapx = metrics.mean_results()
    rows.append({"class": "all", "instances": int(nt.sum()) if nt is not None else None, "precision": float(mp),
                 "recall": float(mr), "mAP50": float(map50), "mAP50-95": float(mapx)})
    return rows, curves


def save_curves(path, curves):
    flat = {f"{cls}__{k}": v for cls, d in curves.items() for k, v in d.items()}
    np.savez_compressed(path, **flat)


def load_curves(path):
    data = np.load(path)
    out = {}
    for key in data.files:
        cls, k = key.split("__")
        out.setdefault(cls, {})[k] = data[key]
    return out


def model_list(ctx: Ctx):
    """[(label, weights, imgsz, is_coco)] of the models that exist."""
    models = [("COCO yolo26l", ctx.coco, ctx.imgsz, True), ("best.pt", ctx.best, ctx.imgsz, False)]
    if Path(ctx.ft).exists():
        models.append(("ball-finetune", ctx.ft, ctx.imgsz, False))
        ft_imgsz = ctx.info.get("finetune", {}).get("imgsz")
        if ft_imgsz and ft_imgsz != ctx.imgsz and not ctx.quick:
            models.append((f"ball-finetune@{ft_imgsz}", ctx.ft, ft_imgsz, False))
            models.append((f"best.pt@{ft_imgsz}", ctx.best, ft_imgsz, False))
    return models


def slug(label):
    return label.replace(" ", "_").replace(".", "").replace("@", "_")


def run(ctx: Ctx, only=None):
    """Evaluate all models (or only those whose label is in `only`) and redraw the combined plots."""
    out = ctx.sub("pr")
    test_yaml = resolved_yaml(ctx.test_data, out / "data_test.yaml", splits=(ctx.test_split,))
    test_names = load_yaml(test_yaml)["names"]
    merged_yaml, counts = build_merged_dataset(test_yaml, ctx.test_split, ctx.out / "data" / "merged_test")
    log(f"testset: {counts['images']} beelden, {counts['ball']} ballen, {counts['person']} personen")
    results = ctx.info.setdefault("pr", {"runs": {}, "test_counts": counts})

    for label, weights, imgsz, is_coco in model_list(ctx):
        if only and label not in only:
            continue
        model_names = _model_names(weights)
        views = [("merged", merged_yaml, "val", merged_map(model_names), MERGED_NAMES)]
        if not is_coco:
            if [model_names[k] for k in sorted(model_names)] != [test_names[k] for k in sorted(test_names)]:
                log(f"LET OP: klassen van {label} {model_names} verschillen van de testset {test_names}")
            views.insert(0, ("classes", test_yaml, ctx.test_split, None, None))
        for view, data, split, pred_map, names in views:
            run_name = f"{slug(label)}_{view}"
            log(f"val {label} ({view}, imgsz {imgsz}) ...")
            # quick: the 1-epoch fine-tune predicts degenerate boxes and the ultralytics plot thread prints errors
            plots = not (ctx.quick and "finetune" in label)
            m = val_model(weights, data, split, imgsz, ctx.device, out, run_name, pred_map=pred_map, names=names,
                          plots=plots)
            rows, curves = extract(m)
            for r in rows:
                r.update(model=label, view=view, imgsz=imgsz)
            write_csv(out / f"metrics_{run_name}.csv", rows,
                      ["model", "view", "imgsz", "class", "instances", "precision", "recall", "mAP50", "mAP50-95"])
            save_curves(out / f"curves_{run_name}.npz", curves)
            results["runs"][run_name] = {"model": label, "view": view, "imgsz": imgsz, "rows": rows,
                                         "curves": str(out / f"curves_{run_name}.npz"), "plots": str(out / run_name)}
    all_rows = [r for v in results["runs"].values() for r in v["rows"]]
    write_csv(out / "metrics_all.csv", all_rows,
              ["model", "view", "imgsz", "class", "instances", "precision", "recall", "mAP50", "mAP50-95"])
    for cls in ("ball", "person"):
        compare_plot(results["runs"], cls, out / f"pr_{cls}_compare.png")
    ctx.save_json("pr.json", results)
    return results


def _model_names(weights):
    import torch
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    model = ckpt.get("ema") or ckpt.get("model")
    names = model.names
    return dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}


def compare_plot(runs, cls, path):
    """PR curves of one class of the 'merged' view of all models in one figure, AP50 in the legend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5.5))
    drawn = 0
    for name, run in runs.items():
        if run["view"] != "merged":
            continue
        curves = load_curves(run["curves"])
        if cls not in curves:
            continue
        ap = next((r["mAP50"] for r in run["rows"] if r["class"] == cls), None)
        c = curves[cls]
        ax.plot(c["px"], c["prec"], lw=2, label=f"{run['model']} (imgsz {run['imgsz']}): AP50 = {ap:.3f}")
        drawn += 1
    title = {"ball": "Bal", "person": "Persoon (spelers + keeper + scheidsrechter samen)"}[cls]
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
           title=f"PR-curve {title}, IoU 0,5 (testset)")
    ax.grid(alpha=0.3)
    if drawn:
        ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    from gpu.common import common_args, ctx_from_args
    ctx = ctx_from_args(common_args(argparse.ArgumentParser(description=__doc__)).parse_args())
    try:
        run(ctx)
    except JobError as e:
        sys.exit(str(e))
    print(f"-> {ctx.out / 'pr'}")


if __name__ == "__main__":
    main()
