"""Evaluate a run on the val or test split and log the result.

Prints precision, recall, mAP@50 and mAP@50:95 per class and overall, the
confidence threshold with the best F1 per class (use it at inference time),
and appends one row to experiments.csv.

Use --split val while comparing runs; use --split test only once, for the
final model (val already chose best.pt, so it is no longer an unbiased score).
"""

import argparse
import csv
from datetime import date

import yaml
from ultralytics import YOLO

from config import DATA_YAML, EXPERIMENTS_CSV, RUNS_DIR, num_workers, pick_device, run_dir

FIELDS = ["date", "run", "split", "model", "imgsz", "epochs", "precision", "recall",
          "mAP50", "mAP50-95", "ball_mAP50", "ball_mAP50-95", "note"]


def evaluate(name, split="val", note=""):
    """Validate runs/<name>/weights/best.pt on one split, print and log the metrics."""
    folder = run_dir(name)
    # Training settings of the run (imgsz, model, overrides) are stored in args.yaml
    train_args = yaml.safe_load((folder / "args.yaml").read_text(encoding="utf-8"))
    epochs_done = len((folder / "results.csv").read_text().strip().splitlines()) - 1

    model = YOLO(str(folder / "weights" / "best.pt"))
    metrics = model.val(
        data=str(DATA_YAML),
        split=split,
        imgsz=train_args["imgsz"],  # same as training
        batch=4,
        device=pick_device(),
        project=str(RUNS_DIR),
        name=f"{name}_{split}",
        exist_ok=True,              # re-evaluating replaces the old evaluation plots
        plots=True,
        workers=num_workers(),
    )
    box = metrics.box
    present = list(box.ap_class_index)  # class ids that occur in this split

    # ---- Per-class table ----
    print(f"\n{'class':<14}{'P':>7}{'R':>7}{'mAP50':>8}{'mAP50-95':>10}{'best conf':>11}")
    per_class = {}
    for cid, cname in model.names.items():
        if cid not in present:
            print(f"{cname:<14}{'(no instances in ' + split + ')':>43}")
            continue
        # class_result and f1_curve take the position in ap_class_index, not the class id
        i = present.index(cid)
        p, r, ap50, ap = box.class_result(i)
        best_conf = box.px[box.f1_curve[i].argmax()]  # confidence with the highest F1
        per_class[cname] = (ap50, ap)
        print(f"{cname:<14}{p:>7.3f}{r:>7.3f}{ap50:>8.3f}{ap:>10.3f}{best_conf:>11.2f}")
    print(f"{'ALL':<14}{box.mp:>7.3f}{box.mr:>7.3f}{box.map50:>8.3f}{box.map:>10.3f}")

    # ---- Append to experiments.csv ----
    ball = per_class.get("ball", (float("nan"), float("nan")))
    row = {
        "date": date.today().isoformat(),
        "run": name,
        "split": split,
        "model": train_args["model"].replace("\\", "/").split("/")[-1],  # file name only
        "imgsz": train_args["imgsz"],
        "epochs": epochs_done,
        "precision": round(box.mp, 4),
        "recall": round(box.mr, 4),
        "mAP50": round(box.map50, 4),
        "mAP50-95": round(box.map, 4),
        "ball_mAP50": round(ball[0], 4),
        "ball_mAP50-95": round(ball[1], 4),
        "note": note,
    }
    new_file = not EXPERIMENTS_CSV.exists()
    with open(EXPERIMENTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()  # header only once
        writer.writerow(row)
    print(f"\nAppended to {EXPERIMENTS_CSV}")
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="folder name in runs/")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--note", default="", help="free text for experiments.csv")
    args = parser.parse_args()
    evaluate(args.run, args.split, args.note)
