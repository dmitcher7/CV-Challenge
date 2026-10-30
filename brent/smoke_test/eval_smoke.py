"""Evaluate best.pt on the val split and log the result.

Prints precision, recall, mAP@50 and mAP@50:95 per class and overall,
and appends one row to experiments.csv.
"""

import argparse
import csv
from pathlib import Path

import yaml
from ultralytics import YOLO

from common import (BEST_WEIGHTS, DATA_YAML, EXPERIMENTS_CSV, RUNS_DIR, RUN_NAME,
                    pick_device, disable_label_cache)

# Training settings of the run (imgsz, epochs, model) are stored in args.yaml
with open(BEST_WEIGHTS.parent.parent / "args.yaml", encoding="utf-8") as f:
    train_args = yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default=str(BEST_WEIGHTS))
parser.add_argument("--imgsz", type=int, default=train_args["imgsz"])  # same as training
args = parser.parse_args()

disable_label_cache()  # do not write labels.cache into data/

model = YOLO(args.weights)
metrics = model.val(
    data=str(DATA_YAML),
    split="val",
    imgsz=args.imgsz,
    batch=4,
    device=pick_device(),
    project=str(RUNS_DIR),
    name=f"{RUN_NAME}_val",
    exist_ok=True,
    plots=True,
)

box = metrics.box
names = model.names  # {class_id: name}, stored inside the weights

# ---- Per-class table ----
print(f"\n{'class':<22}{'P':>7}{'R':>7}{'mAP50':>8}{'mAP50-95':>10}")
for cid, name in names.items():
    if cid in box.ap_class_index:
        # class_result takes the position in ap_class_index, not the class id
        i = list(box.ap_class_index).index(cid)
        p, r, ap50, ap = box.class_result(i)
        print(f"{name:<22}{p:>7.3f}{r:>7.3f}{ap50:>8.3f}{ap:>10.3f}")
    else:
        print(f"{name:<22}{'(no instances in val)':>32}")
print(f"{'ALL':<22}{box.mp:>7.3f}{box.mr:>7.3f}{box.map50:>8.3f}{box.map:>10.3f}")

# ---- Append to experiments.csv ----
row = {
    "run": RUN_NAME,
    "model": Path(train_args["model"]).name,  # file name only, not the full path
    "imgsz": args.imgsz,
    "epochs": train_args["epochs"],
    "precision": round(box.mp, 4),
    "recall": round(box.mr, 4),
    "mAP50": round(box.map50, 4),
    "mAP50-95": round(box.map, 4),
}
new_file = not EXPERIMENTS_CSV.exists()
with open(EXPERIMENTS_CSV, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    if new_file:
        writer.writeheader()  # header only once
    writer.writerow(row)
print(f"\nAppended to {EXPERIMENTS_CSV}")
