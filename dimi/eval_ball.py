"""
Evaluate Dimi's trained model on the validation split.
Displays detailed per-class metrics: Precision, Recall, mAP50, mAP50-95.
"""

import argparse
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

from common import DATA_YAML, RUNS_DIR, RUN_NAME, BEST_WEIGHTS, EXPERIMENTS_CSV, pick_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(BEST_WEIGHTS), help="Path to best.pt")
    parser.add_argument("--imgsz", type=int, default=1280, help="Validation resolution")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}. Run train_ball.py first!")
        return

    device = pick_device()

    print("=" * 60)
    print(f"[EVAL] Evaluating weights: {weights_path}")
    print(f"[EVAL] Device: {device} | imgsz: {args.imgsz}")
    print("=" * 60)

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=args.imgsz,
        device=device,
        project=str(RUNS_DIR),
        name=f"{RUN_NAME}_val",
        exist_ok=True,
        plots=True,
    )

    names = metrics.names
    print("\n" + "=" * 65)
    print(f"{'Klasse':<22}{'Precision':>10}{'Recall':>10}{'mAP@50':>10}{'mAP@50-95':>12}")
    print("-" * 65)

    # Class-level metrics
    p_box = metrics.box.p
    r_box = metrics.box.r
    map50_box = metrics.box.ap50
    map_box = metrics.box.ap

    for i, c_name in names.items():
        p = p_box[i] if i < len(p_box) else 0.0
        r = r_box[i] if i < len(r_box) else 0.0
        m50 = map50_box[i] if i < len(map50_box) else 0.0
        m = map_box[i] if i < len(map_box) else 0.0
        print(f"{c_name:<22}{p:>10.3f}{r:>10.3f}{m50:>10.3f}{m:>12.3f}")

    print("-" * 65)
    print(f"{'ALL (Mean)':<22}{metrics.box.mp:>10.3f}{metrics.box.mr:>10.3f}{metrics.box.map50:>10.3f}{metrics.box.map:>12.3f}")
    print("=" * 65)

    # Append to experiments.csv
    row = {
        "run": RUN_NAME,
        "weights": str(weights_path.name),
        "imgsz": args.imgsz,
        "precision": round(metrics.box.mp, 4),
        "recall": round(metrics.box.mr, 4),
        "mAP50": round(metrics.box.map50, 4),
        "mAP50-95": round(metrics.box.map, 4),
        "ball_mAP50": round(float(map50_box[0]), 4) if len(map50_box) > 0 else 0.0,
    }
    df = pd.DataFrame([row])
    if EXPERIMENTS_CSV.exists():
        df.to_csv(EXPERIMENTS_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(EXPERIMENTS_CSV, index=False)
    print(f"[SAVED] Results logged to {EXPERIMENTS_CSV}")

if __name__ == "__main__":
    main()
