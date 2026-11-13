"""
IoU en NMS Experimenten Benchmark voor YOLO26s.
Auteur: Dmitrij Tchervonisjenko (Team Git Push and Pray)
Thomas More Geel - Deep Learning & Computer Vision

Vergelijkt systematisch 18 verschillende Confidence (conf) en NMS IoU drempelwaarden
om de optimale F1-score, Precision-Recall balans en mAP@50/mAP@50-95 te bepalen.
"""

import sys
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Ensure src directory is on sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import DIMI_DIR, REPO_ROOT, BEST_WEIGHTS, DATA_YAML, BENCHMARK_CSV

WEIGHTS_PATH = BEST_WEIGHTS
OUTPUT_CSV = BENCHMARK_CSV

CONFIGS = [
    {"name": "Ultra-Low (0.005, 0.50)", "conf": 0.005, "iou": 0.50},
    {"name": "Ultra-Low WINNER mAP50 (0.005, 0.60)", "conf": 0.005, "iou": 0.60},
    {"name": "Ultra-Low WINNER mAP50-95 (0.005, 0.70)", "conf": 0.005, "iou": 0.70},
    {"name": "Ultra-High Recall (0.010, 0.50)", "conf": 0.010, "iou": 0.50},
    {"name": "Ultra-High Recall (0.010, 0.60)", "conf": 0.010, "iou": 0.60},
    {"name": "Ultra-High Recall (0.010, 0.70)", "conf": 0.010, "iou": 0.70},
    {"name": "Ball-Tuned (0.015, 0.50)", "conf": 0.015, "iou": 0.50},
    {"name": "Ball-Tuned (0.015, 0.60)", "conf": 0.015, "iou": 0.60},
    {"name": "Ball-Tuned (0.015, 0.65)", "conf": 0.015, "iou": 0.65},
    {"name": "Ball-Tuned (0.020, 0.55)", "conf": 0.020, "iou": 0.55},
    {"name": "Ball-Tuned (0.020, 0.60)", "conf": 0.020, "iou": 0.60},
    {"name": "Ball-Optimized (0.025, 0.60)", "conf": 0.025, "iou": 0.60},
    {"name": "Ball-Optimized Permissive NMS (0.025, 0.70)", "conf": 0.025, "iou": 0.70},
    {"name": "Medium Confidence (0.050, 0.60)", "conf": 0.050, "iou": 0.60},
    {"name": "Conservative (0.100, 0.60)", "conf": 0.100, "iou": 0.60},
    {"name": "Standard YOLO Baseline (0.250, 0.60)", "conf": 0.250, "iou": 0.60},
    {"name": "High Precision Strict (0.250, 0.70)", "conf": 0.250, "iou": 0.70},
    {"name": "High Confidence Baseline (0.500, 0.60)", "conf": 0.500, "iou": 0.60},
]


def run_benchmark():
    print("=" * 80)
    print("START IoU & NMS BENCHMARK EXPERIMENTEN OP DE RTX 4070 GPU")
    print(f"Model: {WEIGHTS_PATH}")
    print(f"Aantal te testen configuraties: {len(CONFIGS)}")
    print("=" * 80)

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Model weights niet gevonden op: {WEIGHTS_PATH}")

    model = YOLO(str(WEIGHTS_PATH))
    results = []

    for i, cfg in enumerate(CONFIGS, 1):
        print(f"\n[{i:02d}/{len(CONFIGS)}] Testen: {cfg['name']} (conf={cfg['conf']}, iou={cfg['iou']})...")
        metrics = model.val(
            data=str(DATA_YAML),
            imgsz=1280,
            conf=cfg["conf"],
            iou=cfg["iou"],
            batch=16,
            workers=0,
            device="cuda",
            verbose=False,
            plots=False
        )

        p = float(metrics.box.mp)
        r = float(metrics.box.mr)
        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

        ball_map50 = float(metrics.box.all_ap[0, 0]) if hasattr(metrics.box, 'all_ap') else 0.0
        ball_map50_95 = float(metrics.box.maps[0]) if len(metrics.box.maps) > 0 else 0.0

        row = {
            "Config Name": cfg["name"],
            "Conf Threshold": cfg["conf"],
            "IoU Threshold": cfg["iou"],
            "Precision (P)": round(p, 4),
            "Recall (R)": round(r, 4),
            "F1-Score": round(f1, 4),
            "mAP@50": round(map50, 4),
            "mAP@50-95": round(map50_95, 4),
            "Ball mAP@50": round(ball_map50, 4),
            "Ball mAP@50-95": round(ball_map50_95, 4),
        }
        results.append(row)
        print(f"  -> P={p:.4f} | R={r:.4f} | F1={f1:.4f} | mAP@50={map50:.4f} | mAP@50-95={map50_95:.4f} | Ball mAP@50={ball_map50:.4f}")

    # Ruim eventuele tijdelijke runs op
    root_runs = Path("runs")
    if root_runs.exists():
        shutil.rmtree(root_runs, ignore_errors=True)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "=" * 80)
    print(f"BENCHMARK VOLTOOID! Opgeslagen in: {OUTPUT_CSV}")
    print("=" * 80)
    print(df.to_string(index=False))

    # Bepaal winnaar
    winner_map50 = df.loc[df["mAP@50"].idxmax()]
    winner_f1 = df.loc[df["F1-Score"].idxmax()]
    winner_map = df.loc[df["mAP@50-95"].idxmax()]

    print("\nOVERZICHT TOP CONFIGURATIES:")
    print(f"1. Hoogste mAP@50:    {winner_map50['mAP@50']} ({winner_map50['Config Name']} | conf={winner_map50['Conf Threshold']}, iou={winner_map50['IoU Threshold']})")
    print(f"2. Hoogste mAP@50-95: {winner_map['mAP@50-95']} ({winner_map['Config Name']} | conf={winner_map['Conf Threshold']}, iou={winner_map['IoU Threshold']})")
    print(f"3. Hoogste F1-Score:  {winner_f1['F1-Score']} ({winner_f1['Config Name']})")


if __name__ == "__main__":
    run_benchmark()
