"""
IoU en NMS Experimenten Benchmark.
Auteur: Dmitrij Tchervonisjenko (Team Git Push and Pray)
Thomas More Geel - Deep Learning & Computer Vision

Vergelijkt systematisch verschillende Confidence (conf) en NMS IoU drempelwaarden
om de optimale F1-score, Precision-Recall balans en mAP@50/mAP@50-95 te bepalen.
"""

import sys
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
    {"name": "Ultra-High Recall", "conf": 0.010, "iou": 0.50},
    {"name": "Ball-Optimized (Tight NMS)", "conf": 0.025, "iou": 0.50},
    {"name": "Ball-Optimized WINNER (Balanced NMS)", "conf": 0.025, "iou": 0.60},
    {"name": "Ball-Optimized (Permissive NMS)", "conf": 0.025, "iou": 0.70},
    {"name": "Medium Confidence", "conf": 0.050, "iou": 0.60},
    {"name": "Conservative", "conf": 0.100, "iou": 0.60},
    {"name": "YOLO Default Baseline", "conf": 0.250, "iou": 0.60},
    {"name": "High Precision / Strict", "conf": 0.250, "iou": 0.70},
]


def run_benchmark():
    print("=" * 70)
    print("🚀 STARTING IoU & NMS BENCHMARK EXPERIMENTEN (RTX 4070 GPU)")
    print("=" * 70)

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Model weights niet gevonden op: {WEIGHTS_PATH}")

    model = YOLO(str(WEIGHTS_PATH))
    results = []

    for cfg in CONFIGS:
        print(f"\n[RUN] Testen: {cfg['name']} (conf={cfg['conf']}, iou={cfg['iou']})...")
        metrics = model.val(
            data=str(DATA_YAML),
            imgsz=1280,
            conf=cfg["conf"],
            iou=cfg["iou"],
            device="cuda",
            verbose=False,
            plots=False
        )

        p = float(metrics.box.mp)
        r = float(metrics.box.mr)
        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

        # Bal klasse (index 0)
        class_maps = metrics.box.maps
        ball_map50 = float(class_maps[0]) if len(class_maps) > 0 else 0.0

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
        }
        results.append(row)
        print(f"  -> P={p:.3f} | R={r:.3f} | F1={f1:.3f} | mAP@50={map50:.3f} | Ball mAP@50={ball_map50:.3f}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "=" * 70)
    print(f"✅ BENCHMARK VOLTOOID! Opgeslagen in: {OUTPUT_CSV}")
    print("=" * 70)
    print(df.to_string(index=False))

    # Bepaal winnaar
    winner = df.loc[df["F1-Score"].idxmax()]
    print("\n🏆 WINNAAR OP BASIS VAN HARMONISCHE F1-SCORE & BAL RECALL:")
    print(f"Configuratie: {winner['Config Name']}")
    print(f"Conf={winner['Conf Threshold']} | IoU={winner['IoU Threshold']} | F1={winner['F1-Score']} | mAP50={winner['mAP@50']}")


if __name__ == "__main__":
    run_benchmark()
