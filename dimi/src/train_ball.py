"""
Train YOLO26 on the RTX 4070 GPU with imgsz=1280 to detect the ball.
Usage:
    python dimi/src/train_ball.py [--model yolo26s.pt] [--epochs 50] [--batch 8] [--imgsz 1280]
"""

import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

# Ensure src directory is on sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import DATA_YAML, RUNS_DIR, RUN_NAME, DIMI_DIR, pick_device, disable_label_cache

def main():
    parser = argparse.ArgumentParser(description="Train YOLO specifically tuned for ball detection.")
    parser.add_argument("--model", type=str, default="yolo26s.pt", help="Pretrained YOLO model (e.g. yolo26s.pt, yolo26m.pt)")
    parser.add_argument("--imgsz", type=int, default=1600, help="High resolution (1600 for sharp small object detection)")
    parser.add_argument("--epochs", type=int, default=45, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (4 fits comfortably at 1600p on 8GB GPU)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    args = parser.parse_args()

    device = pick_device()
    disable_label_cache()

    weights_cache = DIMI_DIR / args.model

    print("=" * 60)
    print(f"[START] Training {args.model} | imgsz={args.imgsz} | epochs={args.epochs} | batch={args.batch} | device={device}")
    print(f"[CONFIG] Data config: {DATA_YAML}")
    print(f"[CONFIG] Runs output: {RUNS_DIR / RUN_NAME}")
    print("=" * 60)

    model = YOLO(str(weights_cache if weights_cache.exists() else args.model))

    results = model.train(
        data=str(DATA_YAML),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=device,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        seed=0,
        plots=True,
        workers=2,
        close_mosaic=15,  # Turn off mosaic 15 epochs before end for clean small-ball boundaries
        box=9.0,          # Extra box loss weight for precise localization of small objects
        cls=1.0,          # Extra class loss weight
    )

    print("\n" + "=" * 60)
    print("[SUCCESS] Training completed!")
    best_pt = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    print(f"[CHECKPOINT] Best model weights: {best_pt}")
    print("=" * 60)

if __name__ == "__main__":
    main()
