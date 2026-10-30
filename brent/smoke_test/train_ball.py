"""Fine-tune a dedicated ball detector on complete and ball-only annotations."""

import argparse

from ultralytics import YOLO

from common import (
    BALL_DATA_YAML,
    BEST_WEIGHTS,
    RUNS_DIR,
    SMOKE_DIR,
    disable_label_cache,
    pick_device,
)


device = pick_device()
on_gpu = device != "cpu"
default_model = str(BEST_WEIGHTS) if BEST_WEIGHTS.is_file() else str(SMOKE_DIR / "yolo26m.pt")

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=default_model)
parser.add_argument("--imgsz", type=int, default=1600 if on_gpu else 640)
parser.add_argument("--epochs", type=int, default=100 if on_gpu else 10)
parser.add_argument("--batch", type=int, default=2 if on_gpu else 4)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--mosaic", type=float, default=0.0)
parser.add_argument("--scale", type=float, default=0.2)
args = parser.parse_args()

print(
    f"device={device} model={args.model} imgsz={args.imgsz} epochs={args.epochs} "
    f"batch={args.batch} patience={args.patience} mosaic={args.mosaic} scale={args.scale}"
)

disable_label_cache()

model = YOLO(args.model)
model.train(
    data=str(BALL_DATA_YAML),
    classes=[0],  # retain only the ball annotations; all players are intentional background here
    imgsz=args.imgsz,
    epochs=args.epochs,
    batch=args.batch,
    patience=args.patience,
    mosaic=args.mosaic,
    scale=args.scale,
    device=device,
    project=str(RUNS_DIR),
    name="ball",
    exist_ok=True,
    seed=0,
    plots=True,
    workers=0,
)
