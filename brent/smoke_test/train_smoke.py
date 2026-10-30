"""Smoke-test training: fine-tune a pretrained YOLO model on our split.

Defaults depend on the hardware:
    CPU -> imgsz=640,  epochs=10, batch=4
    GPU -> imgsz=1280, epochs=50, batch=8
Run make_split.py first (it creates data_smoke.yaml).
"""

import argparse

import ultralytics
from ultralytics import YOLO

from common import DATA_YAML, RUNS_DIR, RUN_NAME, SMOKE_DIR, pick_device, disable_label_cache


def default_model():
    """YOLO26 exists from ultralytics 8.4 on; otherwise fall back to YOLO11."""
    major, minor = (int(x) for x in ultralytics.__version__.split(".")[:2])
    return "yolo26s.pt" if (major, minor) >= (8, 4) else "yolo11s.pt"


device = pick_device()
on_gpu = device != "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=default_model())
parser.add_argument("--imgsz", type=int, default=1280 if on_gpu else 640)
parser.add_argument("--epochs", type=int, default=50 if on_gpu else 10)
parser.add_argument("--batch", type=int, default=8 if on_gpu else 4)
args = parser.parse_args()

print(f"device={device} model={args.model} imgsz={args.imgsz} epochs={args.epochs} batch={args.batch}")

disable_label_cache()  # do not write labels.cache into data/

# Pretrained weights are downloaded into brent/smoke_test/ (not the repo root)
model = YOLO(str(SMOKE_DIR / args.model))
model.train(
    data=str(DATA_YAML),
    imgsz=args.imgsz,
    epochs=args.epochs,
    batch=args.batch,
    device=device,
    project=str(RUNS_DIR),
    name=RUN_NAME,
    exist_ok=True,   # overwrite runs/smoke instead of creating smoke2, smoke3...
    seed=0,          # fixed seed for reproducibility
    plots=True,      # save results.png, confusion matrix, batch images
    workers=0,       # avoid multiprocessing issues on Windows
)
