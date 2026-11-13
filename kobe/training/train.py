"""Fine-tune a pretrained YOLO model on dataset/ (run prepare_dataset.py first).

Defaults depend on the hardware:
    GPU -> yolo26m.pt, imgsz=1280, up to 150 epochs (early stop after 30 without improvement)
    CPU -> yolo26s.pt, imgsz=640,  10 epochs (only to check that everything runs)

Every run gets its own folder runs/<name>/ and an existing run is never
overwritten. Any other ultralytics train argument can be set with --set, e.g.
    python train.py --name aug_scale --set scale=0.9 --set hsv_h=0.0
"""

import argparse
from datetime import datetime

import yaml
from ultralytics import YOLO

from config import DATA_YAML, PRETRAINED_DIR, RUNS_DIR, num_workers, pick_device


def train(name, model, imgsz, epochs, batch, patience, device, **overrides):
    """Train one run into runs/<name>/ and return its folder."""
    if (RUNS_DIR / name).exists():
        raise SystemExit(f"runs/{name} already exists: pick another --name")
    if not DATA_YAML.exists():
        raise SystemExit("dataset/data.yaml not found: run prepare_dataset.py first")

    print(f"run={name} device={device} model={model} imgsz={imgsz} epochs={epochs} "
          f"batch={batch} patience={patience} overrides={overrides}")
    PRETRAINED_DIR.mkdir(exist_ok=True)
    # Pretrained weights are downloaded into kobe/training/pretrained/
    yolo = YOLO(str(PRETRAINED_DIR / model))
    settings = dict(
        data=str(DATA_YAML),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=patience,  # stop when val fitness did not improve for this many epochs
        device=device,
        project=str(RUNS_DIR),
        name=name,
        exist_ok=False,     # never overwrite an earlier run
        seed=0,             # fixed seed for reproducibility
        cos_lr=True,        # cosine learning-rate decay
        close_mosaic=15,    # last 15 epochs without mosaic: train on realistic full frames
        plots=True,         # save results.png, confusion matrix, batch images
        workers=num_workers(),
    )
    settings.update(overrides)  # --set values win over the defaults above
    yolo.train(**settings)
    return RUNS_DIR / name


def parse_set(items):
    """['scale=0.9', 'fliplr=0'] -> {'scale': 0.9, 'fliplr': 0} (values parsed as YAML)."""
    out = {}
    for item in items:
        key, value = item.split("=", 1)
        out[key] = yaml.safe_load(value)
    return out


if __name__ == "__main__":
    device = pick_device()
    on_gpu = device != "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26m.pt" if on_gpu else "yolo26s.pt")
    parser.add_argument("--imgsz", type=int, default=1280 if on_gpu else 640)
    parser.add_argument("--epochs", type=int, default=150 if on_gpu else 10)
    parser.add_argument("--batch", type=int, default=-1 if on_gpu else 4, help="-1 = as large as GPU memory allows")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--name", help="run name (default: <model>_<imgsz>_<date-time>)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="extra ultralytics train argument, can be repeated")
    args = parser.parse_args()

    name = args.name or f"{args.model.removesuffix('.pt')}_{args.imgsz}_{datetime.now():%Y%m%d-%H%M}"
    train(name, args.model, args.imgsz, args.epochs, args.batch, args.patience, device, **parse_set(args.set))
    print(f"\nNext: python evaluate.py --run {name}")
