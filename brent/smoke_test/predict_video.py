"""Run best.pt on the match video and save an annotated copy."""

import argparse

import yaml
from ultralytics import YOLO

from common import BEST_WEIGHTS, RUNS_DIR, RUN_NAME, VIDEO_PATH, pick_device

# Use the same image size as during training (stored in args.yaml)
with open(BEST_WEIGHTS.parent.parent / "args.yaml", encoding="utf-8") as f:
    train_imgsz = yaml.safe_load(f)["imgsz"]

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default=str(BEST_WEIGHTS))
parser.add_argument("--source", default=str(VIDEO_PATH), help="path to the input video")
parser.add_argument("--imgsz", type=int, default=train_imgsz)
parser.add_argument("--conf", type=float, default=0.25)
args = parser.parse_args()

model = YOLO(args.weights)
results = model.predict(
    source=args.source,
    imgsz=args.imgsz,
    conf=args.conf,
    device=pick_device(),
    save=True,                    # write the annotated video
    project=str(RUNS_DIR),
    name=f"{RUN_NAME}_predict",
    exist_ok=True,
    stream=True,                  # yield frame by frame -> low memory use
    verbose=False,
)

# With stream=True nothing happens until we iterate over the results
for n, _ in enumerate(results, start=1):
    if n % 100 == 0:
        print(f"processed {n} frames")
print(f"Done: {n} frames, saved in {RUNS_DIR / (RUN_NAME + '_predict')}")
