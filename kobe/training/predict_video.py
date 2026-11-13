"""Run a trained model on the match video and save an annotated .mp4."""

import argparse

import cv2
import yaml
from ultralytics import YOLO

from config import RUNS_DIR, VIDEO_PATH, pick_device, run_dir

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, help="folder name in runs/")
parser.add_argument("--conf", type=float, default=0.25, help="see 'best conf' in evaluate.py")
parser.add_argument("--video", default=str(VIDEO_PATH))
args = parser.parse_args()

folder = run_dir(args.run)
# Use the same image size as during training (stored in args.yaml)
imgsz = yaml.safe_load((folder / "args.yaml").read_text(encoding="utf-8"))["imgsz"]
model = YOLO(str(folder / "weights" / "best.pt"))

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
cap.release()

out_path = RUNS_DIR / f"{args.run}_predict.mp4"
# Our own writer, because ultralytics writes .avi on Windows
writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
results = model.predict(source=args.video, imgsz=imgsz, conf=args.conf, device=pick_device(),
                        stream=True, verbose=False)  # stream: frame by frame, low memory use
n = 0
for n, res in enumerate(results, start=1):
    writer.write(res.plot())  # frame with boxes drawn on it
    if n % 100 == 0:
        print(f"processed {n} frames")
writer.release()
print(f"Done: {n} frames -> {out_path}")
