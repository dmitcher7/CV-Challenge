"""Run best.pt on the match video and save a clearly annotated copy."""

import argparse
from pathlib import Path

import cv2
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
parser.add_argument("--line-width", type=int, default=2)
parser.add_argument("--font-scale", type=float, default=0.45)
args = parser.parse_args()

# Fixed high-contrast BGR colors (OpenCV order), keyed by dataset class name.
CLASS_COLORS = {
    "ball": (0, 0, 255),
    "player_A": (255, 80, 0),
    "player_B": (0, 220, 255),
    "player_B_goal_keeper": (255, 0, 255),
    "referee": (0, 200, 0),
}
FALLBACK_COLOR = (255, 255, 0)


def draw_detections(result):
    """Draw compact class labels, with the ball rendered above all other detections."""
    frame = result.orig_img.copy()
    if result.boxes is None:
        return frame

    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = result.names[class_id]
        detections.append((class_name == "ball", class_name, float(box.conf.item()), box.xyxy[0].tolist()))

    # Non-ball detections first; ball boxes and labels can therefore never be painted over.
    detections.sort(key=lambda detection: detection[0])
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1

    for _, class_name, confidence, coords in detections:
        x1, y1, x2, y2 = (round(value) for value in coords)
        color = CLASS_COLORS.get(class_name, FALLBACK_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, args.line_width, cv2.LINE_AA)

        label = f"{class_name} {confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, args.font_scale, font_thickness
        )
        label_left = max(0, min(x1, frame.shape[1] - text_width - 6))
        label_bottom = y1 if y1 >= text_height + baseline + 6 else y1 + text_height + baseline + 6
        label_top = label_bottom - text_height - baseline - 6
        cv2.rectangle(
            frame,
            (label_left, label_top),
            (label_left + text_width + 6, label_bottom),
            color,
            cv2.FILLED,
        )
        brightness = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        text_color = (0, 0, 0) if brightness > 145 else (255, 255, 255)
        cv2.putText(
            frame,
            label,
            (label_left + 3, label_bottom - baseline - 3),
            font,
            args.font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

    return frame


model = YOLO(args.weights)
results = model.predict(
    source=args.source,
    imgsz=args.imgsz,
    conf=args.conf,
    device=pick_device(),
    stream=True,                  # yield frame by frame -> low memory use
    verbose=False,
)

source_capture = cv2.VideoCapture(args.source)
fps = source_capture.get(cv2.CAP_PROP_FPS) or 30.0
source_capture.release()

output_dir = RUNS_DIR / f"{RUN_NAME}_predict"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{Path(args.source).stem}.avi"
writer = None
n = 0

try:
    # With stream=True inference starts when we iterate over the results.
    for n, result in enumerate(results, start=1):
        annotated = draw_detections(result)
        if writer is None:
            height, width = annotated.shape[:2]
            writer = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {output_path}")
        writer.write(annotated)
        if n % 100 == 0:
            print(f"processed {n} frames")
finally:
    if writer is not None:
        writer.release()

print(f"Done: {n} frames, saved as {output_path}")
