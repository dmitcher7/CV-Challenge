"""
Run inference on the soccer video with the trained ball & player model.
Outputs an annotated video with bounding boxes.
"""

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

from common import VIDEO_PATH, BEST_WEIGHTS, RUNS_DIR, RUN_NAME, pick_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(BEST_WEIGHTS), help="Model weights path")
    parser.add_argument("--video", type=str, default=str(VIDEO_PATH), help="Input .mp4 path")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference resolution")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[ERROR] Weights not found: {weights}")
        return

    device = pick_device()
    output_dir = RUNS_DIR / f"{RUN_NAME}_predict"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = output_dir / "annotated_match.mp4"

    print("=" * 60)
    print(f"[PREDICT] Video: {args.video}")
    print(f"[PREDICT] Output: {out_video_path}")
    print("=" * 60)

    model = YOLO(str(weights))
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))

    frame_idx = 0
    ball_detections = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)
        annotated = results[0].plot()

        # Count ball detections (class 0 is ball)
        classes = results[0].boxes.cls.cpu().tolist() if results[0].boxes else []
        if 0 in classes:
            ball_detections += 1

        out.write(annotated)
        frame_idx += 1
        if frame_idx % 60 == 0 or frame_idx == total_frames:
            print(f"Processed frame {frame_idx}/{total_frames} (Ball detected in {ball_detections} frames)")

    cap.release()
    out.release()
    print("=" * 60)
    print(f"[SUCCESS] Video saved: {out_video_path}")
    print(f"[SUMMARY] Total frames: {frame_idx} | Ball detected: {ball_detections} frames ({ball_detections/max(1,frame_idx)*100:.1f}%)")
    print("=" * 60)

if __name__ == "__main__":
    main()
