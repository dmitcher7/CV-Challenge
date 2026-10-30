"""Track players in the project video using the trained YOLO model."""

from pathlib import Path
import cv2
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
VIDEO_PATH = REPO_ROOT / "data" / "raw_video" / "CV-video.mp4"
WEIGHTS_PATH = (
    SCRIPT_DIR.parent / "model" / "player_training" / "weights" / "best.pt"
)
OUTPUT_DIR = SCRIPT_DIR / "runs"
OUTPUT_NAME = "player_tracking"

# Updated colors in BGR format (Blue, Green, Red)
COLORS = {
    "team_a": (112, 65, 0),     # PSG Blue
    "team_b": (171, 229, 243),  # Inter Milan Away Yellow
    "referee": (235, 206, 135), # Light Blue
    "ball": (0, 0, 255),        # Red
}

def main() -> None:
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"Trained weights not found: {WEIGHTS_PATH}. Run train_players.py first."
        )
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Input video not found: {VIDEO_PATH}")

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {VIDEO_PATH}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if fps <= 0:
        raise RuntimeError(f"Could not read the input video's frame rate: {VIDEO_PATH}")

    output_dir = OUTPUT_DIR / OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / VIDEO_PATH.with_suffix(".avi").name

    model = YOLO(str(WEIGHTS_PATH))
    results = model.track(
        source=str(VIDEO_PATH),
        tracker="bytetrack.yaml",
        persist=True,
        save=False,
        stream=True,
        verbose=False,
    )

    frame_count = 0
    writer = None
    try:
        for frame_count, result in enumerate(results, start=1):
            frame = result.orig_img.copy()
            draw_detections(frame, result.boxes, model.names)
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"XVID"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {output_path}")
            writer.write(frame)
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
    finally:
        if writer is not None:
            writer.release()

    if frame_count == 0:
        raise RuntimeError(f"No frames were processed from: {VIDEO_PATH}")

    print(f"Tracking complete: {frame_count} frames saved to {output_path}")


def draw_detections(frame, boxes, class_names) -> None:
    if boxes is None or len(boxes) == 0:
        return

    coordinates = boxes.xyxy.cpu().numpy().astype(int)
    class_ids = boxes.cls.cpu().numpy().astype(int)
    track_ids = boxes.id
    if track_ids is not None:
        track_ids = track_ids.cpu().numpy().astype(int)

    for index, (x1, y1, x2, y2) in enumerate(coordinates):
        class_id = class_ids[index]
        class_name = (
            str(class_names.get(class_id, "object"))
            if isinstance(class_names, dict)
            else str(class_names[class_id])
        )
        category = category_for_class(class_name)
        color = COLORS.get(category, (245, 245, 245))
        label = class_name
        if track_ids is not None:
            label = f"{label} #{track_ids[index]}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        label_y = max(y1, text_height + baseline + 4)
        cv2.rectangle(
            frame,
            (x1, label_y - text_height - baseline - 4),
            (x1 + text_width + 6, label_y),
            color,
            -1,
        )
        
        # Adjust text color for readability depending on background brightness
        text_color = (0, 0, 0) if category in ["team_b", "referee"] else (255, 255, 255)
        
        cv2.putText(
            frame,
            label,
            (x1 + 3, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_AA,
        )


def category_for_class(class_name: str) -> str | None:
    name = class_name.lower()
    if "ball" in name:
        return "ball"
    if "referee" in name:
        return "referee"
    if "player_a" in name:
        return "team_a"
    if "player_b" in name:
        return "team_b"
    return None


if __name__ == "__main__":
    main()