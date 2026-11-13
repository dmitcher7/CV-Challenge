"""Quantitatively evaluate the trained YOLO model on the validation dataset."""

from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_YAML = REPO_ROOT / "data" / "dataset_yolo" / "data.yaml"
WEIGHTS_PATH = SCRIPT_DIR.parent / "model" / "player_training" / "weights" / "best.pt"


def main() -> None:
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Trained weights not found: {WEIGHTS_PATH}. Run training first.")
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {DATA_YAML}")

    print(f"Loading weights from: {WEIGHTS_PATH}")
    model = YOLO(str(WEIGHTS_PATH))

    print("Running validation evaluation...")
    metrics = model.val(data=str(DATA_YAML))

    print("\n=== Evaluation Results ===")
    print(f"mAP@50    : {metrics.box.map50:.4f}")
    print(f"mAP@50:95 : {metrics.box.map:.4f}")
    print("==========================")


if __name__ == "__main__":
    main()