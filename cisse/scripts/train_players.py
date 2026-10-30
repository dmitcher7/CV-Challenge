"""Train a YOLO player-detection model on the project dataset."""

from pathlib import Path

from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_YAML = REPO_ROOT / "data" / "dataset_yolo" / "data.yaml"
MODEL_DIR = SCRIPT_DIR.parent / "model"
RUN_NAME = "player_training"
BEST_WEIGHTS = MODEL_DIR / RUN_NAME / "weights" / "best.pt"


def main() -> None:
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {DATA_YAML}")

    disable_label_cache()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolo26l.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=1280,
        project=str(MODEL_DIR),
        name=RUN_NAME,
        exist_ok=True,
        workers=0,
        batch=8,
    )

    if not BEST_WEIGHTS.is_file():
        raise FileNotFoundError(f"Training completed without saving best weights: {BEST_WEIGHTS}")
    print(f"Best weights saved to: {BEST_WEIGHTS}")


def disable_label_cache() -> None:
    """Keep training from writing label-cache files into the source dataset."""
    import ultralytics.data.dataset as dataset

    def no_save(prefix, path, cache, version):
        cache["version"] = version

    dataset.save_dataset_cache_file = no_save


if __name__ == "__main__":
    main()
