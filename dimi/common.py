"""Shared paths and helper functions for Dimi's ball detection model."""

from pathlib import Path
import torch

# Folder of this script: CV-Challenge/dimi/
DIMI_DIR = Path(__file__).resolve().parent
REPO_ROOT = DIMI_DIR.parent

# Inputs
DATASET_DIR = REPO_ROOT / "data" / "dataset_yolo"
ROBOFLOW_YAML = DATASET_DIR / "data.yaml"
VIDEO_PATH = REPO_ROOT / "data" / "raw_video" / "CV-video.mp4"

# Outputs inside dimi/
DATA_YAML = DIMI_DIR / "data_dimi.yaml"
TRAIN_TXT = DIMI_DIR / "train.txt"
VAL_TXT = DIMI_DIR / "val.txt"
RUNS_DIR = DIMI_DIR / "runs"
RUN_NAME = "ball_yolo11_1280"
BEST_WEIGHTS = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
EXPERIMENTS_CSV = DIMI_DIR / "experiments.csv"


def pick_device():
    """Returns CUDA device 0 if NVIDIA GPU is available, else CPU."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[DEVICE] Using GPU: {device_name} (CUDA enabled)")
        return 0
    print("[DEVICE] CUDA not available, falling back to CPU")
    return "cpu"


def disable_label_cache():
    """Prevents ultralytics from creating a labels.cache file in data/."""
    try:
        import ultralytics.data.dataset as ds
        def no_save(prefix, path, x, version):
            x["version"] = version
        ds.save_dataset_cache_file = no_save
    except Exception as e:
        print("[WARN] Could not patch label cache:", e)
