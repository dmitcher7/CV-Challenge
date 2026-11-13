"""Shared paths and helper functions for Dimi's ball detection model."""

import sys
from pathlib import Path
import torch

# Paths: Folder of this script is CV-Challenge/dimi/src/
SRC_DIR = Path(__file__).resolve().parent
DIMI_DIR = SRC_DIR.parent
REPO_ROOT = DIMI_DIR.parent

CONFIG_DIR = DIMI_DIR / "config"
DATA_DIR = DIMI_DIR / "data"
NOTEBOOKS_DIR = DIMI_DIR / "notebooks"
RUNS_DIR = DIMI_DIR / "runs"

# Ensure src/ is on sys.path for direct imports
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Global Project Inputs
DATASET_DIR = REPO_ROOT / "data" / "dataset_yolo"
ROBOFLOW_YAML = DATASET_DIR / "data.yaml"
VIDEO_PATH = REPO_ROOT / "data" / "raw_video" / "CV-video.mp4"
PITCH_IMG_PATH = REPO_ROOT / "data" / "pitch_template" / "tactical_pitch.png"

# Configurations inside dimi/config/
DATA_YAML = CONFIG_DIR / "data_dimi.yaml"
HOMOGRAPHY_CONFIG = CONFIG_DIR / "homography_config.json"
BENCHMARK_CSV = CONFIG_DIR / "iou_nms_benchmark.csv"
EXPERIMENTS_CSV = CONFIG_DIR / "experiments.csv"

# Data inside dimi/data/
TRAIN_TXT = DATA_DIR / "train.txt"
VAL_TXT = DATA_DIR / "val.txt"

# Model weights and runs
RUN_NAME = "ball_yolo26_1280"
BEST_WEIGHTS = RUNS_DIR / RUN_NAME / "weights" / "best.pt"


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
