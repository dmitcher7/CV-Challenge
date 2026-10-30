"""Shared paths and helpers for the smoke-test scripts."""

from pathlib import Path

# Folder of this file: brent/smoke_test/
SMOKE_DIR = Path(__file__).resolve().parent
# Repository root: CV-Challenge/ (two levels above this file)
REPO_ROOT = SMOKE_DIR.parent.parent

# Inputs (read-only, never modified)
DATASET_DIR = REPO_ROOT / "data" / "dataset_yolo"
ROBOFLOW_YAML = DATASET_DIR / "data.yaml"
VIDEO_PATH = REPO_ROOT / "data" / "raw_video" / "CV-video (1).mp4"

# Outputs (all inside brent/smoke_test/)
DATA_YAML = SMOKE_DIR / "data_smoke.yaml"
BALL_DATA_YAML = SMOKE_DIR / "data_ball.yaml"
RUNS_DIR = SMOKE_DIR / "runs"
RUN_NAME = "smoke"
BEST_WEIGHTS = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
BALL_WEIGHTS = RUNS_DIR / "ball" / "weights" / "best.pt"
EXPERIMENTS_CSV = SMOKE_DIR / "experiments.csv"


CPU_THREADS = 6  # measured optimum on an Intel Core Ultra 7 155H


def pick_device():
    """Return 0 (first CUDA GPU) if available, otherwise 'cpu'.

    On CPU we also limit torch to CPU_THREADS threads: on hybrid Intel CPUs
    the default (16 threads) also uses the slow E-cores and was ~3x slower.
    """
    import torch
    if torch.cuda.is_available():
        return 0
    torch.set_num_threads(CPU_THREADS)
    return "cpu"


def disable_label_cache():
    """Stop ultralytics from writing 'labels.cache' next to the labels.

    By default ultralytics saves a cache file in data/dataset_yolo/train/,
    but we are not allowed to modify data/. We replace the save function
    with one that only does its in-memory part (adding the 'version' key,
    which ultralytics reads afterwards) and skips writing the file.
    Labels are simply re-read each run (fast for ~50 images).
    """
    import ultralytics.data.dataset as ds

    def no_save(prefix, path, x, version):
        x["version"] = version  # ultralytics expects this key later

    ds.save_dataset_cache_file = no_save
