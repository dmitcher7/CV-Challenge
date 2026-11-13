"""Shared paths, split rule and helpers for all scripts in kobe/training/."""

import os
import re
from pathlib import Path

# Folder of this file: kobe/training/
TRAIN_DIR = Path(__file__).resolve().parent
# Repository root (two levels up)
REPO_ROOT = TRAIN_DIR.parent.parent

# Inputs (read-only, never modified)
EXPORT_DIR = REPO_ROOT / "data" / "dataset_yolo"   # Roboflow export (any split folders)
FRAMES_DIR = REPO_ROOT / "data" / "frames"         # all 660 frames of the clip
VIDEO_PATH = REPO_ROOT / "data" / "raw_video" / "match.mp4"

# Outputs (all inside kobe/training/; dataset/, to_label/ and runs/ are not committed)
DATASET_DIR = TRAIN_DIR / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
TO_LABEL_DIR = TRAIN_DIR / "to_label"
PRETRAINED_DIR = TRAIN_DIR / "pretrained"
RUNS_DIR = TRAIN_DIR / "runs"
EXPERIMENTS_CSV = TRAIN_DIR / "experiments.csv"

# ---- Split rule ----
# The clip has 660 frames (22 s at 30 fps). We cut it into blocks of 60 frames (2 s)
# and give every block to one split, so val/test come from several moments in the
# clip instead of only the end. Train frames closer than GAP frames to a val/test
# block are never used (near-identical neighbours would leak into the evaluation).
N_FRAMES = 660
BLOCK = 60
BLOCK_SPLITS = ["train", "train", "val", "train", "train", "test",
                "train", "train", "val", "train", "test"]
GAP = 10

# ---- Classes ----
# Team colour is not learned by the detector: both teams become 'player' and are
# split afterwards by jersey colour (K-means). Classes not listed keep their name.
MERGE = {"player_A": "player", "player_B": "player", "player_B_goal_keeper": "goalkeeper"}

# A frame is a duplicate when it is (almost) identical to the previous frame:
# mean absolute grey-value difference below this threshold (0-255 scale)
DUPLICATE_THRESHOLD = 0.5

CPU_THREADS = 6  # fastest on Kobe's Core Ultra 7 155H (6 P-cores), measured


def frame_idx(path):
    """Original frame index from a name like frame_0000_idx_00000_jpg.rf.<hash>.jpg"""
    match = re.search(r"idx_(\d+)", Path(path).name)
    return int(match.group(1)) if match else None


def block_split(idx):
    """Split of the block that frame idx belongs to (ignoring the gap)."""
    return BLOCK_SPLITS[min(idx // BLOCK, len(BLOCK_SPLITS) - 1)]


def split_of(idx):
    """'train', 'val', 'test' or 'gap' for an original frame index."""
    split = block_split(idx)
    if split == "train":
        near = range(max(0, idx - GAP), min(N_FRAMES, idx + GAP + 1))
        if any(block_split(j) != "train" for j in near):
            return "gap"
    return split


def duplicate_frames():
    """Set of frame indices that are (almost) identical to the previous frame.

    The clip was converted from 25 to 30 fps, so every 6th frame is a copy,
    plus a few frozen frames. Labeling those adds work but no information.
    """
    import cv2
    import numpy as np

    duplicates, prev = set(), None
    for path in sorted(FRAMES_DIR.glob("*.jpg"), key=frame_idx):
        small = cv2.cvtColor(cv2.resize(cv2.imread(str(path)), (480, 270)), cv2.COLOR_BGR2GRAY)
        small = small.astype(np.float32)
        if prev is not None and np.abs(small - prev).mean() < DUPLICATE_THRESHOLD:
            duplicates.add(frame_idx(path))
        prev = small
    return duplicates


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


def num_workers():
    """Dataloader workers: 0 on Windows (multiprocessing issues), 4 elsewhere (Kaggle/Colab)."""
    return 0 if os.name == "nt" else 4


def run_dir(name):
    """Folder of a training run; fails with a clear message if it does not exist."""
    path = RUNS_DIR / name
    if not (path / "weights" / "best.pt").exists():
        runs = sorted(p.name for p in RUNS_DIR.glob("*") if (p / "weights" / "best.pt").exists())
        raise SystemExit(f"No weights in {path}. Available runs: {runs or 'none'}")
    return path
