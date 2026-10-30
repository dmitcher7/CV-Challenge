"""Time-based train/val split of the Roboflow export.

Split rule (by original frame index, to avoid leakage between
near-identical neighbouring frames):
    train = idx 0-494, gap = idx 495-524 (never used), val = idx 525-659

Writes (inside brent/smoke_test/):
    train.txt, val.txt   absolute image paths, one per line
    data_smoke.yaml      ultralytics dataset config pointing to those files
    train_ball.txt, val_ball.txt, data_ball.yaml

Frames labeled only for the ball are excluded from the multi-class lists. They
remain available to the dedicated ball model, which filters all non-ball labels.
"""

import re
from collections import Counter

import yaml

from common import BALL_DATA_YAML, DATASET_DIR, ROBOFLOW_YAML, DATA_YAML, SMOKE_DIR

TRAIN_MAX = 494   # last frame index used for training
VAL_MIN = 525     # first frame index used for validation

# Class names come from the Roboflow export, never hardcoded
with open(ROBOFLOW_YAML, encoding="utf-8") as f:
    names = yaml.safe_load(f)["names"]

images_dir = DATASET_DIR / "train" / "images"
labels_dir = DATASET_DIR / "train" / "labels"

splits = {"train": [], "val": []}  # fully labeled frames for the multi-class model
ball_splits = {"train": [], "val": []}  # all frames for the dedicated ball model
gap_frames = []
counts = {"train": Counter(), "val": Counter()}  # instances on fully labeled frames
ball_counts = {"train": Counter(), "val": Counter()}  # instances on all frames
no_label = []
ball_only = []

for img in sorted(images_dir.glob("*.jpg")):
    # Filename looks like frame_0000_idx_00000_jpg.rf.<hash>.jpg
    match = re.search(r"idx_(\d+)", img.name)
    if match is None:
        print(f"WARNING: no idx in filename, skipped: {img.name}")
        continue
    idx = int(match.group(1))

    if idx <= TRAIN_MAX:
        split = "train"
    elif idx >= VAL_MIN:
        split = "val"
    else:
        gap_frames.append(img.name)  # in the gap -> never used
        continue
    ball_splits[split].append(img)

    # Matching label file: same name, .txt extension, in labels/
    label = labels_dir / (img.stem + ".txt")
    lines = label.read_text().split("\n") if label.exists() else []
    lines = [line for line in lines if line.strip()]
    if not lines:
        no_label.append(img.name)
    class_ids = [int(line.split()[0]) for line in lines]
    is_ball_only = bool(class_ids) and all(cid == 0 for cid in class_ids)
    if is_ball_only:
        # In this football footage, a full annotation always contains players.
        # Treat a ball-only file as a deliberately partial annotation.
        ball_only.append(img.name)
    elif lines:
        splits[split].append(img)
    for line in lines:
        cid = int(line.split()[0])  # first column = class id
        ball_counts[split][cid] += 1
        if not is_ball_only:
            counts[split][cid] += 1

# Write the image lists (absolute paths; ultralytics finds the labels
# by replacing /images/ with /labels/ in each path)
for split, imgs in splits.items():
    (SMOKE_DIR / f"{split}.txt").write_text("\n".join(str(p) for p in imgs) + "\n")
for split, imgs in ball_splits.items():
    (SMOKE_DIR / f"{split}_ball.txt").write_text("\n".join(str(p) for p in imgs) + "\n")

# Dataset config for ultralytics
config = {
    "path": str(SMOKE_DIR),
    "train": "train.txt",
    "val": "val.txt",
    "nc": len(names),
    "names": names,
}
with open(DATA_YAML, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)

ball_config = {
    **config,
    "train": "train_ball.txt",
    "val": "val_ball.txt",
}
with open(BALL_DATA_YAML, "w", encoding="utf-8") as f:
    yaml.safe_dump(ball_config, f, sort_keys=False)

# ---- Summary ----
print(
    f"Multi-class images: train={len(splits['train'])}  val={len(splits['val'])}\n"
    f"Ball-model images : train={len(ball_splits['train'])}  val={len(ball_splits['val'])}  "
    f"gap={len(gap_frames)}"
)
print(f"\n{'class':<22}{'multi train':>12}{'multi val':>11}{'ball train':>12}{'ball val':>10}")
for cid, name in enumerate(names):
    print(
        f"{name:<22}{counts['train'][cid]:>12}{counts['val'][cid]:>11}"
        f"{ball_counts['train'][cid]:>12}{ball_counts['val'][cid]:>10}"
    )
print(f"\nImages without labels: {len(no_label)} {no_label if no_label else ''}")
print(f"Ball-only partial labels (excluded from multi-class): {len(ball_only)}")
print(f"Frames in gap (should be 0): {len(gap_frames)} {gap_frames if gap_frames else ''}")

# ---- Warnings ----
if not splits["val"]:
    print("WARNING: val split is empty!")
for split in splits:
    for cid, name in enumerate(names):
        if counts[split][cid] == 0:
            print(f"WARNING: class '{name}' has 0 instances in {split}")
print(f"\nWrote {DATA_YAML} and {BALL_DATA_YAML}")
