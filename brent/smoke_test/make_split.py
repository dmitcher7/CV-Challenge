"""Time-based train/val split of the Roboflow export.

Split rule (by original frame index, to avoid leakage between
near-identical neighbouring frames):
    train = idx 0-494, gap = idx 495-524 (never used), val = idx 525-659

Writes (inside brent/smoke_test/):
    train.txt, val.txt   absolute image paths, one per line
    data_smoke.yaml      ultralytics dataset config pointing to those files
"""

import re
from collections import Counter

import yaml

from common import DATASET_DIR, ROBOFLOW_YAML, DATA_YAML, SMOKE_DIR

TRAIN_MAX = 494   # last frame index used for training
VAL_MIN = 525     # first frame index used for validation

# Class names come from the Roboflow export, never hardcoded
with open(ROBOFLOW_YAML, encoding="utf-8") as f:
    names = yaml.safe_load(f)["names"]

images_dir = DATASET_DIR / "train" / "images"
labels_dir = DATASET_DIR / "train" / "labels"

splits = {"train": [], "val": []}
gap_frames = []
counts = {"train": Counter(), "val": Counter()}  # instances per class id
no_label = []

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
    splits[split].append(img)

    # Matching label file: same name, .txt extension, in labels/
    label = labels_dir / (img.stem + ".txt")
    lines = label.read_text().split("\n") if label.exists() else []
    lines = [line for line in lines if line.strip()]
    if not lines:
        no_label.append(img.name)
    for line in lines:
        counts[split][int(line.split()[0])] += 1  # first column = class id

# Write the image lists (absolute paths; ultralytics finds the labels
# by replacing /images/ with /labels/ in each path)
for split, imgs in splits.items():
    (SMOKE_DIR / f"{split}.txt").write_text("\n".join(str(p) for p in imgs) + "\n")

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

# ---- Summary ----
print(f"Images      : train={len(splits['train'])}  val={len(splits['val'])}  gap={len(gap_frames)}")
print(f"\n{'class':<22}{'train':>7}{'val':>7}")
for cid, name in enumerate(names):
    print(f"{name:<22}{counts['train'][cid]:>7}{counts['val'][cid]:>7}")
print(f"\nImages without labels: {len(no_label)} {no_label if no_label else ''}")
print(f"Frames in gap (should be 0): {len(gap_frames)} {gap_frames if gap_frames else ''}")

# ---- Warnings ----
if not splits["val"]:
    print("WARNING: val split is empty!")
for split in splits:
    for cid, name in enumerate(names):
        if counts[split][cid] == 0:
            print(f"WARNING: class '{name}' has 0 instances in {split}")
print(f"\nWrote {DATA_YAML}")
