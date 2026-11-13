"""Build the training dataset from the Roboflow export.

- Re-splits all labeled frames by the block rule in config.py (train/val/test,
  gap frames are dropped). Roboflow's own split folders are ignored.
- Drops a labeled frame when it is a duplicate of the previous frame and that
  previous frame is also in the dataset (same image twice adds nothing).
- Merges the team classes (see MERGE in config.py), unless --keep-teams.

Writes (inside kobe/training/, data/ is never modified):
    dataset/{train,val,test}/{images,labels}/
    dataset/data.yaml   ultralytics dataset config
"""

import argparse
import shutil
from collections import Counter

import yaml

from config import DATA_YAML, DATASET_DIR, EXPORT_DIR, MERGE, duplicate_frames, frame_idx, split_of

SPLITS = ["train", "val", "test"]

parser = argparse.ArgumentParser()
parser.add_argument("--keep-teams", action="store_true", help="keep player_A / player_B as separate classes")
args = parser.parse_args()

# ---- Class mapping: old class id -> new class id ----
# Class names come from the Roboflow export, never hardcoded
with open(EXPORT_DIR / "data.yaml", encoding="utf-8") as f:
    old_names = yaml.safe_load(f)["names"]
mapped = [name if args.keep_teams else MERGE.get(name, name) for name in old_names]
names = list(dict.fromkeys(mapped))  # unique, original order
remap = {old_id: names.index(new) for old_id, new in enumerate(mapped)}

# ---- Collect labeled frames from every split folder of the export ----
frames = {}  # idx -> (image path, label path)
for img in sorted(EXPORT_DIR.glob("*/images/*.jpg")):
    idx = frame_idx(img)
    if idx is None:
        print(f"WARNING: no idx in filename, skipped: {img.name}")
        continue
    if idx in frames:
        print(f"WARNING: frame {idx} exported twice, kept {frames[idx][0].name}")
        continue
    frames[idx] = (img, img.parent.parent / "labels" / (img.stem + ".txt"))

duplicates = duplicate_frames()

# ---- Write the dataset ----
if DATASET_DIR.exists():
    shutil.rmtree(DATASET_DIR)  # generated folder, always rebuilt from the export
for split in SPLITS:
    for sub in ["images", "labels"]:
        (DATASET_DIR / split / sub).mkdir(parents=True)

n_images = Counter()
counts = {split: Counter() for split in SPLITS}  # instances per new class id
empty = Counter()                                # images without any box
dropped = {"gap": [], "duplicate": []}

for idx, (img, label) in sorted(frames.items()):
    split = split_of(idx)
    if split == "gap":
        dropped["gap"].append(idx)
        continue
    if idx in duplicates and idx - 1 in frames:
        dropped["duplicate"].append(idx)
        continue

    lines = label.read_text().splitlines() if label.exists() else []
    new_lines = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        cid = remap[int(parts[0])]  # first column = class id
        new_lines.append(" ".join([str(cid)] + parts[1:]))
        counts[split][cid] += 1

    shutil.copy2(img, DATASET_DIR / split / "images" / img.name)
    (DATASET_DIR / split / "labels" / label.name).write_text("\n".join(new_lines) + "\n")
    n_images[split] += 1
    empty[split] += not new_lines

# Dataset config for ultralytics (test only when it has images, else val on 'test' fails)
config = {"path": str(DATASET_DIR), "train": "train/images", "val": "val/images"}
if n_images["test"]:
    config["test"] = "test/images"
config.update({"nc": len(names), "names": names})
with open(DATA_YAML, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)

# ---- Summary ----
print(f"Classes: {old_names} -> {names}")
print(f"\n{'':<14}" + "".join(f"{s:>8}" for s in SPLITS))
print(f"{'images':<14}" + "".join(f"{n_images[s]:>8}" for s in SPLITS))
print(f"{'  no boxes':<14}" + "".join(f"{empty[s]:>8}" for s in SPLITS))
for cid, name in enumerate(names):
    print(f"{name:<14}" + "".join(f"{counts[s][cid]:>8}" for s in SPLITS))
print(f"\nDropped (gap frames)      : {dropped['gap']}")
print(f"Dropped (duplicate frames): {dropped['duplicate']}")

# ---- Warnings ----
for split in SPLITS:
    for cid, name in enumerate(names):
        if counts[split][cid] == 0:
            print(f"WARNING: class '{name}' has 0 instances in {split}")
print(f"\nWrote {DATA_YAML}")
