"""How much still needs labeling, and which frames to label next.

Per split (block rule in config.py) it shows how many frames are usable
(not in the gap, not a duplicate), how many are labeled, and how many extra
frames are needed to reach TARGET instances per class.

Then it ranks the unlabeled usable frames per split, spread out in time:
each next frame is the one furthest away from everything already labeled or
picked (farthest-point sampling on the frame index). Neighbouring frames are
near-identical, so spreading them gives the most new information per label.

Writes label_candidates.csv. With --copy it also copies the frames that are
needed into to_label/ (ready to upload to Roboflow).
"""

import argparse
import csv
import math
import shutil
from collections import Counter

import yaml

from config import (EXPORT_DIR, FRAMES_DIR, MERGE, N_FRAMES, TO_LABEL_DIR, TRAIN_DIR,
                    duplicate_frames, frame_idx, split_of)

SPLITS = ["val", "test", "train"]  # label the evaluation splits first

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=int, default=100, help="target instances per class in train")
parser.add_argument("--val", type=int, default=25, help="target instances per class in val")
parser.add_argument("--test", type=int, default=25, help="target instances per class in test")
parser.add_argument("--copy", action="store_true", help="copy the needed frames into to_label/")
args = parser.parse_args()
target = {"train": args.train, "val": args.val, "test": args.test}

# ---- Labeled frames (classes merged like prepare_dataset.py) ----
with open(EXPORT_DIR / "data.yaml", encoding="utf-8") as f:
    old_names = yaml.safe_load(f)["names"]
names = list(dict.fromkeys(MERGE.get(n, n) for n in old_names))

labeled = {}  # idx -> Counter {class name: number of boxes}
for label in EXPORT_DIR.glob("*/labels/*.txt"):
    ids = [int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()]
    labeled[frame_idx(label)] = Counter(MERGE.get(old_names[i], old_names[i]) for i in ids)

duplicates = duplicate_frames()
# Same rule as prepare_dataset.py: a duplicate of a labeled frame is not counted
labeled = {i: c for i, c in labeled.items() if not (i in duplicates and i - 1 in labeled)}
usable = {s: [i for i in range(N_FRAMES) if split_of(i) == s and i not in duplicates] for s in SPLITS}
print(f"{len(labeled)} labeled frames, {len(duplicates)} duplicate frames in the clip")

# Average instances per labeled frame over the whole set, used when a split has no labels yet
all_counts = sum(labeled.values(), Counter())
overall_rate = {n: all_counts[n] / len(labeled) for n in names} if labeled else {}

# ---- How many extra frames per split? ----
needed = {}
for split in SPLITS:
    lab = [i for i in labeled if split_of(i) == split]
    free = [i for i in usable[split] if i not in labeled]
    have = sum((labeled[i] for i in lab), Counter())
    print(f"\n=== {split}: {len(lab)} labeled, {len(free)} usable unlabeled frames left,"
          f" target {target[split]} instances/class ===")
    print(f"{'class':<14}{'have':>6}{'per frame':>11}{'extra frames':>14}")
    worst = 0
    for name in names:
        rate = have[name] / len(lab) if lab and have[name] else overall_rate.get(name, 0)
        if rate == 0:
            print(f"{name:<14}{have[name]:>6}   no labels yet -> cannot estimate")
            continue
        extra = math.ceil(max(0, target[split] - have[name]) / rate)
        worst = max(worst, extra)
        print(f"{name:<14}{have[name]:>6}{rate:>11.2f}{extra:>14}")
    needed[split] = min(worst, len(free))
    if worst > len(free):
        print(f"-> NOT reachable: {worst} frames needed, only {len(free)} left. Lower the target.")
    else:
        print(f"-> label about {worst} more frames")


# ---- Which frames? ----
def spread_order(pool, anchors):
    """Order pool so that each next frame is as far as possible from anchors + earlier picks."""
    anchors, order, pool = list(anchors), [], list(pool)
    while pool:
        best = max(pool, key=lambda i: min((abs(i - a) for a in anchors), default=N_FRAMES))
        order.append(best)
        anchors.append(best)
        pool.remove(best)
    return order


csv_path = TRAIN_DIR / "label_candidates.csv"
if args.copy and TO_LABEL_DIR.exists():
    shutil.rmtree(TO_LABEL_DIR)  # generated folder, only holds the current batch
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "priority", "needed", "frame_idx", "file"])
    for split in SPLITS:
        pool = [i for i in usable[split] if i not in labeled]
        anchors = [i for i in labeled if split_of(i) == split]
        order = spread_order(pool, anchors)
        for prio, idx in enumerate(order, start=1):
            path = next(FRAMES_DIR.glob(f"*_idx_{idx:05d}.jpg"))
            is_needed = prio <= needed[split]
            writer.writerow([split, prio, int(is_needed), idx, path.relative_to(TRAIN_DIR.parent.parent).as_posix()])
            if args.copy and is_needed:
                TO_LABEL_DIR.mkdir(exist_ok=True)
                shutil.copy2(path, TO_LABEL_DIR / path.name)
        print(f"{split}: first 15 to label: {order[:15]}")

print(f"\nWrote {csv_path}")
if args.copy:
    print(f"Copied {sum(needed.values())} frames to {TO_LABEL_DIR}")
