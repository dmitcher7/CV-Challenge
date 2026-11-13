"""
Time-based train/val split using the 100 Roboflow-annotated frames.
Generates:
    - dimi/train.txt
    - dimi/val.txt
    - dimi/data_dimi.yaml
"""

import sys
from pathlib import Path
import re
from collections import Counter
import yaml

# Ensure src directory is on sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import DATASET_DIR, ROBOFLOW_YAML, DATA_YAML, TRAIN_TXT, VAL_TXT, DIMI_DIR

TRAIN_MAX = 494   # Frame index <= 494 -> Training set
VAL_MIN = 525     # Frame index >= 525 -> Validation set

# Read class names
with open(ROBOFLOW_YAML, encoding="utf-8") as f:
    rf_data = yaml.safe_load(f)
    names = rf_data["names"]

images_dir = DATASET_DIR / "train" / "images"
labels_dir = DATASET_DIR / "train" / "labels"

splits = {"train": [], "val": []}
gap_frames = []
counts = {"train": Counter(), "val": Counter()}
missing_labels = []

for img in sorted(images_dir.glob("*.jpg")):
    match = re.search(r"idx_(\d+)", img.name)
    if not match:
        continue
    idx = int(match.group(1))

    if idx <= TRAIN_MAX:
        split = "train"
    elif idx >= VAL_MIN:
        split = "val"
    else:
        gap_frames.append(img.name)
        continue

    splits[split].append(img.resolve())

    # Check matching label file
    label_path = labels_dir / (img.stem + ".txt")
    if label_path.exists():
        lines = [l.strip() for l in label_path.read_text(encoding="utf-8").split("\n") if l.strip()]
        if not lines:
            missing_labels.append(img.name)
        for line in lines:
            cid = int(line.split()[0])
            counts[split][cid] += 1
    else:
        missing_labels.append(img.name)

# Save image lists
TRAIN_TXT.write_text("\n".join(str(p) for p in splits["train"]) + "\n", encoding="utf-8")
VAL_TXT.write_text("\n".join(str(p) for p in splits["val"]) + "\n", encoding="utf-8")

# Write YOLO dataset configuration
dataset_cfg = {
    "path": str(DIMI_DIR),
    "train": str(TRAIN_TXT),
    "val": str(VAL_TXT),
    "nc": len(names),
    "names": names,
}

with open(DATA_YAML, "w", encoding="utf-8") as f:
    yaml.safe_dump(dataset_cfg, f, sort_keys=False)

print("=" * 55)
print(f"[INFO] DATASET SPLIT OVERZICHT (Totaal: {len(splits['train']) + len(splits['val'])} frames)")
print("=" * 55)
print(f"Train beelden (idx <= {TRAIN_MAX}) : {len(splits['train'])}")
print(f"Val beelden   (idx >= {VAL_MIN}) : {len(splits['val'])}")
print(f"Gap beelden   (buffer)         : {len(gap_frames)}")
print("-" * 55)
print(f"{'Klasse':<24}{'Train':>8}{'Validation':>12}")
print("-" * 55)
for cid, name in enumerate(names):
    print(f"{name:<24}{counts['train'][cid]:>8}{counts['val'][cid]:>12}")
print("-" * 55)
print(f"[OK] data_dimi.yaml opgeslagen: {DATA_YAML}")
print("=" * 55)
