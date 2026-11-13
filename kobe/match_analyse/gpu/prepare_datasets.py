"""One-time: copy the labelled data (and the COCO model) from the repo into match_analyse/, so the folder is self-contained.

    venv\\Scripts\\python kobe\\match_analyse\\gpu\\prepare_datasets.py

The teammate does NOT need to run this: the result (datasets/ and models/yolo26l_coco.pt) is handed over with the folder.
Writes:
    datasets/roboflow_train/   the 100 Roboflow frames best.pt was trained on (repo data/dataset_yolo, only train/ exists)
    datasets/notebook_split/   the notebook split (repo kobe/notebooks/dataset): train / val / test, a SUBSET of the above
    datasets/ball_finetune/    the 100 Roboflow frames split by time block for the ball fine-tune:
                               train = frames < 200 and 246..599, val = 200..245; frames >= 600 (= notebook test) left out
    models/yolo26l_coco.pt     the pretrained COCO YOLO26l (baseline), copied from kobe/notebooks/yolo26l.pt
The data.yaml files have no 'path' key: ultralytics then resolves the folders from the yaml itself.
"""

import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpu.common import COCO_WEIGHTS, DATASETS_DIR, ROOT, frame_number, images_of, label_of  # noqa: E402

REPO = ROOT.parent.parent
ROBOFLOW = REPO / "data" / "dataset_yolo"
NOTEBOOK = REPO / "kobe" / "notebooks" / "dataset"
COCO_SRC = REPO / "kobe" / "notebooks" / "yolo26l.pt"
NAMES = ["ball", "player_A", "player_B", "player_B_goal_keeper", "referee"]
FT_VAL = range(200, 246)        # time block used for validation (early stopping / best epoch)
FT_EXCLUDED_FROM = 600          # frames >= 600 are the notebook test split: never in the fine-tune data


def copy_split(images, dst):
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    for img in images:
        shutil.copy2(img, dst / "images" / img.name)
        lab = label_of(img)
        if lab.exists():
            shutil.copy2(lab, dst / "labels" / lab.name)
    return len(images)


def write_yaml(dst, splits, comment):
    body = yaml.safe_dump({**splits, "nc": len(NAMES), "names": NAMES}, sort_keys=False)
    (dst / "data.yaml").write_text(comment + body, encoding="utf-8")


def main():
    for src in (ROBOFLOW / "train" / "images", NOTEBOOK):
        if not src.exists():
            sys.exit(f"bron ontbreekt: {src}")
    if DATASETS_DIR.exists():
        shutil.rmtree(DATASETS_DIR)

    rf = images_of([ROBOFLOW / "train" / "images"])
    dst = DATASETS_DIR / "roboflow_train"
    n = copy_split(rf, dst / "train")
    shutil.copy2(ROBOFLOW / "README.roboflow.txt", dst / "README.roboflow.txt")
    write_yaml(dst, {"train": "train/images", "val": "train/images"},
               "# The 100 Roboflow frames (export v2) that best.pt was trained on. Only train/ exists locally,\n"
               "# so val points to the same images (only usable for sanity checks, not as held-out data).\n")
    print(f"roboflow_train: {n} beelden")

    dst = DATASETS_DIR / "notebook_split"
    splits = {}
    for s in ("train", "val", "test"):
        n = copy_split(images_of([NOTEBOOK / s / "images"]), dst / s)
        splits[s] = f"{s}/images"
        print(f"notebook_split/{s}: {n} beelden")
    write_yaml(dst, splits, "# Notebook split (kobe/notebooks): train 0-62 + 525-588, val 200-245, test 600-658.\n"
                            "# WARNING: all of these frames are part of the 100 frames best.pt was trained on.\n")

    dst = DATASETS_DIR / "ball_finetune"
    train = [f for f in rf if frame_number(f) not in FT_VAL and frame_number(f) < FT_EXCLUDED_FROM]
    val = [f for f in rf if frame_number(f) in FT_VAL]
    print(f"ball_finetune: train {copy_split(train, dst / 'train')}, val {copy_split(val, dst / 'val')} beelden")
    write_yaml(dst, {"train": "train/images", "val": "val/images"},
               "# Ball fine-tune data: the Roboflow frames split by time block (no neighbouring frames across splits).\n"
               "# train = frames < 200 and 246..599, val = 200..245; frames >= 600 (notebook test) are left out.\n")

    if COCO_WEIGHTS.exists():
        print(f"COCO-model staat er al: {COCO_WEIGHTS}")
    elif COCO_SRC.exists():
        shutil.copy2(COCO_SRC, COCO_WEIGHTS)
        print(f"COCO-model gekopieerd -> {COCO_WEIGHTS}")
    else:
        print("COCO-model niet gevonden; check_env.py downloadt yolo26l.pt dan via ultralytics")


if __name__ == "__main__":
    main()
