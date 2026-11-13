"""List every false positive, false negative and class mix-up of a run on one split.

A prediction and a ground-truth box match when they overlap with IoU >= --iou.
    TP        same class, matched
    FP        prediction without any matching ground truth (e.g. ball on an advertising board)
    FN        ground truth that no prediction found (e.g. ball hidden behind a leg)
    CONFUSION matched box, wrong class (e.g. referee predicted as player)

Writes runs/<run>_errors_<split>/:
    errors.csv  one row per error, with box size and an empty 'cause' column
                to fill in by hand (blur, occlusion, board, too small, label error ...)
    crops/      a zoomed crop around each error (green = ground truth, red = prediction)
"""

import argparse
import csv
from collections import Counter

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from config import DATASET_DIR, RUNS_DIR, frame_idx, pick_device, run_dir

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, help="folder name in runs/")
parser.add_argument("--split", default="val", choices=["train", "val", "test"])
parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold for predictions")
parser.add_argument("--iou", type=float, default=0.5, help="IoU needed for a match")
args = parser.parse_args()

folder = run_dir(args.run)
imgsz = yaml.safe_load((folder / "args.yaml").read_text(encoding="utf-8"))["imgsz"]
model = YOLO(str(folder / "weights" / "best.pt"))
names = model.names
out_dir = RUNS_DIR / f"{args.run}_errors_{args.split}"
(out_dir / "crops").mkdir(parents=True, exist_ok=True)


def iou(a, b):
    """IoU of two xyxy boxes."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def read_gt(label_path, width, height):
    """YOLO label file -> list of (class id, xyxy box in pixels)."""
    boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cid, xc, yc, w, h = line.split()
        xc, yc, w, h = float(xc) * width, float(yc) * height, float(w) * width, float(h) * height
        boxes.append((int(cid), [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]))
    return boxes


def save_crop(image, gt_box, pred_box, path):
    """Crop around the error (at least 160 px, 4x the box) with both boxes drawn."""
    boxes = [b for b in (gt_box, pred_box) if b is not None]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(80, 2 * max(x1 - x0, y1 - y0))
    h, w = image.shape[:2]
    left, top = int(max(0, cx - half)), int(max(0, cy - half))
    right, bottom = int(min(w, cx + half)), int(min(h, cy + half))
    crop = image[top:bottom, left:right].copy()
    for box, colour in ((gt_box, (0, 200, 0)), (pred_box, (0, 0, 255))):
        if box is not None:
            p0 = (int(box[0]) - left, int(box[1]) - top)
            p1 = (int(box[2]) - left, int(box[3]) - top)
            cv2.rectangle(crop, p0, p1, colour, 1)
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)  # zoom in
    cv2.imwrite(str(path), crop)


rows = []
tp = Counter()
images = sorted((DATASET_DIR / args.split / "images").glob("*.jpg"))
for img_path in images:
    image = cv2.imread(str(img_path))
    height, width = image.shape[:2]
    gt = read_gt(DATASET_DIR / args.split / "labels" / (img_path.stem + ".txt"), width, height)

    res = model.predict(image, imgsz=imgsz, conf=args.conf, device=pick_device(), verbose=False)[0]
    preds = list(zip(res.boxes.cls.int().tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist()))

    # Greedy matching, each box used at most once: same-class pairs first, then highest IoU
    pairs = [(g[0] == p[0], iou(g[1], p[2]), gi, pi)
             for gi, g in enumerate(gt) for pi, p in enumerate(preds)]
    pairs = sorted((pair for pair in pairs if pair[1] >= args.iou), reverse=True)
    used_gt, used_pred, matches = set(), set(), []
    for _, overlap, gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, overlap))

    errors = []  # (type, gt index or None, pred index or None, iou)
    for gi, pi, overlap in matches:
        if gt[gi][0] == preds[pi][0]:
            tp[names[gt[gi][0]]] += 1
        else:
            errors.append(("CONFUSION", gi, pi, overlap))
    errors += [("FN", gi, None, 0.0) for gi in range(len(gt)) if gi not in used_gt]
    errors += [("FP", None, pi, 0.0) for pi in range(len(preds)) if pi not in used_pred]

    for n, (kind, gi, pi, overlap) in enumerate(errors):
        gt_box = gt[gi][1] if gi is not None else None
        pred_box = preds[pi][2] if pi is not None else None
        ref = gt_box or pred_box
        crop_name = f"{img_path.stem[:20]}_{n:02d}_{kind}.jpg"
        save_crop(image, gt_box, pred_box, out_dir / "crops" / crop_name)
        rows.append({
            "frame_idx": frame_idx(img_path),
            "type": kind,
            "true_class": names[gt[gi][0]] if gi is not None else "",
            "pred_class": names[preds[pi][0]] if pi is not None else "",
            "conf": round(preds[pi][1], 3) if pi is not None else "",
            "iou": round(overlap, 2) if kind == "CONFUSION" else "",
            "box_w_px": round(ref[2] - ref[0]),
            "box_h_px": round(ref[3] - ref[1]),
            "crop": f"crops/{crop_name}",
            "cause": "",
        })

with open(out_dir / "errors.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["frame_idx", "type", "true_class", "pred_class", "conf",
                                           "iou", "box_w_px", "box_h_px", "crop", "cause"])
    writer.writeheader()
    writer.writerows(rows)

# ---- Summary per class ----
fp = Counter(r["pred_class"] for r in rows if r["type"] == "FP")
fn = Counter(r["true_class"] for r in rows if r["type"] == "FN")
print(f"{len(images)} images in {args.split}, conf >= {args.conf}, IoU >= {args.iou}\n")
print(f"{'class':<14}{'TP':>6}{'FP':>6}{'FN':>6}")
for cname in names.values():
    print(f"{cname:<14}{tp[cname]:>6}{fp[cname]:>6}{fn[cname]:>6}")
mix = Counter((r["true_class"], r["pred_class"]) for r in rows if r["type"] == "CONFUSION")
for (true, pred), n in mix.most_common():
    print(f"confusion: true {true} -> predicted {pred}: {n}x")
sizes = [r["box_w_px"] for r in rows if r["type"] == "FN" and r["true_class"] == "ball"]
if sizes:
    print(f"missed balls: median width {np.median(sizes):.0f} px")
print(f"\nWrote {out_dir / 'errors.csv'} and {len(rows)} crops")
