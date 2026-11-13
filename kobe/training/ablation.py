"""Short comparison runs to choose the settings of the final training (needs a GPU).

Each experiment changes one thing compared to the baseline, trains briefly
and is evaluated on val; the results end up in experiments.csv. Runs that
already exist are skipped, so the script can be restarted after a crash.

Afterwards: train the final model once with train.py, using the winning settings.
"""

from config import RUNS_DIR, pick_device
from evaluate import evaluate
from train import train

EPOCHS, PATIENCE = 40, 15  # short runs: enough to see differences, not to converge fully

# name -> (model, imgsz, extra ultralytics arguments, why)
EXPERIMENTS = {
    "abl_s_640":      ("yolo26s.pt", 640,  {}, "baseline, ball ~5 px after downscaling"),
    "abl_s_1280":     ("yolo26s.pt", 1280, {}, "higher resolution, ball ~11 px"),
    "abl_s_1280_aug": ("yolo26s.pt", 1280, {"scale": 0.9, "translate": 0.2},
                       "stronger zoom/shift augmentation: ball at more sizes and positions"),
    "abl_m_1280":     ("yolo26m.pt", 1280, {}, "bigger model"),
}

device = pick_device()
if device == "cpu":
    raise SystemExit("No GPU found: run this on Kaggle/Colab (see README).")

for name, (model, imgsz, overrides, why) in EXPERIMENTS.items():
    if (RUNS_DIR / name).exists():
        print(f"skip {name}: already trained")
        continue
    train(name, model, imgsz, EPOCHS, -1, PATIENCE, device, **overrides)
    evaluate(name, "val", note=f"ablation: {why} {overrides or ''}".strip())
