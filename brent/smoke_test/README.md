# Smoke test: Roboflow export → YOLO train → eval → video prediction

Goal: prove that the full chain works end to end. **Model quality does not matter yet.**

## Setup (run from `brent/`)

Create the local environment and install all imports once:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

The local `requirements.txt` supports Python 3.13, installs the CUDA 13.0 builds of PyTorch for
the NVIDIA GPU, and does not use the repository-wide `numpy<2.0` constraint, for which Python
3.13 has no compatible wheel.

This machine has an NVIDIA GeForce RTX 4070 Laptop GPU. `pick_device()` selects CUDA device 0,
so training and inference run on the **GPU**.

## Scripts (run from `brent/`)

| Step | Command | Output |
|---|---|---|
| 1. Environment | `.\.venv\Scripts\python smoke_test\check_env.py` | versions + CUDA/GPU info |
| 2. Split | `.\.venv\Scripts\python smoke_test\make_split.py` | `train.txt`, `val.txt`, `data_smoke.yaml` |
| 3. Train | `.\.venv\Scripts\python smoke_test\train_smoke.py` | `runs/smoke/` |
| 4. Evaluate | `.\.venv\Scripts\python smoke_test\eval_smoke.py` | table + `runs/smoke_val/` + row in `experiments.csv` |
| 5. Video | `.\.venv\Scripts\python smoke_test\predict_video.py` | `runs/smoke_predict/match.avi` (Windows writes `.avi`) |

`train_smoke.py` accepts `--model --imgsz --epochs --batch` (defaults: CPU 640/10/4, GPU 1280/50/8;
model `yolo26s.pt` when ultralytics ≥ 8.4, else `yolo11s.pt`). `eval_smoke.py` and `predict_video.py`
read `imgsz` from `runs/smoke/args.yaml`, so they always use the training resolution.

`common.py` holds the shared paths. It also turns off ultralytics' `labels.cache`, which would otherwise
be written into `data/dataset_yolo/train/` (we never modify `data/`).

### Split rule (time based, avoids leakage between neighbouring frames)

train = frame idx 0–494 · gap = 495–524 (never used) · val = 525–659

Result on the current export:

| class | train | val |
|---|---:|---:|
| ball | 42 | 4 |
| player_A | 469 | 40 |
| player_B | 468 | 40 |
| player_B_goal_keeper | 45 | 4 |
| referee | 174 | 4 |
| **images** | **47** | **4** |

No images without labels, no frames in the gap.

## Output files explained

- `runs/smoke/results.png`: training curves per epoch: box/cls/dfl loss (train and val) and
  precision, recall, mAP50, mAP50-95 on val. Losses should go down; if the val loss goes up while the
  train loss goes down, the model is overfitting.
- `runs/smoke/confusion_matrix.png` (+ `_normalized`): rows = predicted class, columns = true class.
  The diagonal is correct. The `background` row/column shows missed objects (FN) and false detections (FP).
  Mix-ups such as player_B ↔ player_B_goal_keeper show up off the diagonal.
- `runs/smoke/val_batch*_labels.jpg` vs `val_batch*_pred.jpg`: the same val images with the
  ground-truth boxes and the predicted boxes (with confidence). Compare them visually.
- `runs/smoke/weights/best.pt` / `last.pt`: the best checkpoint (by fitness ≈ mAP50-95 on val) and the last epoch.
- `experiments.csv`: one row per evaluation (run, model, imgsz, epochs, P, R, mAP50, mAP50-95).

## Notebook: `smoke_test_labeling.ipynb`

Runs the same chain step by step and then estimates **how much still needs labeling**: label coverage over the clip,
a model pass over all 660 frames (also writes `runs/smoke_predict/match_pred.mp4`), a count-based estimate of extra
frames per split, and `label_candidates.csv` (which frames to label next, referring to `data/frames/`).
An optional learning curve (`RUN_LEARNING_CURVE = True`) shows whether more data still helps.
`ipykernel` is included in `requirements.txt`; select the `.venv` kernel in VS Code.

## CPU speed note

On the Intel Core Ultra 7 155H, torch uses 16 threads by default, including the slow E-cores, and was ~3× slower
than with 6 threads. `common.py` sets 6 threads on CPU. One epoch takes ≈ 2–3.5 min; inference ≈ 1 s per frame.

## Results (2026-10-19, YOLO26s, imgsz 640, 10 epochs, batch 4, CPU, 22.7 min training)

| class | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| ball | 0.000 | 0.000 | 0.000 | 0.000 |
| player_A | 0.767 | 0.875 | 0.863 | 0.399 |
| player_B | 0.748 | 0.575 | 0.667 | 0.215 |
| player_B_goal_keeper | 1.000 | 0.000 | 0.781 | 0.263 |
| referee | 1.000 | 0.000 | 0.016 | 0.002 |
| **all** | 0.703 | 0.290 | 0.466 | 0.176 |

## Interpretation

- **What works:** the full chain runs end to end, and the players are found reasonably well (player_A mAP50 0.86).
- **Ball: never detected**, not in val and in none of the 579 unlabeled video frames. The median box is ≈ 16×17 px
  in the 1920×1080 frame. At `imgsz=640` the image is downscaled 3×, so the ball is ≈ 5 px, too small for
  YOLO's finest (stride-8) feature map. With only 42 training instances on top of that, the model does not learn it.
  Next steps: `imgsz=1280` (needs a GPU in practice), or tiling/SAHI, plus more ball labels.
- **Team confusion:** the confusion matrix shows many player_A ↔ player_B mix-ups and false positives on the
  advertising boards. The colour difference (dark blue vs yellow) is not learned well yet from 47 near-identical frames.
- **Goalkeeper / referee:** P = 1.0, R = 0 means that at the default confidence threshold no box was correct.
  mAP50 0.78 for the goalkeeper comes from low-confidence detections. All of this rests on **4 val instances**
  per class, so these numbers are not meaningful (one hit = 25 %).
- **Validation set is tiny:** only 4 images (idx 525, 529, 533, 537).
- **Train set is less diverse than 47 images suggest:** most frames are consecutive, and the clip contains
  128 exact duplicate frames.

## How much still needs labeling (from the notebook)

With targets of 300 instances per class in train and 50 in val: ≈ **46 more val frames** and ≈ **289 more train
frames**. The rare classes (ball, goalkeeper, referee in val) decide this. For train, that is 80 % of the remaining
non-duplicate frames. Because the model never finds the ball, `label_candidates.csv` is currently a time-spread
list (every 3rd frame), not an uncertainty ranking. Suggested order: first the val frames, then train in rounds
of ~30 frames, retraining in between.
