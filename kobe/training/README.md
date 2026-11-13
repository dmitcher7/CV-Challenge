# Training pipeline: Roboflow export → dataset → ablations → final model → error analysis

All scripts run from the repo root, e.g. `venv\Scripts\python kobe\training\prepare_dataset.py`.
Shared paths, the split rule and the class merge live in `config.py`. Nothing in `data/` is ever modified.

## Workflow (in this order)

| # | Step | Command | Where |
|---|---|---|---|
| 1 | Check environment | `python kobe/training/check_env.py` | local |
| 2 | What to label next | `python kobe/training/plan_labeling.py --copy` | local |
| 3 | Label `to_label/` in Roboflow, peer review, export a **new version without augmentations** into `data/dataset_yolo/` | | Roboflow |
| 4 | Build dataset | `python kobe/training/prepare_dataset.py` | local + GPU |
| 5 | Ablations (4 short runs) | `python kobe/training/ablation.py` | GPU |
| 6 | Final training, once | `python kobe/training/train.py --name final --model <winner> --set key=value ...` | GPU |
| 7 | Final score, once | `python kobe/training/evaluate.py --run final --split test` | GPU or local |
| 8 | Error analysis | `python kobe/training/error_analysis.py --run final --split test` | local |
| 9 | Video | `python kobe/training/predict_video.py --run final --conf <best conf>` | local |

Repeat 2–4 until `plan_labeling.py` says no more frames are needed.

## Design choices (and why)

**Split by time blocks.** The 660 frames (22 s at 30 fps) are cut into 11 blocks of 60 frames:

```
block   0     1     2     3     4     5     6     7     8     9     10
frames  0-59  -119  -179  -239  -299  -359  -419  -479  -539  -599  -659
split   train train VAL   train train TEST  train train VAL   train TEST
```

Neighbouring frames are almost identical, so a random split would leak train frames into val/test.
Train frames within 10 frames of a val/test block are not used (gap). Val and test each come from two
different moments of the clip, instead of only the end. **Val** chooses `best.pt` and compares runs;
**test** is used only once, for the final model, so its score is not biased by those choices.

**Duplicates.** The clip was converted from 25 to 30 fps: every 6th frame is a copy of the previous one,
plus a few frozen frames (127 in total). They are never proposed for labeling, and a labeled duplicate
of a labeled frame is dropped from the dataset.

**Classes: `ball`, `player`, `goalkeeper`, `referee`.** `player_A`/`player_B` are merged
(`MERGE` in `config.py`): the detector only finds players, the team is assigned afterwards by jersey colour
(K-means, Extra Mile "Tactical Analytics"). In the smoke test the detector confused the two teams a lot,
and merging doubles the examples per class. `prepare_dataset.py --keep-teams` keeps the old 5 classes.

**Labeling order.** `plan_labeling.py` first fills val and test (25 instances per class), then train
(100 per class). Within a split it picks frames as far apart in time as possible (farthest-point sampling),
because neighbouring frames add almost no new information. Targets can be changed with `--val/--test/--train`.

**Training.** Defaults on a GPU: `yolo26m.pt`, `imgsz=1280`, up to 150 epochs with early stopping
(`patience=30`), batch as large as memory allows, cosine LR, mosaic off for the last 15 epochs. At 640
the ball (≈16 px in 1920×1080) shrinks to ≈5 px, too small for YOLO's finest feature map; at 1280 it is ≈11 px.
Every run gets its own folder (`exist_ok=False`), so a run can never overwrite an earlier one.

**Ablations** (`ablation.py`, YOLO26s, 40 epochs each) change one thing at a time:
640 vs 1280, default vs stronger zoom/shift augmentation, s vs m. The results in `experiments.csv`
justify the settings of the final run.

## Running on Kaggle (free GPU)

This laptop has no CUDA GPU (Intel Arc), so steps 5–7 run on Kaggle: *New Notebook → Settings →
Accelerator: GPU T4 x2, Internet on*.

```python
!git clone https://github.com/dmitcher7/CV-Challenge.git   # private repo: use a token, or upload it as a Kaggle dataset
%cd CV-Challenge
!git checkout kv-first-model
!pip install -q ultralytics
!python kobe/training/prepare_dataset.py
!python kobe/training/ablation.py
```

`data/frames/` and the Roboflow export are in git, so the clone has everything except the video
(only needed for `predict_video.py`). Download `kobe/training/runs/` and `experiments.csv` before the session ends
(*Save Version* keeps `/kaggle/working`).

## Output files

- `experiments.csv`: one row per evaluation (run, split, model, imgsz, epochs, P, R, mAP50, mAP50-95, ball mAP, note). Committed.
- `runs/<name>/`: ultralytics training output: `results.png` (loss and mAP curves), confusion matrix,
  `val_batch*_labels/pred.jpg`, `weights/best.pt`.
- `runs/<name>_<split>/`: evaluation plots. `evaluate.py` also prints the **best confidence per class**
  (highest F1); use it as `--conf` for the video and the dashboard.
- `runs/<name>_errors_<split>/errors.csv` + `crops/`: every FP, FN and class mix-up with a zoomed crop
  (green = ground truth, red = prediction). Fill in the `cause` column (blur, occlusion, advertising board,
  too small, label error ...): that is the input for the error analysis in the report.
- `label_candidates.csv`: ranked frames to label per split; `needed = 1` for the ones the estimate asks for.

`dataset/`, `to_label/`, `pretrained/` and `runs/` are generated and not committed.

## History

### 2026-10-19: smoke test (milestone 0)

YOLO26s, imgsz 640, 10 epochs, CPU, 5 classes, old split (train idx 0–494, val 525–659 = 4 images).
Plots in `history/2026-10-19_smoke/`.

| class | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| ball | 0.000 | 0.000 | 0.000 | 0.000 |
| player_A | 0.767 | 0.875 | 0.863 | 0.399 |
| player_B | 0.748 | 0.575 | 0.667 | 0.215 |
| player_B_goal_keeper | 1.000 | 0.000 | 0.781 | 0.263 |
| referee | 1.000 | 0.000 | 0.016 | 0.002 |
| **all** | 0.703 | 0.290 | 0.466 | 0.176 |

Findings, and what changed because of them:

- **The ball was never detected** (not in val, not in any of the 579 unlabeled frames). At imgsz 640 it is ≈5 px,
  and there were only 42 training instances. → imgsz 1280 on a GPU, ball-driven labeling targets.
- **Many player_A ↔ player_B mix-ups** and false positives on the advertising boards. → teams merged,
  team assigned by jersey colour; boards to be checked in `error_analysis.py`.
- **Goalkeeper/referee P = 1.0, R = 0** rests on 4 val instances each: not meaningful. → val and test of
  25+ instances per class, from two moments in the clip.
- **Train set less diverse than it looks:** consecutive frames and 127 duplicate frames. → duplicates removed,
  frames picked spread out in time.
- A later 2-epoch run overwrote the smoke-test weights (same run name). → unique run names, `exist_ok=False`.
