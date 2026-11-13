"""Job 5: fine-tune best.pt for the ball (higher resolution + class weighting), on a GPU.

    python kobe\\match_analyse\\gpu\\finetune_ball.py [--data datasets/ball_finetune/data.yaml] [--imgsz 1280]

Starts from models/best.pt and trains on --data (default datasets/ball_finetune: the 100 Roboflow frames split by
time block, train = frames < 200 and 246..599, val = 200..245; the notebook test frames >= 600 are not in it).
Settings: imgsz 1280 (the ball is ~5-10 px at 960), 60 epochs, patience 15, seed 0, deterministic, cos_lr,
close_mosaic 10, cls_pw 0.5 (see below). Best weights -> models/best_ball_ft.pt (quick mode: only in the results).

Class weighting: ultralytics 8.4.182 has `cls_pw` (cfg/default.yaml; models/yolo/detect/train.py
set_class_weights): class weights = (1 / class frequency in the train labels) ** cls_pw, normalised to mean 1,
multiplied into the BCE classification loss (utils/loss.py v8DetectionLoss). 0 = off, 1 = full inverse frequency.
The ball is ~1 label per image against ~20 persons, so cls_pw 0.5 gives the ball a few times more weight.

CUDA out of memory: ultralytics itself halves the batch up to 3 times in the first epoch; if training still runs
out of memory this script retries with a smaller batch / imgsz (ATTEMPTS).

Leakage check: counts the test images that also occur in --data (train or val; same frame number or same file
content) and warns loudly; the count always goes into SUMMARY.md. Only with --strict-leakage it refuses to train.
(Owner's decision, option B: no new labels yet, so overlap with the notebook test split is accepted and reported.)

Resume: when a run was interrupted, finetune/train_*/weights/last.pt exists and training continues from there
(ultralytics resume=True); if that is not possible it starts again from best.pt.
"""

import argparse
import gc
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpu.common import (DEFAULT_TRAIN_DATA, FT_WEIGHTS, Ctx, JobError, dataset_overlap, log,  # noqa: E402
                        resolved_yaml)

EPOCHS, PATIENCE, IMGSZ, BATCH, CLS_PW = 60, 15, 1280, 4, 0.5


def attempts(imgsz, batch):
    """(imgsz, batch) combinations to try in order when CUDA runs out of memory."""
    out = []
    for a in [(imgsz, batch), (imgsz, max(1, batch // 2)), (min(imgsz, 1024), 2), (min(imgsz, 960), 2),
              (min(imgsz, 960), 1)]:
        if a not in out:
            out.append(a)
    return out


class LeakageError(JobError):
    pass


def check_leakage(train_yaml, test_yaml, test_split, strict=False):
    """Overlap between the fine-tune data and the test set; raises LeakageError only when strict."""
    ov = dataset_overlap(train_yaml, test_yaml, test_split)
    log(f"lekcontrole: {ov['n_overlap']}/{ov['n_test']} testbeelden ook in de fine-tune-data "
        f"(zelfde framenummer {ov['by_frame']}, zelfde bestand {ov['by_hash']})")
    if ov["n_overlap"]:
        frames = f"{ov['frames'][:20]}{' ...' if len(ov['frames']) > 20 else ''}"
        if strict:
            raise LeakageError(
                f"GEWEIGERD (--strict-leakage): {ov['n_overlap']} van de {ov['n_test']} testbeelden zitten ook in de "
                f"trainings-/valdata ({train_yaml}). Kies andere data (--train-data / --test-data) of laat "
                f"--strict-leakage weg. Overlappende frames: {frames}")
        banner = "!" * 100
        log(f"\n{banner}\n  LET OP LEAKAGE: {ov['n_overlap']} van de {ov['n_test']} testbeelden zitten ook in de "
            f"fine-tune-data. De testcijfers van de fine-tune zijn dan te optimistisch (staat in SUMMARY.md).\n"
            f"  Frames: {frames}\n{banner}")
    return ov


def is_oom(e):
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def train(ctx: Ctx, data=DEFAULT_TRAIN_DATA, imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH, cls_pw=CLS_PW,
          strict_leakage=False, patience=PATIENCE, resume=True):
    """Fine-tune; returns the info dict (also in ctx.info['finetune']). Sets ctx.ft to the new weights."""
    from ultralytics import YOLO
    out = ctx.sub("finetune")
    info = {"data": str(data), "status": "gestart"}
    ctx.info["finetune"] = info
    info["overlap"] = check_leakage(data, ctx.test_data, ctx.test_split, strict_leakage)
    info["strict_leakage"] = strict_leakage
    data_yaml = resolved_yaml(data, out / "data_train.yaml", splits=("train", "val"))

    if ctx.quick:
        imgsz, epochs, batch, patience = 320, 1, 2, 1
    common = dict(data=str(data_yaml), epochs=epochs, patience=patience, seed=0, deterministic=True, cos_lr=True,
                  close_mosaic=min(10, epochs), cls_pw=cls_pw, device=ctx.device, project=str(out), exist_ok=True,
                  plots=not ctx.quick, workers=0 if ctx.quick else 2, verbose=False)
    # quick: no plots (a 1-epoch model predicts degenerate boxes and the ultralytics plot thread prints errors)
    if ctx.quick:
        common["fraction"] = 0.25
    info.update({"epochs": epochs, "patience": patience, "cls_pw": cls_pw, "seed": 0})
    t0, last = time.time(), None
    trainer = _resume(out, ctx.device) if resume else None
    if trainer is not None:
        info.update(_trainer_info(trainer, resumed=True))
    for n, (sz, bs) in enumerate([] if trainer is not None else attempts(imgsz, batch)):
        try:
            import torch
            gc.collect()        # frees the trainer of a previous (OOM) attempt before the cache is emptied
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log(f"fine-tune poging {n + 1}: imgsz {sz}, batch {bs}, {epochs} epochs ({ctx.device}) ...")
            model = YOLO(str(ctx.best))
            model.train(imgsz=sz, batch=bs, name=f"train_{sz}_b{bs}", **common)
            info.update(_trainer_info(model.trainer))
            last = None
            break
        except Exception as e:  # noqa: BLE001 - only CUDA OOM is retried, the rest is raised
            if not is_oom(e):
                raise
            last = str(e)       # not the exception itself: its traceback keeps the trainer (GPU memory) alive
            model = None
            log(f"CUDA te weinig geheugen bij imgsz {sz}, batch {bs}: kleiner proberen")
    if last is not None:
        raise JobError(f"Fine-tune lukt niet, ook niet met de kleinste instelling (te weinig GPU-geheugen): {last}")

    best = Path(info["save_dir"]) / "weights" / "best.pt"
    if not best.exists():
        best = Path(info["save_dir"]) / "weights" / "last.pt"
    target = out / "quick_best_ball_ft.pt" if ctx.quick else FT_WEIGHTS
    shutil.copy2(best, target)
    if not ctx.quick:                       # also in the results: the teammate only has to send that one folder
        (ctx.out / "model").mkdir(exist_ok=True)
        shutil.copy2(best, ctx.out / "model" / FT_WEIGHTS.name)
    ctx.ft = target
    info.update({"weights": str(target), "seconds": round(time.time() - t0, 1), "status": "klaar"})
    ctx.save_json("finetune.json", info)
    log(f"fine-tune klaar in {info['seconds']:.0f} s -> {target}")
    return info


def _trainer_info(trainer, resumed=False):
    return {"imgsz": int(trainer.args.imgsz), "batch": int(trainer.batch_size), "save_dir": str(trainer.save_dir),
            "epochs_run": int(trainer.epoch) + 1, "best_epoch": _best_epoch(trainer), "resumed": resumed,
            "val_metrics": {k: float(v) for k, v in (trainer.metrics or {}).items()}}


def is_resumable(path):
    """True when a checkpoint still holds the epoch and optimizer state of an unfinished training."""
    import torch
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    return isinstance(ckpt, dict) and ckpt.get("epoch", -1) >= 0 and ckpt.get("optimizer") is not None


def _resume(out, device):
    """Continue an interrupted training (newest finetune/train_*/weights/last.pt); returns the trainer or None."""
    from ultralytics import YOLO
    lasts = sorted(Path(out).glob("train_*/weights/last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not lasts:
        return None
    try:
        if not is_resumable(lasts[0]):   # ultralytics would silently start a default coco8 training instead
            raise ValueError("training was al klaar (geen optimizer-toestand in last.pt)")
        log(f"onderbroken fine-tune gevonden, verder vanaf {lasts[0]} ...")
        model = YOLO(str(lasts[0]))
        model.train(resume=True, device=device)
        return model.trainer
    except Exception as e:  # noqa: BLE001 - finished or broken checkpoint: start a fresh training instead
        log(f"verdergaan lukt niet ({type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}); "
            f"opnieuw starten vanaf best.pt")
        for d in {p.parent.parent for p in lasts}:
            shutil.rmtree(d, ignore_errors=True)
        return None


def _best_epoch(trainer):
    """Epoch (1-based) with the best fitness according to results.csv, or None."""
    import csv
    path = Path(trainer.save_dir) / "results.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    key = next((k for k in (rows[0] if rows else {}) if k.strip() == "metrics/mAP50-95(B)"), None)
    if not key:
        return None
    best = max(rows, key=lambda r: float(r[key] or 0))
    return int(float(best["epoch"]))


def main():
    from gpu.common import common_args, ctx_from_args
    parser = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--data", default=str(DEFAULT_TRAIN_DATA), help="data.yaml om op te trainen (train + val)")
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--cls-pw", type=float, default=CLS_PW, help="klassengewicht-macht 0..1 (0 = uit)")
    parser.add_argument("--strict-leakage", action="store_true", help="weigeren als train en test overlappen")
    args = parser.parse_args()
    ctx = ctx_from_args(args)
    try:
        train(ctx, Path(args.data), args.imgsz, args.epochs, args.batch, args.cls_pw, args.strict_leakage)
    except JobError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
