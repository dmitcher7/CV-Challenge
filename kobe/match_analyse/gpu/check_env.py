"""Job 1: check the environment before anything heavy runs (torch, CUDA, GPU, ultralytics, files). Writes env.json.

    python kobe\\match_analyse\\gpu\\check_env.py            (standalone: prints the report)
"""

import argparse
import json
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpu.common import (COCO_WEIGHTS, DEFAULT_TRAIN_DATA, EXPECTED_ULTRALYTICS, GT_BODY, GT_FEET,  # noqa: E402
                        JobError, ROOT, Ctx, log, split_dirs)
from core.paths import KEYPOINTS_JSON  # noqa: E402


def check(ctx: Ctx, need_train=True, train_data=DEFAULT_TRAIN_DATA):
    """Return the env dict; raise JobError (Dutch) when something essential is missing."""
    problems, warnings = [], []
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
           "device_requested": ctx.device, "quick": ctx.quick}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            env["gpu"] = p.name
            env["vram_gb"] = round(p.total_memory / 2**30, 1)
            if env["vram_gb"] < 7.5:
                warnings.append(f"GPU heeft maar {env['vram_gb']} GB VRAM; de fine-tune verkleint batch/imgsz vanzelf.")
    except ImportError:
        problems.append("PyTorch is niet geinstalleerd. Zie gpu/README.md, stap 'Installatie'.")
        env["torch"] = None
    try:
        import ultralytics
        env["ultralytics"] = ultralytics.__version__
        if ultralytics.__version__ != EXPECTED_ULTRALYTICS:
            warnings.append(f"ultralytics {ultralytics.__version__} gevonden, verwacht {EXPECTED_ULTRALYTICS}. "
                            f"Installeer: pip install ultralytics=={EXPECTED_ULTRALYTICS}")
    except ImportError:
        problems.append(f"ultralytics ontbreekt: pip install ultralytics=={EXPECTED_ULTRALYTICS}")
    for mod in ("cv2", "scipy", "matplotlib", "pandas", "yaml"):
        try:
            __import__(mod)
        except ImportError:
            problems.append(f"Python-pakket '{mod}' ontbreekt: pip install -r kobe/match_analyse/gpu/requirements-gpu.txt")

    if ctx.device != "cpu" and not env.get("cuda_available"):
        problems.append(f"--device {ctx.device} gevraagd, maar PyTorch ziet geen CUDA-GPU. Waarschijnlijk is de "
                        "CPU-versie van torch geinstalleerd: zie gpu/README.md (Problemen: 'CUDA niet gevonden'). "
                        "Rooktest zonder GPU: --quick --device cpu")
    if ctx.device == "cpu" and not ctx.quick:
        warnings.append("Volledige run op de CPU: de fine-tune duurt dan vele uren. Bedoeld is een GPU.")

    files = {"best.pt (ons model)": ctx.best, "video": ctx.video, "keypoints.json": KEYPOINTS_JSON,
             "ground_truth.json": GT_BODY, "ground_truth_feet.json": GT_FEET}
    for what, p in files.items():
        if not Path(p).exists():
            problems.append(f"Bestand ontbreekt ({what}): {p}")
    if not Path(ctx.coco).exists():
        warnings.append(f"COCO-model ontbreekt ({ctx.coco}); ultralytics downloadt yolo26l.pt (internet nodig).")
    elif env.get("torch"):
        names = model_names(ctx.coco)
        env["coco_classes"] = len(names)
        if len(names) != 80 or names.get(0) != "person" or names.get(32) != "sports ball":
            problems.append(f"{ctx.coco} is geen COCO-model met 80 klassen (gevonden: {len(names)} klassen). "
                            "Zet de originele yolo26l.pt (COCO) op die plek.")
    if Path(ctx.best).exists() and env.get("torch"):
        names = model_names(ctx.best)
        env["best_classes"] = [names[k] for k in sorted(names)]
        if "ball" not in names.values():
            problems.append(f"{ctx.best} heeft geen klasse 'ball': {env['best_classes']}")
    try:
        env["test_images"] = sum(len(list(d.iterdir())) for d in split_dirs(ctx.test_data, ctx.test_split))
    except JobError as e:
        problems.append(f"Testset: {e}")
    if need_train:
        try:
            split_dirs(train_data, "train")
            split_dirs(train_data, "val")
        except JobError as e:
            problems.append(f"Fine-tune-data: {e}")
    env["files"] = {k: str(v) for k, v in files.items()}
    env["test_data"], env["test_split"] = str(ctx.test_data), ctx.test_split
    env["root"] = str(ROOT)
    env["warnings"], env["problems"] = warnings, problems
    ctx.save_json("env.json", env)
    for w in warnings:
        log(f"LET OP: {w}")
    if problems:
        raise JobError("Omgeving niet in orde:\n  - " + "\n  - ".join(problems))
    log(f"omgeving ok: torch {env.get('torch')}, CUDA {env.get('cuda_available')}, GPU {env.get('gpu', '-')}, "
        f"{env.get('vram_gb', '-')} GB, ultralytics {env.get('ultralytics')}")
    return env


def model_names(weights):
    """{class id: name} stored in a .pt checkpoint (without building the model)."""
    import torch
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    model = ckpt.get("ema") or ckpt.get("model")
    names = model.names
    return dict(enumerate(names)) if isinstance(names, (list, tuple)) else {int(k): v for k, v in names.items()}


def ensure_coco(ctx: Ctx):
    """Path of the COCO baseline; downloads yolo26l.pt via ultralytics when the local copy is missing."""
    if Path(ctx.coco).exists():
        return Path(ctx.coco)
    from ultralytics import YOLO
    log("COCO-model downloaden (yolo26l.pt) ...")
    model = YOLO("yolo26l.pt")
    src = Path(model.ckpt_path)
    COCO_WEIGHTS.parent.mkdir(exist_ok=True)
    import shutil
    shutil.copy2(src, COCO_WEIGHTS)
    ctx.coco = COCO_WEIGHTS
    return COCO_WEIGHTS


def main():
    from gpu.common import common_args, ctx_from_args
    args = common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    ctx = ctx_from_args(args)
    try:
        env = check(ctx)
    except JobError as e:
        sys.exit(str(e))
    print(json.dumps(env, indent=1))


if __name__ == "__main__":
    main()
