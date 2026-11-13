"""All GPU jobs in one command; results in gpu/results/<timestamp>/ with a Dutch SUMMARY.md.

    python kobe\\match_analyse\\gpu\\run_all.py                       (full run, GPU 0)
    python kobe\\match_analyse\\gpu\\run_all.py --quick --device cpu  (smoke test, a few minutes, no GPU needed)

Order: check_env -> leakage overview -> PR curves (COCO, best.pt) -> thresholds -> NMS sweep (+ pipeline)
       -> ball fine-tune -> PR curves of the fine-tuned model -> thresholds again -> pipeline with the fine-tune
       -> SUMMARY.md
A failing step is logged (log.txt, SUMMARY.md) and the next steps still run; only a broken environment stops early.

Resumable: the state of every step is kept in <results>/state.json. Running the same command again after an
interruption (or after a failed step) continues the newest unfinished run with the same settings and skips the steps
that already finished; an interrupted fine-tune continues from its last epoch. --force runs every step again,
--new always starts a new results folder.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpu.common import (BEST_TRAIN_DATA, DEFAULT_TEST_DATA, DEFAULT_TRAIN_DATA, FT_WEIGHTS, RESULTS_DIR,  # noqa: E402
                        Ctx, JobError, dataset_overlap, default_device, find_resumable, load_state, log,
                        new_results_dir, save_state, step_done)


class Tee:
    """Write to the console and to log.txt at the same time."""

    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.stream, name)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="rooktest: imgsz 320, 1 epoch, weinig frames")
    p.add_argument("--device", default=None, help="0 = eerste GPU (standaard als CUDA er is), cpu = processor")
    p.add_argument("--test-data", default=str(DEFAULT_TEST_DATA), help="data.yaml van de testset")
    p.add_argument("--test-split", default="test")
    p.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA), help="data.yaml voor de fine-tune (train + val)")
    p.add_argument("--strict-leakage", action="store_true",
                   help="fine-tune weigeren als train- en testbeelden overlappen (standaard: alleen waarschuwen)")
    p.add_argument("--skip-finetune", action="store_true",
                   help="geen fine-tune (gebruikt models/best_ball_ft.pt als die er is)")
    p.add_argument("--imgsz", type=int, default=1280, help="imgsz van de fine-tune")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=4, help="batch van de fine-tune (bij geheugentekort automatisch kleiner)")
    p.add_argument("--cls-pw", type=float, default=0.5, help="klassengewicht-macht 0..1 (0 = uit)")
    p.add_argument("--out", default=None, help="resultatenmap (standaard: onderbroken run verderzetten, anders nieuw)")
    p.add_argument("--new", action="store_true", help="altijd een nieuwe resultatenmap (niet verderzetten)")
    p.add_argument("--force", action="store_true", help="ook stappen opnieuw draaien die al klaar waren")
    return p.parse_args(argv)


def settings_of(args):
    """The arguments that define a run: only an unfinished run with the same settings is continued."""
    return {"quick": args.quick, "test_data": str(Path(args.test_data).resolve()), "test_split": args.test_split,
            "train_data": str(Path(args.train_data).resolve()), "skip_finetune": args.skip_finetune,
            "imgsz": args.imgsz, "epochs": args.epochs, "cls_pw": args.cls_pw}


def choose_out(args, settings):
    """(results folder, resumed?) : --out, else the newest unfinished run with these settings, else a new one."""
    if args.out:
        return Path(args.out), False
    if not args.new and not args.force:
        found = find_resumable(RESULTS_DIR, settings)
        if found is not None:
            return found, True
    return new_results_dir(quick=args.quick), False


def main(argv=None):
    args = parse_args(argv)
    settings = settings_of(args)
    out, resumed = choose_out(args, settings)
    out.mkdir(parents=True, exist_ok=True)
    logfile = open(out / "log.txt", "a", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.stdout, logfile), Tee(sys.stderr, logfile)

    from gpu import check_env, finetune_ball, nms_sweep, pr_curves, thresholds
    from gpu.summary import write_summary

    ctx = Ctx(out=out, device=args.device or default_device(), quick=args.quick, test_data=Path(args.test_data),
              test_split=args.test_split)
    old = {} if args.force else (load_state(out) or {})
    ctx.info = old.get("info", {})
    state = {"settings": settings, "finished": False, "steps": old.get("steps", {}), "info": ctx.info}
    # the fine-tuned model only takes part when this run makes it (or with --skip-finetune when it already exists)
    ft = ctx.info.get("finetune", {})
    if ft.get("status") == "klaar" and Path(ft.get("weights", "")).exists():
        ctx.ft = Path(ft["weights"])
    elif args.skip_finetune and FT_WEIGHTS.exists() and not args.quick:
        ctx.ft = FT_WEIGHTS
    else:
        ctx.ft = out / "geen_finetune.pt"
    t_start, steps = time.time(), []
    log(f"resultaten -> {out}" + ("  (onderbroken run wordt verdergezet; --new = nieuwe map)" if resumed else ""))

    def step(name, fn, fatal=False, always=False):
        if not always and step_done(state, name, args.force):
            steps.append({**state["steps"][name], "opmerking": "al klaar in een eerdere poging (overgeslagen)"})
            log(f"=== {name}: al klaar, overgeslagen (--force = opnieuw) ===")
            return True
        log(f"=== {name} ===")
        t0 = time.time()
        try:
            fn()
            row = {"stap": name, "status": "ok", "minuten": (time.time() - t0) / 60, "opmerking": ""}
            log(f"=== {name}: ok ({time.time() - t0:.0f} s) ===")
        except Exception as e:  # noqa: BLE001 - every step is isolated, the summary reports the failure
            msg = str(e) if isinstance(e, JobError) else f"{type(e).__name__}: {e}"
            if not isinstance(e, JobError):
                traceback.print_exc()
            row = {"stap": name, "status": "MISLUKT", "minuten": (time.time() - t0) / 60,
                   "opmerking": msg.splitlines()[0][:300] if msg else type(e).__name__}
            log(f"=== {name}: MISLUKT: {msg}")
        steps.append(row)
        state["steps"][name] = row
        save_state(out, state)
        if row["status"] != "ok" and fatal:
            write_summary(ctx, steps, t_start)
            log(f"gestopt. Zie {out / 'SUMMARY.md'}")
            sys.exit(1)
        return row["status"] == "ok"

    def env():
        ctx.info["env"] = check_env.check(ctx, need_train=not args.skip_finetune, train_data=Path(args.train_data))
        check_env.ensure_coco(ctx)

    def leakage():
        lk = ctx.info.setdefault("leakage", {})
        lk["best_train"] = dataset_overlap(BEST_TRAIN_DATA, ctx.test_data, ctx.test_split, train_splits=("train",))
        b = lk["best_train"]
        if b["n_overlap"]:
            banner = "!" * 100
            log(f"\n{banner}\n  LET OP LEAKAGE: {b['n_overlap']} van de {b['n_test']} testbeelden zitten in de "
                f"trainingsdata van best.pt.\n  De testcijfers van best.pt zijn dus te optimistisch "
                f"(zie SUMMARY.md).\n{banner}")
        if not args.skip_finetune:
            lk["finetune_train"] = dataset_overlap(Path(args.train_data), ctx.test_data, ctx.test_split)

    def finetune():
        finetune_ball.train(ctx, Path(args.train_data), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                            cls_pw=args.cls_pw, strict_leakage=args.strict_leakage, resume=not args.force)

    ft_labels = ["ball-finetune", f"ball-finetune@{args.imgsz}", f"best.pt@{args.imgsz}"]
    step("1. omgeving controleren", env, fatal=True, always=True)
    step("2. leakage-overzicht", leakage)
    step("3. PR-curves COCO + best.pt", lambda: pr_curves.run(ctx, only=["COCO yolo26l", "best.pt"]))
    step("4. drempels uit F1-curve", lambda: thresholds.run(ctx))
    step("5. NMS-sweep + pijplijn", lambda: nms_sweep.run_job(ctx))
    if not args.skip_finetune:
        step("6. fine-tune bal", finetune)
    if Path(ctx.ft).exists():
        step("7. PR-curves fine-tune", lambda: pr_curves.run(ctx, only=ft_labels))
        step("8. drempels (met fine-tune)", lambda: thresholds.run(ctx))
        step("9. pijplijn met fine-tune", lambda: nms_sweep.run_finetuned(ctx))
    failed = [s["stap"] for s in steps if s["status"] != "ok"]
    state["finished"] = not failed            # a failed step: the same command retries only the failed steps
    save_state(out, state)
    path = write_summary(ctx, steps, t_start)
    log(f"klaar in {(time.time() - t_start) / 60:.1f} min. Samenvatting: {path}")
    if failed:
        log(f"MISLUKTE STAPPEN: {', '.join(failed)} (zie log.txt). Hetzelfde commando nog eens draaien probeert "
            f"alleen die stappen opnieuw.")
    log(f"Terugsturen: zip de map {out} (zie 'Terugsturen' onderaan SUMMARY.md)")
    logfile.flush()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
