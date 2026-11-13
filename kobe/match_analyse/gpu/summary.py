"""SUMMARY.md (Dutch) of one run: everything the owner needs for the report, from ctx.info."""

import time
from pathlib import Path

from gpu.common import CACHE_EXPORT, FT_WEIGHTS, Ctx, fmt, md_table

PIPE_COLS = ["model", "setting", "imgsz", "frames", "seconds", "ball_detected", "ball_interpolated", "ball_none",
             "body_ball_found", "body_false_ball", "body_players_f1", "body_role_acc", "body_possession_acc",
             "feet_ball_found", "feet_false_ball", "feet_players_f1", "feet_role_acc", "feet_possession_acc",
             "possession_A_pct", "possession_B_pct"]


def _ball_ap(pr_runs, model, view="classes"):
    """Ball AP50 of a model label in the PR results, or None."""
    for r in pr_runs.values():
        if r["model"] == model and r["view"] == view:
            return next((c["mAP50"] for c in r["rows"] if c["class"] == "ball"), None)
    return None


def _pipe_balls(row):
    """(balls found, false balls) summed over both GT files of one pipeline row, or None."""
    if not row or "_body_found" not in row:
        return None
    return (row.get("_body_found", 0) + row.get("_feet_found", 0), row.get("_body_false", 0) + row.get("_feet_false", 0))


def verdict(ap_best, ap_ft, pipe_best, pipe_ft):
    """("ja" / "nee" / "twijfel" / "onbekend", reasons) : is the fine-tune better than best.pt?

    ap_*: ball AP50 on the test set; pipe_*: (balls found, false balls) of the pipeline on the clip.
    ja = better on both criteria; nee = better on neither; twijfel = mixed."""
    if None in (ap_best, ap_ft) or None in (pipe_best, pipe_ft):
        return "onbekend", ["niet alle cijfers zijn er (stap mislukt of overgeslagen)"]
    ap_ok = ap_ft > ap_best
    pipe_ok = pipe_ft[0] > pipe_best[0] and pipe_ft[1] <= pipe_best[1]
    reasons = [f"bal-AP50 testset: {ap_best:.3f} -> {ap_ft:.3f} ({'beter' if ap_ok else 'niet beter'})",
               f"pijplijn (body + voeten): bal gevonden {pipe_best[0]} -> {pipe_ft[0]}, valse ballen "
               f"{pipe_best[1]} -> {pipe_ft[1]} ({'beter' if pipe_ok else 'niet beter'})"]
    if ap_ok and pipe_ok:
        return "ja", reasons
    if not ap_ok and not pipe_ok:
        return "nee", reasons
    return "twijfel", reasons


def _rel(ctx, p):
    try:
        return Path(p).resolve().relative_to(ctx.out.resolve()).as_posix()
    except ValueError:
        return str(p)


def leakage_block(ctx: Ctx):
    lk = ctx.info.get("leakage", {})
    best, ft = lk.get("best_train"), lk.get("finetune_train")
    L = ["## ⚠️ LET OP: leakage (lees dit eerst)", ""]
    if best is not None:
        bad = best["n_overlap"] > 0
        L += [f"> **{'WAARSCHUWING' if bad else 'OK'}: {best['n_overlap']} van de {best['n_test']} testbeelden "
              f"zitten in de 100 Roboflow-frames waarop `best.pt` getraind is** "
              f"(zelfde framenummer: {best['by_frame']}, zelfde bestand: {best['by_hash']}).", ">"]
        if bad:
            L += ["> De cijfers van `best.pt` (en van de fine-tune, die vertrekt van `best.pt`) op deze testset zijn dus "
                  "**te optimistisch**: het model heeft deze beelden (of hun buren) al gezien. Gebruik ze niet als "
                  "eerlijke generalisatie. Een eerlijke test vraagt nieuwe, gelabelde frames uit 63–199 en 246–524 "
                  "(`gpu/export_unseen_frames.py`, zie `gpu/README.md`). De vergelijking COCO ↔ best.pt blijft wel "
                  "zinvol als illustratie, maar met dit voorbehoud.", ""]
        else:
            L += ["> De testset overlapt niet met de trainingsbeelden van `best.pt`.", ""]
    if ft is not None:
        L += [f"- Overlap fine-tune-data (train + val) ↔ testset: **{ft['n_overlap']} / {ft['n_test']}** beelden"
              + (" — **ook de fine-tune-cijfers op deze testset zijn dan te optimistisch** (bewust toegelaten; "
                 "`--strict-leakage` weigert in dat geval)." if ft["n_overlap"] else ".")]
    L += [f"- Testset: `{ctx.test_data}` (split `{ctx.test_split}`)", ""]
    return L


def verdict_block(ctx: Ctx):
    """'Is best_ball_ft.pt beter dan best.pt?' from the ball AP50 and the pipeline on the whole clip."""
    info = ctx.info
    pipe_ft = info.get("pipeline_ft")
    if not pipe_ft:
        return []
    runs = info.get("pr", {}).get("runs", {})
    pipe_best = {r["imgsz"]: r for r in info.get("nms", {}).get("pipeline", []) + pipe_ft
                 if r["model"].startswith("best.pt") and r["setting"] == "NMS iou=0.7"}
    L = ["## Oordeel: is `best_ball_ft.pt` beter dan `best.pt`?", ""]
    if ctx.quick:
        L += ["_Rooktest: 1 epoch op 320 px, dit oordeel betekent niets._", ""]
    for r in pipe_ft:
        if not r["model"].startswith("ball-finetune"):
            continue
        imgsz = r["imgsz"]
        suffix = "" if imgsz == ctx.imgsz else f"@{imgsz}"
        word, reasons = verdict(_ball_ap(runs, f"best.pt{suffix}"), _ball_ap(runs, f"ball-finetune{suffix}"),
                                _pipe_balls(pipe_best.get(imgsz)), _pipe_balls(r))
        L += [f"**imgsz {imgsz}: {word.upper()}**", ""] + [f"- {x}" for x in reasons] + [""]
    L += ["Criteria: hogere bal-AP50 op de testset (per klasse) én in de pijplijn meer ballen gevonden zonder meer "
          "valse ballen (`ground_truth.json` + `ground_truth_feet.json` samen, NMS iou 0,7). Let op de leakage "
          "hierboven: beide modellen zagen de testframes al tijdens het trainen van `best.pt`; de pijplijncijfers "
          "op de hele clip zijn de eerlijkste vergelijking die er nu is (de GT-frames liggen ook buiten de "
          "trainingsframes).", ""]
    return L


def write_summary(ctx: Ctx, steps, t_start):
    info, env = ctx.info, ctx.info.get("env", {})
    L = [f"# GPU-run match_analyse — {ctx.out.name}", ""]
    if ctx.quick:
        L += ["> **ROOKTEST (--quick)**: imgsz 320, 1 epoch, weinig frames. De getallen betekenen niets; "
              "dit controleert alleen dat alles draait.", ""]
    total = time.time() - t_start
    L += [f"- Apparaat: `{ctx.device}` — GPU: {env.get('gpu', 'geen')} ({env.get('vram_gb', '–')} GB), "
          f"torch {env.get('torch')}, CUDA {env.get('cuda_version')}, ultralytics {env.get('ultralytics')}",
          f"- Totale duur: {total / 60:.1f} min", f"- Val-imgsz: {ctx.imgsz}", ""]
    L += leakage_block(ctx)

    L += ["## Stappen", "", md_table(steps, ["stap", "status", "minuten", "opmerking"])]

    pr = info.get("pr", {}).get("runs", {})
    if pr:
        counts = info["pr"].get("test_counts", {})
        L += ["## 1. PR-curves en mAP (les 3.8, 5.6, 5.10)", "",
              f"Testset: {counts.get('images')} beelden, {counts.get('ball')} ballen, {counts.get('person')} personen. "
              "`val()` met conf 0,001 en NMS-IoU 0,7 (ultralytics-standaard).", "",
              "### Eerlijke vergelijking met COCO: bal + persoon (alle spelers, keeper en scheidsrechter samen)", "",
              "COCO kent geen teams: `sports ball` → bal, `person` → persoon. Ook onze modellen worden zo samengevoegd "
              "(zelfde code voor alle modellen).", ""]
        rows = []
        for r in pr.values():
            if r["view"] != "merged":
                continue
            row = {"model": r["model"], "imgsz": r["imgsz"]}
            for c in r["rows"]:
                if c["class"] in ("ball", "person"):
                    row.update({f"{c['class']} P": c["precision"], f"{c['class']} R": c["recall"],
                                f"{c['class']} AP50": c["mAP50"], f"{c['class']} AP50-95": c["mAP50-95"]})
            rows.append(row)
        L += [md_table(rows, ["model", "imgsz", "ball P", "ball R", "ball AP50", "ball AP50-95",
                              "person P", "person R", "person AP50", "person AP50-95"]),
              "Figuren: `pr/pr_ball_compare.png`, `pr/pr_person_compare.png`. P en R zijn de waarden bij de "
              "confidence met de hoogste gemiddelde F1 (ultralytics); bij R = 0 meldt ultralytics P = 1.", "",
              "### Per klasse (onze modellen)", ""]
        rows = [{**c, "model": r["model"], "imgsz": r["imgsz"]} for r in pr.values() if r["view"] == "classes"
                for c in r["rows"]]
        L += [md_table(rows, ["model", "imgsz", "class", "instances", "precision", "recall", "mAP50", "mAP50-95"]),
              "Ultralytics-figuren per model: `pr/<model>_<view>/BoxPR_curve.png`, `BoxF1_curve.png`, "
              "`confusion_matrix.png`. Alle cijfers: `pr/metrics_all.csv`.", ""]

    th = info.get("thresholds")
    if th:
        L += ["## 2. Drempelkeuze per klasse uit de F1-curve (les 5.4)", "",
              "Per klasse de confidence met de hoogste F1 op de testset. De pijplijn cachet YOLO op conf 0,05, "
              "lagere drempels werken daar als 0,05.", "",
              md_table(th, ["model", "view", "imgsz", "class", "best_conf", "f1", "precision", "recall"])]
        ball = [r for r in th if r["model"] == "best.pt" and r["class"] == "ball" and r["view"] == "classes"]
        pers = [r for r in th if r["model"] == "best.pt" and r["class"] == "person"]
        if ball and pers and ball[0]["best_conf"] is None:
            L += ["best.pt vindt op deze testset geen enkele bal (F1 = 0): geen baldrempel te kiezen.", ""]
        elif ball and pers:
            L += [f"Voorstel voor de pijplijn (best.pt): `--ball-conf {fmt(max(ball[0]['best_conf'], 0.05), 2)}`, "
                  f"`--conf {fmt(pers[0]['best_conf'], 2)}` (start van een persoonstrack). Huidige standaard: "
                  "0,1 en 0,3. Met weinig ballen in de testset is de baldrempel statistisch zwak; de Kalman-tracker "
                  "ruimt losse valse ballen op, dus een iets lagere baldrempel (meer recall) is verdedigbaar.", "",
                  "Figuren: `thresholds/f1_<run>.png`.", ""]

    nms = info.get("nms")
    if nms:
        L += ["## 3. NMS: IoU-sweep en NMS-vrije kop (les 5.4)", "",
              "YOLO26 heeft twee koppen: standaard draait de one2many-kop **met** NMS (`iou` telt); met `nms=False` "
              "draait de NMS-vrije one2one-kop (`iou` doet dan niets).", "",
              md_table(nms["val"], [c for c in nms["val_cols"] if c != "nms"]),
              "Figuur: `nms/sweep_val.png`.", "",
              "### Effect op de hele pijplijn, best.pt (volledige clip, alles ook in `pipeline_effect.csv`; GT body = `ground_truth.json`, voet = "
              "`ground_truth_feet.json`)", "",
              md_table(nms["pipeline"], PIPE_COLS),
              "`ball_found` = bal gevonden op GT-frames met zichtbare bal; `false_ball` = bal gemeld op GT-frames "
              "zonder zichtbare bal; `role_acc` = juiste rol (team/keeper/scheidsrechter) van gevonden spelers.", ""]

    ft = info.get("finetune")
    if ft:
        L += ["## 4. Fine-tune voor de bal", ""]
        if ft.get("status") == "klaar":
            vm = ft.get("val_metrics", {})
            L += [f"- Data: `{ft['data']}`", f"- imgsz {ft.get('imgsz')}, batch {ft.get('batch')}, "
                  f"{ft.get('epochs_run')} van max. {ft.get('epochs')} epochs (patience {ft.get('patience')}), "
                  f"beste epoch {ft.get('best_epoch')}, seed 0, deterministic, cos_lr, close_mosaic, "
                  f"cls_pw {ft.get('cls_pw')} (klassengewicht = 1/frequentie^cls_pw → bal weegt zwaarder)",
                  f"- Val-split van de fine-tune-data (standaard frames 200–245)" + (" — verdergezet na onderbreking" if ft.get("resumed") else "") + f": mAP50 {fmt(vm.get('metrics/mAP50(B)'))}, "
                  f"mAP50-95 {fmt(vm.get('metrics/mAP50-95(B)'))}",
                  f"- Duur: {fmt((ft.get('seconds') or 0) / 60, 1)} min — gewichten: `{ft.get('weights')}`",
                  f"- Trainingscurves: `{_rel(ctx, ft.get('save_dir', ''))}/results.png`", ""]
        else:
            L += [f"- Status: {ft.get('status')}", ""]
        if ft.get("overlap"):
            ov = ft["overlap"]
            L += [f"- Lekcontrole: {ov['n_overlap']} / {ov['n_test']} testbeelden in de fine-tune-data.", ""]

    L += verdict_block(ctx)

    L += ["## Omgeving", "", md_table([{"sleutel": k, "waarde": v} for k, v in env.items()
                                       if k not in ("files", "warnings", "problems")], ["sleutel", "waarde"])]
    if env.get("warnings"):
        L += ["Waarschuwingen:", ""] + [f"- {w}" for w in env["warnings"]] + [""]

    errors = [s for s in steps if s["status"] != "ok"]
    if errors:
        L += ["## Fouten", "", "Details: `log.txt`.", ""] + [f"- **{s['stap']}**: {s['opmerking']}" for s in errors] + [""]

    model_in_results = (ctx.out / "model" / FT_WEIGHTS.name).exists()
    n_cache = len(list((ctx.out / CACHE_EXPORT).glob("*.json"))) if (ctx.out / CACHE_EXPORT).exists() else 0
    L += ["## Terugsturen naar Kobe", "",
          f"Zip de hele map `kobe/match_analyse/gpu/results/{ctx.out.name}/` en stuur die zip terug. Daarin zit alles:",
          "",
          "- `SUMMARY.md` (dit bestand), `log.txt`, `env.json`, `state.json`, CSV's en figuren (`pr/`, `thresholds/`, "
          "`nms/`, `finetune/`, `pipeline_effect.csv`)",
          f"- `model/{FT_WEIGHTS.name}` — het fine-tuned model"
          + ("" if model_in_results else " (**ontbreekt**: fine-tune niet gelukt, overgeslagen of rooktest)"),
          f"- `{CACHE_EXPORT}/*.json` — {n_cache} detectiecaches van de hele clip (YOLO-uitvoer per instelling)",
          "",
          "### Wat Kobe ermee doet (op de laptop, zonder GPU)", "",
          "Vanuit `kobe/match_analyse/`:", "",
          f"1. `model/{FT_WEIGHTS.name}` kopiëren naar `models/{FT_WEIGHTS.name}`.",
          f"2. Alle `{CACHE_EXPORT}/*.json` kopiëren naar `cache/`. De cachesleutel is draagbaar (bestandsnaam, grootte "
          "en inhoud-hash van video en model, geen absolute paden of wijzigingstijden), dus de pijplijn gebruikt ze "
          "meteen en YOLO hoeft op de CPU niet opnieuw te draaien.",
          f"3. `python run_all.py --no-video --weights models/{FT_WEIGHTS.name}` (minimap-CSV's, analyses, "
          "evaluatie, `output/RAPPORT_best_ball_ft.md`). Met `--imgsz 1280` als de fine-tune op 1280 beter scoorde "
          "(zie het oordeel hierboven; die cache zit er ook bij).",
          "4. Voor de NMS-instellingen: de caches van `best.pt` met andere `iou`/`nms` worden alleen gebruikt door code "
          "die die instelling vraagt (bv. `core.pipeline.run(..., iou=0.5)`); de standaardinstelling (iou 0,7) "
          "gebruikt de cache van `best.pt` met iou 0,7.",
          ""]
    path = ctx.out / "SUMMARY.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
