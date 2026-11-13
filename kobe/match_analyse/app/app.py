"""Streamlit frontend of match_analyse: play with the parameters of video -> detection -> tracking -> minimap.

Run from the repo root (or anywhere; everything is found relative to this file):
    venv\\Scripts\\streamlit run kobe/match_analyse/app/app.py

Only UI: all logic is in core/ (see core/pipeline.py). Three parameter groups in the sidebar:
1. Model:           needs YOLO -> only on the "Detecteer" button, cached on disk by core.detection
2. Post-processing: core.pipeline.run() on the cached detections (no YOLO) -> fast
   (optional ball re-detection in crops: YOLO again, only on its own button, cached on disk too)
3. Display:         size of the preview
"""

import hashlib
import io
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # make `core` importable

import core.pipeline as pipe                      # noqa: E402
import core.tracking as tracking                  # noqa: E402
from analytics import draw_heatmap, heatmap, player_table, smooth_track, zone_figure   # noqa: E402
from analytics.extra import extra_stats           # noqa: E402
from analytics.report import TEAM_NAMES, zone_team_view   # noqa: E402
from core.ball_refine import crop_cache_path, detect_crops, missing_crops   # noqa: E402
from core.detection import cache_path, detect, load_model   # noqa: E402
from core.paths import CACHE_DIR, DEFAULT_WEIGHTS, MODELS_DIR, OUTPUT_DIR, VIDEO   # noqa: E402
from core.render import label, render_frame, write_video   # noqa: E402
from core.video import frame_ids, read_frame, video_info   # noqa: E402

st.set_page_config(page_title="match_analyse", layout="wide")


# ======================================================================
# Helpers
# ======================================================================

@st.cache_resource(show_spinner="Model laden ...")
def get_model(weights, mtime):
    """One YOLO model per weights file (mtime in the key: a retrained file is loaded again)."""
    return load_model(weights)


@contextmanager
def tuned(**overrides):
    """Temporarily set tunable constants of core (run() reads them at call time), then restore them."""
    targets = {"MAX_BALL_SIZE": pipe, "MAX_BALL_GAP": pipe, "BALL_START_CONF": pipe,
               "TRACK_BUFFER": tracking, "MATCH_THRESH": tracking}
    old = {k: getattr(targets[k], k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(targets[k], k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(targets[k], k, v)


@st.cache_resource(show_spinner="Post-processing ...", max_entries=6)
def get_result(video, weights, imgsz, iou, start, end, stride, conf, ball_conf, knobs, cache_mtime,
               ball_refine=False, crop_mtime=None):
    """Group 2 on the cached detections (the cache file exists, so run() never starts YOLO).

    ball_refine is only passed when all crops are cached (see below), so it never starts YOLO either.
    """
    with tuned(**dict(knobs)):
        return pipe.run(video, weights, imgsz, iou, start, end, stride, conf=conf, ball_conf=ball_conf,
                        ball_refine=ball_refine)


@st.cache_data(show_spinner=False, max_entries=4)
def get_frame(video, idx):
    return read_frame(video, idx)


def possession_so_far(result, idx):
    teams = [result.possession[i]["team"] for i in result.ids if i <= idx and result.possession[i]["team"] != "-"]
    return {t: 100 * teams.count(t) / len(teams) for t in sorted(set(teams))}


@st.cache_data(show_spinner=False, max_entries=4)
def get_analyses(tracks_csv, frames_csv):
    """Extra statistics + team heatmap PNGs from the csv bytes of the current result (cheap, cached by content)."""
    tracks, frames = pd.read_csv(io.BytesIO(tracks_csv)), pd.read_csv(io.BytesIO(frames_csv))
    table = player_table(tracks)
    extra = extra_stats(tracks, frames, table)
    heat = {}
    kept = tracks[tracks["track_id"].isin(table["track_id"])].sort_values(["track_id", "time_s"])
    for team in ("A", "B"):
        xs, ys = [], []
        for _, t in kept[kept["team"] == team].groupby("track_id"):
            sx, sy = smooth_track(t["time_s"], t["x_m"], t["y_m"])
            xs.extend(sx)
            ys.extend(sy)
        path = OUTPUT_DIR / f"app_heatmap_{team}.png"
        draw_heatmap(heatmap(xs, ys), f"Heatmap team {team}", path, attack_direction=extra["direction"]["sign"][team])
        heat[team] = path.read_bytes()
    return table, extra, heat


def csv_bytes(saver, result, name):
    return saver(result, OUTPUT_DIR / name).read_bytes()


def short_hash(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:10]


# ======================================================================
# Sidebar: video
# ======================================================================

OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
st.sidebar.title("Parameters")
source = st.sidebar.radio("Video", ["match.mp4", "Eigen mp4 uploaden"],
                          help="De aangeklikte homografiepunten (data/keypoints.json) gelden alleen voor match.mp4.")
if source == "match.mp4":
    video = VIDEO
    if not video.exists():
        st.error(f"Video niet gevonden: `{video}`. Zet `match.mp4` in `data/` (video's zitten niet in git).")
        st.stop()
else:
    upload = st.sidebar.file_uploader("mp4-bestand", type=["mp4"], help="Wordt bewaard in cache/uploads/.")
    if upload is None:
        st.info("Upload een mp4 in de sidebar.")
        st.stop()
    video = CACHE_DIR / "uploads" / Path(upload.name).name
    if not video.exists() or video.stat().st_size != upload.size:
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(upload.getbuffer())
    st.sidebar.warning("De homografie is gemaakt voor match.mp4: bij een andere video kloppen de minimap en "
                       "de veldposities niet.")

info = video_info(video)
if info is None or info["frames"] == 0:
    st.error(f"Kan de video `{video.name}` niet openen.")
    st.stop()

# ======================================================================
# Sidebar group 1: model (needs detection)
# ======================================================================

st.sidebar.header("1. Model")
st.sidebar.caption("Vereist opnieuw detecteren (traag, wordt gecachet).")
MODELS_DIR.mkdir(exist_ok=True)
weights_upload = st.sidebar.file_uploader("Eigen .pt uploaden (optioneel)", type=["pt"],
                                          help="Wordt bewaard in models/ en verschijnt in de lijst.")
if weights_upload is not None:
    target = MODELS_DIR / Path(weights_upload.name).name
    if not target.exists() or target.stat().st_size != weights_upload.size:
        target.write_bytes(weights_upload.getbuffer())
weights_files = {p.name: p for p in sorted(MODELS_DIR.glob("*.pt"))}
if not weights_files:
    st.error("Geen gewichten gevonden: zet een `.pt` in `models/` (zit niet in git) of upload er een.")
    st.stop()
names = list(weights_files)
weights_label = st.sidebar.selectbox(
    "Gewichten", names, index=names.index(DEFAULT_WEIGHTS.name) if DEFAULT_WEIGHTS.name in names else 0,
    help="Getrainde modellen in models/ (ball, player_A, player_B, keeper, referee), of een COCO-model "
         "(person, sports ball).")
weights = weights_files[weights_label]
imgsz = st.sidebar.select_slider("Inferentie-imgsz", [640, 960, 1280], value=pipe.DEFAULT_IMGSZ,
                                 help="Resolutie waarop YOLO het beeld ziet. 960 = waarop het model getraind is; "
                                      "640 mist de kleine bal.")
iou = st.sidebar.slider("NMS IoU", 0.3, 0.9, 0.7, 0.05,
                        help="Boxen die meer dan dit overlappen worden samengevoegd. Lager = minder dubbele boxen, "
                             "maar overlappende spelers kunnen verdwijnen.")
start, end = st.sidebar.slider("Framebereik", 0, info["frames"] - 1, (0, min(info["frames"] - 1, 120)),
                               help="Detecteer maar een deel van de clip om snel te testen.")
stride = int(st.sidebar.number_input("Stride", 1, 10, 2, help="Elke n-de frame. Stride 2 = half zoveel detectietijd."))
ids = frame_ids(start, end, stride)
cache_file = cache_path(video, weights, imgsz, iou, start, end, stride)
cached = cache_file.exists()

detect_clicked = st.sidebar.button("Detecteer", type="primary", width="stretch",
                                   help="Draait YOLO één keer op conf 0.05 en bewaart alle boxen op schijf.")
if cached:
    st.sidebar.success(f"Detecties in cache ({len(ids)} frames).")
else:
    st.sidebar.warning(f"Geen detecties voor deze parameters: druk op **Detecteer** ({len(ids)} frames).")

# ======================================================================
# Sidebar group 2: post-processing (fast)
# ======================================================================

st.sidebar.header("2. Post-processing")
st.sidebar.caption("Snel: draait op de gecachete detecties (geen YOLO).")
conf_people = st.sidebar.slider("Confidence spelers", 0.05, 0.95, 0.3, 0.05,
                                help="Minimale confidence om een spelertrack te starten.")
conf_ball = st.sidebar.slider("Confidence bal", 0.05, 0.95, pipe.DEFAULT_BALL_CONF, 0.05,
                              help="Aparte drempel voor balkandidaten: de bal is klein en krijgt lagere scores.")
with st.sidebar.expander("Tracker"):
    track_buffer = st.slider("Lost buffer (frames bij 30 fps)", 1, 120, tracking.TRACK_BUFFER,
                             help="Hoe lang een verloren track bewaard wordt (automatisch herschaald met de stride).")
    match_thresh = st.slider("Matching-drempel", 0.1, 1.0, tracking.MATCH_THRESH, 0.05,
                             help="Hoger = boxen worden makkelijker aan een bestaande track gekoppeld.")
with st.sidebar.expander("Bal"):
    max_ball_size = st.slider("Max. balgrootte (px)", 10, 100, pipe.MAX_BALL_SIZE, 5,
                              help="Een grotere 'bal'-box is iets anders (hoofd, reclamebord).")
    max_ball_gap = st.slider("Max. gat (frames)", 0, 60, pipe.MAX_BALL_GAP,
                             help="Gaten tot zoveel frames worden opgevuld (interpolatie).")
    ball_start_conf = st.slider("Confidence om een baltrack te starten", 0.05, 0.9, pipe.BALL_START_CONF, 0.05,
                                help="Zwakkere kandidaten kunnen een bestaande baltrack wel voortzetten.")
    ball_refine = st.checkbox("Herdetectie in crops", value=False,
                              help="Zoekt de bal opnieuw met YOLO in een uitsnede rond de verwachte positie, "
                                   "alleen in frames zonder bal. Traag de eerste keer (±0,1–0,3 s per frame, "
                                   "start pas na een klik), daarna uit de cache.")
knobs = (("MAX_BALL_SIZE", max_ball_size), ("MAX_BALL_GAP", max_ball_gap), ("BALL_START_CONF", ball_start_conf),
         ("TRACK_BUFFER", track_buffer), ("MATCH_THRESH", match_thresh))

# ======================================================================
# Sidebar group 3: display
# ======================================================================

st.sidebar.header("3. Weergave")
st.sidebar.caption("Alleen de preview; de renderer zelf is vast (boxen, track-ID's, veldlijnen, minimap).")
preview_scale = st.sidebar.slider("Preview-breedte (%)", 40, 100, 100, 10, help="Grootte van de preview in de pagina.")

# ======================================================================
# Main area
# ======================================================================

st.title("Voetbal: video → detecties → minimap")

if detect_clicked and not cached:
    bar = st.progress(0.0, text="Detecteren ...")
    t0 = time.time()

    def show_progress(done, total):
        eta = (time.time() - t0) / done * (total - done)
        bar.progress(done / total, text=f"Detecteren: frame {done}/{total}, nog ≈ {eta:.0f} s")

    detect(video, weights, imgsz, iou, start, end, stride,
           model=get_model(str(weights), weights.stat().st_mtime), progress=show_progress)
    bar.empty()
    st.rerun()

if not cached:
    st.info("Druk op **Detecteer** in de sidebar. Tip: test eerst met een kort framebereik of stride 2–3.")
    st.stop()

result = get_result(str(video), str(weights), imgsz, iou, start, end, stride, conf_people, conf_ball, knobs,
                    cache_file.stat().st_mtime)
refine_on = False   # True when the result below includes the crop re-detection
if ball_refine:
    windows = pipe.refine_windows(result)
    todo = missing_crops(video, weights, windows)
    if todo:
        st.info(f"Herdetectie in crops: {len(todo)} frames zonder bal moeten nog door YOLO. "
                "Druk op **Herdetecteer bal**; tot dan zie je het resultaat zonder herdetectie.")
        if st.button("Herdetecteer bal", type="primary"):
            bar = st.progress(0.0, text="Crops detecteren ...")
            detect_crops(video, weights, windows, model=get_model(str(weights), weights.stat().st_mtime),
                         progress=lambda done, total: bar.progress(done / total, text=f"Crop {done}/{total}"))
            bar.empty()
            st.rerun()
    else:
        crop_file = crop_cache_path(video, weights)
        result = get_result(str(video), str(weights), imgsz, iou, start, end, stride, conf_people, conf_ball, knobs,
                            cache_file.stat().st_mtime, True, crop_file.stat().st_mtime if crop_file.exists() else None)
        refine_on = True
share = pipe.possession_share(result)

tab_frame, tab_video, tab_stats, tab_analyses = st.tabs(["Frame", "Video", "Statistieken", "Analyses"])

with tab_frame:
    idx = st.slider("Frame", ids[0], ids[-1], ids[0], stride, help="Alleen verwerkte frames (stride).") \
        if len(ids) > 1 else ids[0]
    frame = get_frame(str(video), idx)
    if frame is None:
        st.error(f"Frame {idx} kan niet gelezen worden.")
    else:
        image = render_frame(frame, idx, result, possession_so_far(result, idx))
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width=int(image.shape[1] * preview_scale / 100))

with tab_video:
    fps_out = result.fps / stride
    out_path = OUTPUT_DIR / f"app_{short_hash(cache_file.name, conf_people, conf_ball, knobs, refine_on)}.mp4"
    st.write(f"Video + minimap voor frames {start}–{end} (stride {stride}, {len(ids)} frames, {fps_out:.0f} fps).")
    if st.button("Render video", type="primary") and not out_path.exists():
        bar = st.progress(0.0, text="Renderen ...")
        write_video(result, out_path,
                    progress=lambda done, total: bar.progress(done / total, text=f"Renderen {done}/{total}"))
        bar.empty()
    if out_path.exists():
        data = out_path.read_bytes()
        st.video(data)
        st.download_button("Download mp4", data, file_name=f"match_analyse_{start}-{end}.mp4", mime="video/mp4")
    else:
        st.caption("Nog niet gerenderd voor deze parameters.")

with tab_stats:
    c1, c2, c3 = st.columns(3)
    tracks = {p["track_id"] for ps in result.people.values() for p in ps}
    detected = sum(1 for i in result.ids if i in result.ball and result.ball[i]["state"] == "detected")
    c1.metric("Frames", len(ids))
    c2.metric("Tracks", len(tracks))
    c3.metric("Bal gedetecteerd", f"{100 * detected / len(ids):.0f}%",
              help=f"{100 * len(result.ball) / len(ids):.0f}% incl. interpolatie")

    st.subheader("Balbezit")
    if share:
        for col, (team, pct) in zip(st.columns(len(share)), share.items()):
            col.metric(label(team), f"{pct:.0f}%")
        if len(share) == 2:
            st.progress(share["A"] / 100, text="PSG (A) tegenover Inter (B)")
    else:
        st.caption("Geen balbezit gedetecteerd in dit bereik.")

    st.subheader("Download")
    d1, d2 = st.columns(2)
    d1.download_button("tracks.csv (per speler per frame)", csv_bytes(pipe.save_tracks_csv, result, "app_tracks.csv"),
                       file_name=f"tracks_{weights.stem}.csv", mime="text/csv")
    d2.download_button("frames.csv (bal + balbezit per frame)",
                       csv_bytes(pipe.save_frames_csv, result, "app_frames.csv"),
                       file_name=f"frames_{weights.stem}.csv", mime="text/csv")

    st.subheader("Gebruikte parameters")
    rows = [("1. Model", "video", video.name), ("1. Model", "gewichten", weights_label), ("1. Model", "imgsz", imgsz),
            ("1. Model", "NMS IoU", iou), ("1. Model", "frames", f"{start}-{end}, stride {stride}"),
            ("2. Post-processing", "conf spelers", conf_people), ("2. Post-processing", "conf bal", conf_ball)]
    rows += [("2. Post-processing", k, v) for k, v in knobs]
    rows += [("2. Post-processing", "herdetectie in crops", refine_on)]
    st.dataframe(pd.DataFrame(rows, columns=["groep", "parameter", "waarde"]).astype({"waarde": str}),
                 hide_index=True, width="stretch")

with tab_analyses:
    st.caption("Extra statistieken op het huidige resultaat (gecachet, snel; er wordt niets zwaars gestart). "
               "Bij een kort framebereik zijn de cijfers zeer grof.")
    a_tracks = csv_bytes(pipe.save_tracks_csv, result, "app_tracks.csv")
    a_frames = csv_bytes(pipe.save_frames_csv, result, "app_frames.csv")
    table, extra, heat = get_analyses(a_tracks, a_frames)
    if table.empty:
        st.info("Te weinig tracks (minstens 1 s) in dit framebereik voor analyses.")
    else:
        d, att = extra["direction"], extra["attacks"]
        st.subheader("Aanvalsrichting")
        st.markdown(f"**{d['text']['A']}** · {d['text']['B']}")
        st.caption(f"Betrouwbaarheid: {d['confidence']} ({d['agreement']}). Signalen (+1 = PSG naar rechts): "
                   f"keeper {d['signals']['keeper']:+d}, gem. positie {d['signals']['mean_x']:+d}, "
                   f"balrichting {d['signals']['ball']:+d}.")
        st.dataframe(pd.DataFrame([{"team": TEAM_NAMES[t], "balbezitfases": att[t]["phases"],
                                    "ingangen laatste derde": att[t]["final_third_entries"],
                                    "aanvallen": att[t]["attacks"], "ingangen 16 m-gebied": att[t]["box_entries"]}
                                   for t in ("A", "B")]), hide_index=True, width="stretch")
        st.caption("Schoten worden niet geteld (te onbetrouwbaar in trackingdata).")

        st.subheader("Snelheidszones")
        st.dataframe(zone_team_view(extra["zone_teams"]), hide_index=True, width="stretch")
        cols = st.columns(2)
        for col, team in zip(cols, ("A", "B")):
            zt = extra["zones"][extra["zones"]["team"] == team]
            if len(zt):
                row = extra["zone_teams"].set_index("team").loc[team]
                fig = zone_figure(zt, row, TEAM_NAMES[team], TEAM_NAMES[team])
                col.pyplot(fig)
                plt.close(fig)

        st.subheader("Kilometers (ondergrens)")
        km = extra["km_teams"].copy()
        km["team"] = km["team"].map(TEAM_NAMES)
        st.dataframe(km, hide_index=True, width="stretch")
        st.caption("Ondergrens: ID-switches splitsen spelers en korte tracks vallen weg. 'Per 90 minuten' is een "
                   "ruwe extrapolatie en alleen indicatief.")

        st.subheader("Heatmaps met aanvalsrichting")
        for col, team in zip(st.columns(2), ("A", "B")):
            col.image(heat[team], caption=TEAM_NAMES[team])
