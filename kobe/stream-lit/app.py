"""Streamlit frontend: play with the parameters of the pipeline video -> detections -> minimap.

Run from the repo root:
    venv\\Scripts\\streamlit run kobe/stream-lit/app.py

This file only contains the UI; all logic is in pipeline.py (no Streamlit there, so it can be tested).
The sidebar has three groups of parameters:
1. Model:           needs YOLO again -> only on the "Detecteer" button, cached on disk
2. Post-processing: runs on the cached detections -> instant
3. Display:         only changes the drawing -> instant
"""

import hashlib
import json
import time
from dataclasses import asdict

import cv2
import pandas as pd
import streamlit as st

import pipeline as pl

st.set_page_config(page_title="CV-Challenge pipeline", layout="wide")


# ======================================================================
# Cached helpers
# ======================================================================

@st.cache_resource(show_spinner="Model laden ...")
def get_model(weights, mtime):
    """One YOLO model per weights file (mtime in the key: a retrained best.pt is loaded again)."""
    return pl.load_model(weights)


@st.cache_data(show_spinner=False)
def get_detections(cache_file):
    return pl.load_detections(cache_file)


@st.cache_data(show_spinner="Homografieën berekenen ...")
def get_homographies(ids, keypoints_mtime):
    """H per processed frame; keypoints_mtime in the key: new clicks are picked up."""
    return pl.frame_homographies(list(ids))


@st.cache_data(show_spinner=False)
def get_keyframes(keypoints_mtime):
    """Frames with clicked points (H is exact there, interpolated in between)."""
    return set(pl.Homographies().keyframes)


@st.cache_data(show_spinner="Post-processing ...", max_entries=8)
def get_result(cache_file, post, use_h, keypoints_mtime, video_fps):
    """Group 2 on the cached detections. Cached per parameter combination, so going back is instant."""
    dets = get_detections(cache_file)
    Hs = get_homographies(tuple(sorted(dets["frames"])), keypoints_mtime) if use_h else None
    return pl.postprocess(dets, Hs, pl.PostParams(**post), video_fps)


@st.cache_data(show_spinner=False, max_entries=4)
def get_frame(video, idx):
    return pl.read_frame(video, idx)


def rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def short_hash(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:10]


# ======================================================================
# Sidebar: video
# ======================================================================

st.sidebar.title("Parameters")
source = st.sidebar.radio("Video", ["match.mp4 (met minimap)", "Eigen mp4 uploaden"],
                          help="De aangeklikte homografiepunten (keypoints.json) gelden alleen voor match.mp4. "
                               "Bij een eigen video werkt de detectie, maar is de minimap uitgeschakeld.")
if source.startswith("match"):
    video = pl.DEFAULT_VIDEO
    if not video.exists():
        st.error(f"Video niet gevonden: `{video.relative_to(pl.REPO_ROOT)}`. Zet `match.mp4` in "
                 "`data/raw_video/` (video's zitten niet in git) of upload een eigen mp4.")
        st.stop()
else:
    upload = st.sidebar.file_uploader("mp4-bestand", type=["mp4"], help="Wordt opgeslagen in kobe/stream-lit/cache/uploads/.")
    if upload is None:
        st.info("Upload een mp4 in de sidebar.")
        st.stop()
    video = pl.CACHE_DIR / "uploads" / upload.name
    if not video.exists() or video.stat().st_size != upload.size:
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(upload.getbuffer())

info = pl.video_info(video)
if info is None or info["frames"] == 0:
    st.error(f"Kan de video `{video.name}` niet openen.")
    st.stop()
use_h = pl.has_keypoints(video)
keypoints_mtime = pl.KEYPOINTS_JSON.stat().st_mtime if pl.KEYPOINTS_JSON.exists() else 0.0

# ======================================================================
# Sidebar group 1: model (needs detection)
# ======================================================================

st.sidebar.header("1. Model")
st.sidebar.caption("Vereist opnieuw detecteren (traag, wordt gecachet).")
weights_files = pl.list_weights()
if not weights_files:
    st.error("Geen gewichten gevonden. Verwacht: `kobe/training/runs/<naam>/weights/best.pt`, "
             "`kobe/stream-lit/models/*.pt` of `kobe/homography/pretrained/yolo26s.pt` "
             "(.pt-bestanden zitten niet in git).")
    st.stop()
weights_label = st.sidebar.selectbox(
    "Gewichten", list(weights_files),
    help="Getrainde runs (ball, player, goalkeeper, referee) of het COCO-model (person → player, "
         "sports ball → ball). Eigen .pt-bestanden: zet ze in kobe/stream-lit/models/.")
weights = weights_files[weights_label]
imgsz = st.sidebar.select_slider(
    "Inferentie-imgsz", [640, 960, 1280], value=1280,
    help="Resolutie waarop YOLO het beeld ziet. Bij 640 is de bal ≈5 px en wordt hij bijna nooit gevonden; "
         "1280 is ≈3× trager.")
iou = st.sidebar.slider("NMS IoU", 0.3, 0.9, 0.7, 0.05,
                        help="Twee boxen die meer dan dit overlappen worden samengevoegd (non-maximum suppression). "
                             "Lager = minder dubbele boxen, maar spelers die elkaar overlappen kunnen verdwijnen.")
start, end = st.sidebar.slider("Framebereik", 0, info["frames"] - 1, (0, info["frames"] - 1),
                               help="Detecteer maar een deel van de clip om snel te testen.")
stride = st.sidebar.number_input("Stride", 1, 10, 1,
                                 help="Elke n-de frame. Stride 2 = half zoveel frames, half de detectietijd.")
ids = pl.frame_ids(start, end, stride)
cache_file = pl.cache_path(video, weights, imgsz, iou, start, end, stride)

SECONDS_PER_FRAME = {640: 0.2, 960: 0.37, 1280: 0.65}   # measured on Kobe's CPU with yolo26s
detect_clicked = st.sidebar.button("Detecteer", type="primary", width="stretch",
                                   help="Draait YOLO één keer op conf 0.05 en bewaart alle boxen op schijf.")
if cache_file.exists():
    st.sidebar.success(f"Detecties in cache ({len(ids)} frames).")
else:
    st.sidebar.warning(f"Geen detecties voor deze parameters: druk op **Detecteer** "
                       f"(≈ {len(ids) * SECONDS_PER_FRAME[imgsz] / 60:.1f} min op CPU met yolo26s).")

# ======================================================================
# Sidebar group 2: post-processing (instant)
# ======================================================================

st.sidebar.header("2. Post-processing")
st.sidebar.caption("Direct, op de gecachete detecties (geen YOLO).")
model_classes = get_detections(str(cache_file))["meta"]["classes"] if cache_file.exists() else \
    ["ball", "player", "goalkeeper", "referee"]
conf_people = st.sidebar.slider("Confidence spelers / keeper / scheidsrechter", 0.05, 0.95, 0.3, 0.05,
                                help="Boxen onder deze score worden weggelaten.")
conf_ball = st.sidebar.slider("Confidence bal", 0.05, 0.95, 0.1, 0.05,
                              help="Aparte drempel: de bal is klein en krijgt lagere scores dan spelers.")
classes = st.sidebar.multiselect("Klassen tonen", model_classes, default=model_classes,
                                 help="Welke klassen van dit model getoond en geteld worden.")
with st.sidebar.expander("Balfilter"):
    max_ball_size = st.slider("Max. balgrootte (px)", 10, 100, 40, 5,
                              help="Een grotere 'bal'-box is iets anders (hoofd, reclamebord).")
    confirm_frames = st.slider("Bevestiging: venster (frames)", 1, 15, 5,
                               help="Een bal telt alleen als een andere frame binnen dit venster ook een bal ziet.")
    confirm_speed = st.slider("Bevestiging: max. px per frame", 5, 100, 40, 5,
                              help="... en die andere bal ligt op hoogstens zoveel pixels per frame afstand.")
    max_ball_gap = st.slider("Max. gat (frames)", 0, 60, 15,
                             help="Gaten tot zoveel frames worden opgevuld (interpolatie / Kalman-voorspelling).")
    pitch_margin = st.slider("Marge rond het veld (m)", 0.0, 10.0, 3.0, 0.5,
                             help="Detecties verder buiten de lijnen (bank, publiek) worden genegeerd. "
                                  "Werkt alleen met homografie.")
teams_possible = "player" in model_classes
teams = st.sidebar.toggle("Teamkleuren (K-means op shirtkleur)", value=teams_possible, disabled=not teams_possible,
                          help="Verdeelt 'player' in 2 teams op de gemiddelde shirtkleur. Uitgeschakeld voor modellen "
                               "zonder samengevoegde 'player'-klasse (die hebben eigen teamklassen).")
outlier_factor = st.sidebar.slider("Outlier-factor", 1.0, 5.0, 2.5, 0.1, disabled=not teams,
                                   help="Een shirt verder dan factor × de typische afstand van beide teams "
                                        "wordt 'other' (keeper, scheidsrechter).")
tracking = st.sidebar.toggle("Tracking (ByteTrack)", value=True,
                             help="Koppelt boxen over frames tot tracks met een ID. Goedkoop: draait op de cache. "
                                  "Met teamkleuren krijgt elke track het team dat het vaakst voorkomt.")
with st.sidebar.expander("ByteTrack-drempels", expanded=False):
    track_activation = st.slider("Activatiedrempel", 0.05, 0.9, 0.25, 0.05, disabled=not tracking,
                                 help="Minimale confidence om een nieuwe track te starten.")
    lost_buffer = st.slider("Lost buffer (frames)", 1, 120, 30, disabled=not tracking,
                            help="Hoe lang (in frames bij 30 fps) een verloren track bewaard wordt.")
    match_threshold = st.slider("Matching-drempel", 0.1, 1.0, 0.8, 0.05, disabled=not tracking,
                                help="Hoger = boxen worden makkelijker aan een bestaande track gekoppeld.")
smoothing = st.sidebar.radio("Bal-smoothing", ["none", "interpolation", "kalman"], index=1,
                             format_func={"none": "Geen", "interpolation": "Alleen interpolatie",
                                          "kalman": "Kalmanfilter"}.get,
                             help="Interpolatie vult gaten lineair tussen twee ballen. Kalman (constante snelheid, "
                                  "in meters op het veld) maakt de baan gladder en voorspelt na het laatste zicht.")
kalman_accel, kalman_meas = 30.0, 0.5
if smoothing == "kalman":
    kalman_accel = st.sidebar.slider("Kalman: versnelling-std (m/s²)", 1.0, 100.0, 30.0, 1.0,
                                     help="Hoe hard de bal onverwacht kan versnellen (trap). Hoger = volgt metingen sneller.")
    kalman_meas = st.sidebar.slider("Kalman: meetfout-std (m)", 0.1, 5.0, 0.5, 0.1,
                                    help="Hoe onnauwkeurig een gemeten balpositie is. Hoger = gladder, maar meer vertraging.")
post = pl.PostParams(conf_people=conf_people, conf_ball=conf_ball, classes=tuple(classes),
                     max_ball_size=max_ball_size, ball_confirm_frames=confirm_frames,
                     ball_confirm_speed=float(confirm_speed), max_ball_gap=max_ball_gap, pitch_margin=pitch_margin,
                     teams=teams, outlier_factor=outlier_factor, tracking=tracking,
                     track_activation=track_activation, lost_buffer=lost_buffer, match_threshold=match_threshold,
                     ball_smoothing=smoothing, kalman_accel=kalman_accel, kalman_meas=kalman_meas)

# ======================================================================
# Sidebar group 3: display (instant)
# ======================================================================

st.sidebar.header("3. Weergave")
st.sidebar.caption("Direct, verandert alleen de tekening.")
display = pl.DisplayParams(
    pitch_lines=st.sidebar.checkbox("Veldlijnen op de video", True, disabled=not use_h,
                                    help="Geprojecteerde lijnen: liggen ze op de echte lijnen, dan klopt de homografie."),
    boxes=st.sidebar.checkbox("Boxen", True, help="Bounding boxes op de video."),
    labels=st.sidebar.checkbox("Labels (klasse/team + confidence)", False, help="Tekst boven elke box."),
    track_ids=st.sidebar.checkbox("Track-ID's", True, disabled=not tracking, help="#ID van ByteTrack, ook op de minimap."),
    trail=st.sidebar.slider("Trail-lengte (frames)", 0, 60, 0, disabled=not tracking,
                            help="Toon de laatste n posities van elke track (0 = uit)."),
    mini_scale=st.sidebar.slider("Minimap-schaal (px/m)", 4, 15, 8, disabled=not use_h,
                                 help="Grootte van de minimap in tab Frame, scherpte in de video."),
    camera_view=st.sidebar.checkbox("Camerabeeld op minimap", True, disabled=not use_h,
                                    help="Vierhoek = het deel van het veld dat de camera ziet."),
)

st.sidebar.info("Trainingsparameters (epochs, augmentations, imgsz bij training, modelgrootte) zitten niet in de "
                "app: train een ander model en kies het in de lijst.")

# ======================================================================
# Main area
# ======================================================================

st.title("Voetbal: video → detecties → minimap")
if not use_h:
    st.warning("Geen homografie voor deze video (keypoints.json geldt alleen voor match.mp4): "
               "detecties werken, maar de **minimap, veldlijnen en de veldmarge-filter zijn uitgeschakeld**.")

if detect_clicked and not cache_file.exists():
    bar = st.progress(0.0, text="Detecteren ...")
    t0 = time.time()

    def show_progress(done, total):
        eta = (time.time() - t0) / done * (total - done)
        bar.progress(done / total, text=f"Detecteren: frame {done}/{total}, nog ≈ {eta:.0f} s")

    model = get_model(str(weights), weights.stat().st_mtime)
    pl.detect(video, weights, imgsz, iou, start, end, stride, model=model, progress=show_progress)
    bar.empty()
    st.rerun()   # redraw the sidebar with the classes of this model

if not cache_file.exists():
    st.info("Druk op **Detecteer** in de sidebar. Tip: test eerst met een kort framebereik of stride 2–3.")
    st.stop()

detections = get_detections(str(cache_file))
result = get_result(str(cache_file), asdict(post), use_h, keypoints_mtime, info["fps"])
Hs = get_homographies(tuple(ids), keypoints_mtime) if use_h else None
for note in result["notes"]:
    st.info(note)

tab_frame, tab_video, tab_stats = st.tabs(["Frame", "Video", "Statistieken"])

# ---- Tab Frame: instant playground ----
with tab_frame:
    idx = st.slider("Frame", start, ids[-1], start, stride, help="Alleen gedetecteerde frames (stride).")
    frame = get_frame(str(video), idx)
    if frame is None:
        st.error(f"Frame {idx} kan niet gelezen worden.")
    else:
        image, mini = pl.render_frame(frame, idx, result, Hs[idx] if Hs else None, display)
        if mini is None:
            st.image(rgb(image), width="stretch")
        else:
            left, right = st.columns([3, 2])
            left.image(rgb(image), width="stretch")
            right.image(rgb(mini), width=mini.shape[1])
        people = result["frames"].get(idx, [])
        ball = result["ball"].get(idx)
        st.caption(f"{len(people)} personen · bal: {ball['kind'] if ball else 'niet gezien'}"
                   + (" · keyframe (aangeklikt)" if use_h and idx in get_keyframes(keypoints_mtime) else ""))

# ---- Tab Video: render the selected frame range ----
with tab_video:
    fps_out = info["fps"] / stride   # real-time playback, also with a stride
    out_path = pl.OUTPUT_DIR / f"render_{short_hash(cache_file.name, asdict(post), asdict(display), keypoints_mtime)}.mp4"
    st.write(f"Video naast minimap voor frames {start}–{end} (stride {stride}, {len(ids)} frames, {fps_out:.0f} fps).")
    if st.button("Render video", type="primary") and not out_path.exists():
        bar = st.progress(0.0, text="Renderen ...")
        pl.OUTPUT_DIR.mkdir(exist_ok=True)
        pl.render_video(video, ids, result, Hs, display, out_path, fps_out,
                        progress=lambda done, total: bar.progress(done / total, text=f"Renderen {done}/{total}"))
        bar.empty()
    if out_path.exists():
        data = out_path.read_bytes()
        st.video(data)
        st.download_button("Download mp4", data, file_name=f"pipeline_{start}-{end}.mp4", mime="video/mp4")
    else:
        st.caption("Nog niet gerenderd voor deze parameters.")

# ---- Tab Statistieken ----
with tab_stats:
    stats = pl.statistics(result, len(ids))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames", len(ids))
    c2.metric("Tracks", stats["tracks"], help=f"{stats['long_tracks']} tracks van ≥ 10 frames")
    c3.metric("Bal gezien (na filter + smoothing)", f"{stats['ball_pct']['track']:.0f}%")
    c4.metric("Detectietijd", f"{detections['meta']['seconds']:.0f} s",
              help=f"{detections['meta']['seconds'] / max(len(ids), 1):.2f} s per frame")

    left, right = st.columns(2)
    left.subheader("Detecties per klasse (na filter)")
    per_class = dict(stats["per_class"])
    if "ball" in post.classes:
        per_class["ball"] = result["ball_counts"]["confirmed"]   # confirmed detections, not interpolated
    left.dataframe(pd.DataFrame({"klasse": list(per_class), "aantal": list(per_class.values()),
                                 "per frame": [round(v / len(ids), 1) for v in per_class.values()]}),
                   hide_index=True, width="stretch")
    if len(stats["per_team"]) > 1:
        left.dataframe(pd.DataFrame({"team": list(stats["per_team"]), "aantal": list(stats["per_team"].values())}),
                       hide_index=True, width="stretch")
    right.subheader("Bal: % van de frames")
    right.dataframe(pd.DataFrame({
        "stap": ["ruw (conf + grootte + veld + stippen)", "na bevestiging", f"na smoothing ({smoothing})"],
        "%": [round(stats["ball_pct"][k], 1) for k in ("raw", "confirmed", "track")]}),
        hide_index=True, width="stretch")

    st.subheader("Gebruikte parameters")
    rows = [("1. Model", "video", video.name), ("1. Model", "gewichten", weights_label),
            ("1. Model", "imgsz", str(imgsz)), ("1. Model", "NMS IoU", str(iou)),
            ("1. Model", "frames", f"{start}–{end}, stride {stride}"), ("1. Model", "detectie-conf", str(pl.DETECT_CONF))]
    rows += [("2. Post-processing", k, str(v)) for k, v in asdict(post).items()]
    rows += [("3. Weergave", k, str(v)) for k, v in asdict(display).items()]
    st.dataframe(pd.DataFrame(rows, columns=["groep", "parameter", "waarde"]), hide_index=True,
                 width="stretch")
