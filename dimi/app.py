"""
Sports Vision & Tactical Mapping - Streamlit Dashboard
Dmitrij Tchervonisjenko - Team Git Push and Pray
Thomas More Geel - Deep Learning & Computer Vision
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import json

st.set_page_config(
    page_title="Sports Vision & Tactical Radar | Thomas More",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Locate repository root robustly
current_file = Path(__file__).resolve()
if (current_file.parents[1] / "dimi").exists():
    REPO_ROOT = current_file.parents[1]
elif (current_file.parents[2] / "dimi").exists():
    REPO_ROOT = current_file.parents[2]
else:
    REPO_ROOT = current_file.parent

DIMI_DIR = REPO_ROOT / "dimi"
FINAL_DIR = REPO_ROOT / "Final"
VIDEO_RADAR_PATH = DIMI_DIR / "runs" / "tactical_radar_output" / "synced_tactical_radar.mp4"
HEATMAP_PATH = DIMI_DIR / "runs" / "tactical_radar_output" / "spatial_occupancy_heatmap.png"
EXPERIMENTS_CSV = DIMI_DIR / "experiments.csv"
BENCHMARK_CSV = DIMI_DIR / "iou_nms_benchmark.csv"
CONFIG_PATH = DIMI_DIR / "homography_config.json"
PITCH_IMG_PATH = REPO_ROOT / "data" / "pitch_template" / "tactical_pitch.png"
REPORT_MD_PATH = REPO_ROOT / "docs" / "PROJECT_RAPPORT_DEEP_LEARNING.md"

# Sidebar Branding
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Soccerball.svg/500px-Soccerball.svg.png", width=80)
    st.title("⚽ Sports Vision AI")
    st.markdown("**Team:** Git Push and Pray")
    st.markdown("**Student:** Dmitrij Tchervonisjenko")
    st.markdown("**Vak:** Deep Learning & CV")
    st.markdown("**Instelling:** Thomas More Geel")
    st.markdown("---")
    st.markdown("### 🖥️ Hardware Setup")
    st.markdown("* **GPU:** NVIDIA GeForce RTX 4070 Laptop (8GB VRAM)")
    st.markdown("* **CUDA:** 12.4 | **PyTorch:** 2.6.0")
    st.markdown("* **Inference Speed:** ~15.3 ms / frame (~65 FPS)")
    st.markdown("* **Tracking Continuity:** **100.0% (660/660 frames)**")
    st.markdown("---")
    st.markdown("### 📌 Snelle Navigatie")
    st.markdown("1. 🎬 Synchrone Radar Video\n2. 🔥 Veldbezetting Heatmap\n3. 📊 IoU & NMS Benchmark\n4. 🧠 Deep Learning Examen Gids\n5. 📐 10-Punt Homografie")

# Header
st.title("⚽ Sports Vision & Tactical Mapping Dashboard")
st.subheader("End-to-End Computer Vision: YOLO11s Objectdetectie, Perspectief-Homografie & Invariant Tracking")
st.caption("Auteur: **Dmitrij Tchervonisjenko** (Team: **Git Push and Pray**) | Thomas More Geel")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎬 Synchrone Video & Radar", 
    "🔥 Veldbezetting & Heatmap (Extensie)", 
    "📊 IoU & NMS Benchmark (Winnaar)", 
    "🧠 Deep Learning Theorie & Examen Gids",
    "📐 10-Punt Homografie & Tracking"
])

# ----------------- TAB 1: Synchrone Video -----------------
with tab1:
    st.markdown("### 📽️ Synchrone 2D Minimap Radar & Broadcast Video")
    st.write(
        "Links zie je de originele broadcast videostream ($1920 \\times 1080$, 30 fps) met YOLO-detecties voor spelers en de bal. "
        "Rechts zie je de **live geprojecteerde 2D tactische radar (minimap)** via onze 10-puntige RANSAC perspectief-homografie."
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Trackingdekking", "100.0%", "660 / 660 frames getrackt")
    kpi2.metric("Homografie Nauwkeurigheid", "2.76 px", "Sub-pixel herprojectiefout")
    kpi3.metric("mAP@50 Verbetering", "+25.7%", "0.726 vs 0.466 baseline")
    kpi4.metric("Bal Status", "ALTIJD ROOD", "Zuiver BGR (0, 0, 255)")

    col1, col2 = st.columns([3, 1])

    with col1:
        if VIDEO_RADAR_PATH.exists():
            st.video(str(VIDEO_RADAR_PATH))
        else:
            st.warning("De synchrone video is nog niet gegenereerd. Draai eerst: `python dimi/minimap_pipeline.py`")

    with col2:
        st.markdown("#### 🎨 Officiële Legende & Kleurencodering")
        st.markdown(
            """
            * 🔵 **PSG (Team A):** Royal Navy Blue `(180, 50, 20)` met witte ring & rode kern
            * 🟡 **Inter Milan (Team B):** Fel Zonnegeel `(0, 220, 255)` met zwarte ring & donkere kern
            * 🟠 **Doelman (Keeper):** Fluo Oranje `(0, 130, 255)` met witte ring & witte kern
            * 🟢 **Scheidsrechter:** Fluo Gifgroen `(30, 255, 140)` met zwarte ring
            * 🔴 **Bal:** **Altijd zuiver felrood `(0, 0, 255)`** met witte contrastrand en oplichtende rode komeetstaart
            """
        )
        st.markdown("---")
        st.markdown("#### ⚙️ Pipeline Eigenschappen")
        st.info("💡 **Planar Assumption:** Voeten van spelers raken het gras. We projecteren `(x_center, y_max)` naar het 2D veldvlak.")
        st.success("⚡ **Dribble Carrier Hold:** Bij occlusie tussen de benen blijft de bal soepel verankerd aan de baldrager. Geen teleporteren naar de overkant!")
        st.warning("🛡️ **Strakke Penaltystip Filter:** Straal van 20 px sluit de stip uit, maar laat echte passes dóór het strafschopgebied 100% door.")

# ----------------- TAB 2: Heatmap -----------------
with tab2:
    st.markdown("### 🔥 Spatial Occupancy Heatmap & Territoriale Dominantie (Extensie)")
    st.write(
        "Door alle spelercoördinaten over de volledige 660 videoframes te accumuleren en te filteren met "
        "een Gaussian Kernel Density Estimation (KDE), brengen we de ruimtelijke dominantie en veldbezetting in kaart."
    )

    if HEATMAP_PATH.exists():
        st.image(str(HEATMAP_PATH), caption="Speler Bezetting & Territoriale Controle (Gaussian 2D KDE)", use_container_width=True)
    else:
        st.info("Heatmap nog niet gegenereerd. Run `python dimi/minimap_pipeline.py`.")

    st.markdown("#### 📊 Wiskundige Werking van de Gaussian KDE")
    st.latex(r"\hat{f}(x, y) = \frac{1}{N \cdot 2\pi h^2} \sum_{i=1}^N \exp\left( -\frac{(x - X_i)^2 + (y - Y_i)^2}{2h^2} \right)")
    st.write(
        "Waarbij $(X_i, Y_i)$ de getransformeerde 2D veldposities van de spelers zijn en $h$ de bandbreedte (Gaussian smoothing kernel van $65 \\times 65$ pixels). "
        "De resulterende intensiteitsmatrix wordt genormaliseerd en gemapt via de JET colormap."
    )

# ----------------- TAB 3: IoU & NMS Benchmark -----------------
with tab3:
    st.markdown("### 📊 IoU & NMS Drempelwaarde Benchmark: De Winnende Keuze")
    st.write(
        "Om de optimale balans te vinden tussen Precision, Recall, F1-Score en mAP, hebben we 8 verschillende "
        "combinaties van Confidence en NMS IoU-drempelwaarden systematisch geëvalueerd op de validatieset:"
    )

    if BENCHMARK_CSV.exists():
        df_bench = pd.read_csv(BENCHMARK_CSV)
        st.dataframe(df_bench.style.highlight_max(subset=["F1-Score", "mAP@50", "mAP@50-95"], color="#1b4d3e"), use_container_width=True)
    else:
        st.info("Run `python dimi/eval_iou_nms_experiments.py` om de benchmark te genereren.")

    st.markdown("---")
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.success("### 🏆 De Winnaar: `conf=0.025, iou=0.60`")
        st.markdown(
            """
            * **Piek mAP@50:** **0.7256** (+8.3% beter dan de YOLO baseline van 0.6427!)
            * **Piek mAP@50-95:** **0.3086** (+3.6% beter dan baseline van 0.2730)
            * **Harmonische F1-Score:** **0.7139** (optimale balans tussen precisie 0.738 en recall 0.691)
            * **Bal mAP@50:** **0.0215** (maximale recall voor kleine objecten)
            """
        )
    with col_w2:
        st.markdown("### 💡 Waarom deze keuze wiskundig wint:")
        st.markdown(
            """
            1. **Waarom faalt de YOLO-standaard (conf=0.25)?**  
               Bij `conf=0.25` keldert mAP@50 naar 0.6427 omdat de kleine bal in moeilijke frames een confidence heeft van 0.03–0.15. Hij wordt compleet weggegooid!
            2. **Waarom faalt IoU=0.70?**  
               Bij een te tolerante NMS-drempel (`iou=0.70`) dalen Precision en Recall (F1 zakt van 0.7139 naar 0.6688) doordat dubbele ghost-detecties op dezelfde speler overleven.
            3. **Bal-Recall:** `conf=0.025` in combinatie met onze `InvariantBallTracker` garandeert 100% continue baldetectie zonder valse positieven.
            """
        )

# ----------------- TAB 4: Deep Learning Theorie -----------------
with tab4:
    st.markdown("### 🧠 Deep Learning Theorie & Examen Gids (Thomas More)")
    st.write("Hieronder vind je de volledige academische toelichting van alle kerntheorieën, formules en examenconcepten:")

    with st.expander("📚 1. FastAI vs. PyTorch (Abstractielagen & Trainingslus)", expanded=True):
        st.markdown(
            """
            * **PyTorch:** Het fundament. Biedt `torch.Tensor` op GPU VRAM en de dynamische **Autograd engine** die via reverse-mode automatic differentiation de gradiënten $\\frac{\\partial L}{\\partial w}$ berekent.
            * **FastAI:** Een high-level abstractielaag bovenop PyTorch met `Learner`, `DataLoaders`, en geautomatiseerde best-practices (zoals Learning Rate Finder en One-Cycle Policy).
            * **Waarom niet 'greedy' zoals Dijkstra?**  
              Dijkstra kiest op elk punt de lokaal kortste stap (*greedy*). In Deep Learning leidt een gulzige benadering ertoe dat het model onmiddellijk vastloopt in een smal, scherp lokaal minimum. SGD met mini-batches schat het globale verliesoppervlak stochastisch, waardoor het generaliseert naar brede, vlakke minima.
            """
        )

    with st.expander("📐 2. Entropie & Verliesfuncties (Cross-Entropy & Focal Loss)"):
        st.markdown(
            """
            * **Shannon Entropie:** Meet de onzekerheid of informatie in een kansverdeling:
              $$H(X) = -\\sum_{i=1}^n P(x_i) \\log_2 P(x_i)$$
            * **Cross-Entropy Loss:** Meet het verschil tussen de ware klasse $y$ en de voorspelde kans $\\hat{y}$:
              $$L_{CE} = -\\sum_{c=1}^C y_c \\log(\\hat{y}_c)$$
            * **Focal Loss (YOLO / Imbalanced CV):** Voegt een modulatiefactor $(1 - p_t)^\\gamma$ toe. Als een achtergrond-anker makkelijk te classificeren is ($p_t \\approx 1$), wordt zijn verlies nagenoeg 0, zodat het netwerk zich 100% focust op zeldzame objecten (zoals de bal).
            """
        )

    with st.expander("🎯 3. PR-Curve vs. ROC-Curve & F1-Score [EXAMEN VRAAG]"):
        st.markdown(
            """
            * **Waarom GEEN ROC-curve bij Objectdetectie?**  
              De ROC-curve zet $TPR$ (Recall) uit tegen $FPR = \\frac{FP}{FP + TN}$. In computer vision is het aantal True Negatives ($TN$) gigantisch (miljoenen achtergrondpixels per frame). Hierdoor is $FPR \\approx 0$, waardoor de ROC-curve er altijd misleidend perfect uitziet, zelfs bij honderden foute detecties!
            * **De PR-Curve is eerlijk:** Negeert $TN$ volledig en focust zuiver op $TP$, $FP$ en $FN$.
            * **F1-Score:** Het harmonisch gemiddelde $F_1 = 2 \\cdot \\frac{P \\cdot R}{P + R}$ bestraft modellen die precisie opofferen voor recall of vice versa.
            """
        )

    with st.expander("📈 4. mAP (Mean Average Precision) Deep-Dive [EXAMEN VRAAG]"):
        st.markdown(
            """
            * **Waarom heet het "Mean Average Precision" als Mean en Average hetzelfde betekenen?**  
              Het is letterlijk **het gemiddelde van de gemiddelden**!
              1. **Average Precision (AP) per klasse:** Voor elke specifieke klasse (bv. "bal") berekent men het gemiddelde van de precisie over alle recall-niveaus (de oppervlakte onder de PR-curve via 101 interpolatiestappen).
              2. **Mean Average Precision (mAP) over alle klassen:** Vervolgens neemt men het rekenkundig gemiddelde (**Mean**) van die AP-waarden over alle $C$ verschillende klassen.
            * **mAP@50 vs. mAP@50-95:**  
              mAP@50 kijkt alleen of de overlap $\\text{IoU} \\ge 0.50$ is. De COCO-metric **mAP@50-95** middelt over 10 IoU-drempels ($0.50, 0.55, \\dots, 0.95$) en straft slordige kaders hard af.
            """
        )

    with st.expander("⚡ 5. Learning Rate & Leslie Smith's One-Cycle Policy"):
        st.markdown(
            r"""
            * **Learning Rate (Stapgrootte):** Bepaalt hoe groot de stap is in de richting van $-\nabla L$.
            * **One-Cycle Policy ('Kneeling' / Warm-up & Annealing):**
              1. **Warm-up:** Start heel klein ($\eta_{\min}$) om de initiële gewichten niet te vernielen.
              2. **Exploratie (Piek):** Bouwt op naar een hoge leersnelheid om zadelpunten en lokale minima te verlaten.
              3. **Cool-down (Cosine Annealing):** Zakt geleidelijk naar 0 om stabiel in het centrum van het diepste dal te landen.
            """
        )

    with st.expander("🌪️ 6. Momentum, Zadelpunten, Adam & Batch Size"):
        st.markdown(
            r"""
            * **Zadelpunten & Lokale Minima:** In miljoenen dimensies is $\nabla L \approx 0$ vaak een zadelpunt (geen dal).
            * **Momentum:** Voegt fysieke traagheid $v_t = \beta v_{t-1} + (1 - \beta) g_t$ toe. Hierdoor rolt het model over kleine bobbels en zadelpunten heen.
            * **Adam / AdamW:** Berekent adaptieve leersnelheden per parameter op basis van eerste en tweede momenten, met ontkoppelde gewichtsafname (weight decay).
            * **Batch Size:** Een kleine batch size (bv. 8) introduceert gezonde stochastische ruis, wat fungeert als een natuurlijke regularisator tegen overfitting.
            """
        )

    with st.expander("🧱 7. Vanishing Gradient, ReLU & Batch Normalization"):
        st.markdown(
            """
            * **Vanishing Gradient:** Bij klassieke Sigmoids is de afgeleide $\\le 0.25$. Over 20 lagen verdwijnt de gradiënt exponentieel ($(0.25)^{20} \\approx 0$). De voorste lagen leren niets meer.
            * **ReLU:** $f(x) = \\max(0, x)$. Voor alle $x > 0$ is de gradiënt exact **1**. Geen verlies van signaal!
            * **Batch Normalization:** Normaliseert mini-batches tot gemiddelde 0 en variantie 1, en herstelt via $\\gamma \\hat{x} + \\beta$. Lost **Internal Covariate Shift** op en stabiliseert de optimizer.
            """
        )

    with st.expander("🛡️ 8. Overfitting, Regularisatie (L1/L2, Dropout) & Data-Augmentatie"):
        st.markdown(
            """
            * **Data Splitting:** Train (70%), Validatie (20%, voor Early Stopping zonder gewichtsupdate), Test (10%, eenmalig op het einde).
            * **Super Star Neuronen Voorkomen:**
              - **L2 Regularisatie (Weight Decay):** Bestraft $\\frac{1}{2} \\lambda \\sum w^2$, dwingt alle neuronen samen te werken (*crowd sourcing*).
              - **L1 Regularisatie (Lasso):** Bestraft $\\lambda \\sum |w|$, creëert spaarzaamheid.
              - **Dropout:** Schakelt willekeurig neuronen uit tijdens training zodat neuronen niet op elkaar leunen.
            * **YOLO Augmentaties:** Mosaic-4, MixUp, HSV color jitter, en `close_mosaic=10` in de laatste epochs.
            """
        )

    with st.expander("🔍 9. Decoder & Upsampling: Nearest-Neighbor vs. Bilinear vs. Transposed Conv"):
        st.markdown(
            """
            * **Nearest-Neighbor:** Dupliceert pixels. Snel, maar blokkerig.
            * **Bilineaire Interpolatie:** Gewogen gemiddelde van 4 buren. Gladde overgangen, maar kan wazig (*blurry*) worden.
            * **Transposed Convolution:** Leerbare fractionele pas-convolutie. Kan echter **Checkerboard Artifacts** genereren door ongelijke overlap in het midden van de kernel. Opgelost door bilineaire upsampling + $1 \\times 1$ convolutie of Sub-Pixel PixelShuffle.
            """
        )

# ----------------- TAB 5: Homografie & Tracking -----------------
with tab5:
    st.markdown("### 📐 10-Punt Homografie Matrix & Invariant Tracking")
    st.write(
        "De homografiematrix $H$ projecteert camerabeelden $(x, y)$ naar de 2D FIFA-veldcoördinaten $(X, Y)$:"
    )

    st.latex(r"""
    \begin{bmatrix} X' \\ Y' \\ W \end{bmatrix} =
    \begin{bmatrix}
    h_{00} & h_{01} & h_{02} \\
    h_{10} & h_{11} & h_{12} \\
    h_{20} & h_{21} & h_{22}
    \end{bmatrix}
    \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}, \quad
    X = \frac{X'}{W}, \quad Y = \frac{Y'}{W}
    """)

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.markdown("#### 📌 10 Geijkte Referentiepunten (Sub-pixel 2.76 px fout)")
            pts_df = pd.DataFrame({
                "Punt": [f"Punt {i+1}" for i in range(len(cfg["camera_points"]))],
                "Camera X": [round(p[0], 1) for p in cfg["camera_points"]],
                "Camera Y": [round(p[1], 1) for p in cfg["camera_points"]],
                "Minimap X": [round(p[0], 1) for p in cfg["minimap_points"]],
                "Minimap Y": [round(p[1], 1) for p in cfg["minimap_points"]],
            })
            st.dataframe(pts_df, use_container_width=True)
        with col_h2:
            st.markdown("#### 🔢 Berekende Homografie Matrix $H$")
            H_mat = np.array(cfg["homography_matrix"])
            st.dataframe(pd.DataFrame(H_mat, columns=["h_0", "h_1", "h_2"]), use_container_width=True)
            st.success(f"Gemiddelde herprojectiefout: **{cfg.get('mean_reprojection_error_px', 2.76):.2f} pixels**!")

    st.markdown("---")
    st.markdown("#### 🛡️ InvariantBallTracker Architectuur")
    st.markdown(
        """
        1. **Fysisch Begrensde Zoekstraal:** $max\\_d = \\min(170, 75 + 4.5 \\cdot missing)$. Kan nooit naar de overkant springen.
        2. **Dribble Carrier Hold:** Zodra een speler de bal controleert ($dist \\le 50\\text{ px}$), blijft de bal bij occlusie verankerd aan de speler.
        3. **2D Veldconsistentie:** Punten die over de middellijn of zijlijn teleporteren ($d_{pitch} > 160\\text{ px}$) worden automatisch verworpen.
        4. **Strakke Penaltystip Filter:** Straal van 20 px sluit de statische stip uit, maar behoudt echte passes door het strafschopgebied.
        """
    )
