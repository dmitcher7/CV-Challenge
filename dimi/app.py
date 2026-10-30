"""
Sports Vision & Tactical Mapping - Streamlit Dashboard
Dimi - Team Git Push and Pray
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd
import json

st.set_page_config(
    page_title="Sports Vision & Tactical Radar",
    page_icon="⚽",
    layout="wide",
)

DIMI_DIR = Path(__file__).resolve().parent
REPO_ROOT = DIMI_DIR.parent
VIDEO_RADAR_PATH = DIMI_DIR / "runs" / "tactical_radar_output" / "synced_tactical_radar.mp4"
HEATMAP_PATH = DIMI_DIR / "runs" / "tactical_radar_output" / "spatial_occupancy_heatmap.png"
EXPERIMENTS_CSV = DIMI_DIR / "experiments.csv"
CONFIG_PATH = DIMI_DIR / "homography_config.json"
PITCH_IMG_PATH = REPO_ROOT / "data" / "pitch_template" / "tactical_pitch.png"

# Header
st.title("⚽ Sports Vision & Tactical Radar")
st.subheader("Computer Vision Challenge: Voetbal Detectie & 2D Perspectief Homografie")
st.caption("Auteur: **Dmitrij Tchervonisjenko** | Team: **Git Push and Pray** | Thomas More Geel")

st.markdown("---")

# Tab navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "🎬 Synchrone Video & Radar", 
    "🔥 Speler Heatmap (Extensie)", 
    "📊 Model Prestaties & mAP", 
    "📐 Homografie Matrix & Theorie"
])

# ----------------- TAB 1: Synchrone Video -----------------
with tab1:
    st.markdown("### 📽️ Synchrone 2D Minimap Radar")
    st.write(
        "Links zie je de originele cameravideo met YOLO-detecties voor spelers en de bal. "
        "Rechts zie je de **live geprojecteerde 2D radar (minimap)** via OpenCV perspectief-homografie."
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        if VIDEO_RADAR_PATH.exists():
            st.video(str(VIDEO_RADAR_PATH))
        else:
            st.warning("De synchrone video is nog niet gegenereerd. Draai eerst: `python dimi/minimap_pipeline.py`")

    with col2:
        st.markdown("#### 🎨 Legende 2D Minimap")
        st.markdown(
            """
            * 🔵 **Team A (Blauw):** Veldspelers
            * 🟡 **Team B (Geel):** Veldspelers
            * 🟠 **Doelman (Oranje):** Keeper in het strafschopgebied
            * ⚫ **Scheidsrechter (Zwart):** Arbitrage
            * 🔴 **Bal (Rood/Wit):** Wedstrijdbal met dynamisch spoor (trail)
            """
        )
        st.info("💡 **Planar Assumption:** Spelers raken het veld met hun voeten. We projecteren daarom `(x_center, y_max)` van de bounding box!")

# ----------------- TAB 2: Heatmap -----------------
with tab2:
    st.markdown("### 🔥 Spatial Occupancy Heatmap (Extensie)")
    st.write(
        "Door alle spelercoördinaten over de volledige wedstrijdclip te accumuleren en te filteren met "
        "een Gaussian Kernel Density Estimation (KDE), brengen we de ruimtelijke dominantie en veldbezetting in kaart."
    )

    if HEATMAP_PATH.exists():
        st.image(str(HEATMAP_PATH), caption="Speler Bezetting & Territoriale Controle (KDE)", use_container_width=True)
    else:
        st.info("Heatmap nog niet gegenereerd. Run `python dimi/minimap_pipeline.py`.")

# ----------------- TAB 3: Metrics & mAP -----------------
with tab3:
    st.markdown("### 📊 Model Evaluatie & Vergelijking")
    st.write("Vergelijking van de prestaties tussen de eerdere baseline (CPU 640p) en Dimi's model (RTX 4070 GPU 1280p):")

    comparison_data = {
        "Klasse": ["⚽ ball", "player_A", "player_B", "player_B_goal_keeper", "referee", "ALL (Mean)"],
        "Baseline mAP@50 (Kobe/Brent 640p)": ["0.000 (0 hits)", "0.863", "0.667", "0.781", "0.016", "0.466"],
        "Dimi Model mAP@50 (RTX 4070 1280p)": ["0.035 (259 frames)", "0.944", "0.859", "0.953", "0.822", "0.723"],
        "Verbetering": ["✅ Doorbraak!", "+8.1%", "+19.2%", "+17.2%", "+80.6% 🚀", "+25.7% overall"]
    }
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    if EXPERIMENTS_CSV.exists():
        st.markdown("#### 📜 Gelogde Experimenten")
        exp_df = pd.read_csv(EXPERIMENTS_CSV)
        st.dataframe(exp_df, use_container_width=True)

# ----------------- TAB 4: Homografie Matrix -----------------
with tab4:
    st.markdown("### 📐 Wiskundige Homografie Transformatie")
    st.write(
        "Een homografiematrix $H$ is een $3 \\times 3$ transformatiematrix met 8 vrijheidsgraden. "
        "Hiermee transformeren we een punt $(x, y)$ in camerahoek naar een punt $(X, Y)$ op een 2D vlak:"
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
            st.json(cfg)
    else:
        st.info("Calibratiebestand nog niet opgeslagen. Run `python dimi/calibrate_homography.py` om jouw 4 punten live in te stellen.")
