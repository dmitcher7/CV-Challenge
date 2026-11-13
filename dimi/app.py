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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Robust path resolution within dimi
DIMI_DIR = Path(__file__).resolve().parent
REPO_ROOT = DIMI_DIR.parent
CONFIG_DIR = DIMI_DIR / "config"
VIDEO_RADAR_PATH = DIMI_DIR / "runs" / "tactical_radar_output" / "synced_tactical_radar.mp4"
HEATMAP_PATH = DIMI_DIR / "runs" / "tactical_radar_output" / "spatial_occupancy_heatmap.png"
EXPERIMENTS_CSV = CONFIG_DIR / "experiments.csv" if (CONFIG_DIR / "experiments.csv").exists() else DIMI_DIR / "experiments.csv"
BENCHMARK_CSV = CONFIG_DIR / "iou_nms_benchmark.csv" if (CONFIG_DIR / "iou_nms_benchmark.csv").exists() else DIMI_DIR / "iou_nms_benchmark.csv"
CONFIG_PATH = CONFIG_DIR / "homography_config.json" if (CONFIG_DIR / "homography_config.json").exists() else DIMI_DIR / "homography_config.json"
PITCH_IMG_PATH = REPO_ROOT / "data" / "pitch_template" / "tactical_pitch.png"
REPORT_MD_PATH = DIMI_DIR / "PROJECT_RAPPORT_DEEP_LEARNING.md"

# Clean Sidebar
with st.sidebar:
    st.subheader("Project Specificaties")
    st.markdown("**Project:** Sports Vision & Tactical Mapping")
    st.markdown("**Student:** Dmitrij Tchervonisjenko")
    st.markdown("**Team:** Git Push and Pray")
    st.markdown("**Instelling:** Thomas More Geel (Jaar 3)")
    st.markdown("**Vak:** Deep Learning & Computer Vision")
    st.markdown("---")
    st.subheader("Hardware & Runtime")
    st.markdown("- **GPU:** NVIDIA GeForce RTX 4070 Laptop (8GB VRAM)")
    st.markdown("- **CUDA:** 12.4 | **PyTorch:** 2.6.0")
    st.markdown("- **Inference Tijd:** ~15.3 ms / frame (~65 FPS)")
    st.markdown("- **Trackingdekking:** **100.0% (660/660 frames)**")
    st.markdown("---")
    st.subheader("Navigatie")
    st.markdown("1. Synchrone Video & Radar\n2. Veldbezetting Heatmap\n3. IoU & NMS Benchmark\n4. Multi-Scale & Pyramid Heads\n5. Deep Learning Theorie & Examen Gids\n6. 10-Punt Homografie & Tracking")

# Header
st.title("Sports Vision & Tactical Mapping Dashboard")
st.subheader("End-to-End Computer Vision: YOLO26s Multi-Scale Detectie, 10-Punt Homografie & Invariant Tracking")
st.caption("Auteur: Dmitrij Tchervonisjenko | Team: Git Push and Pray | Thomas More Geel")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Synchrone Video & Radar", 
    "Veldbezetting Heatmap", 
    "IoU & NMS Benchmark", 
    "Multi-Scale & Pyramid Heads",
    "Deep Learning Theorie & Examen Gids",
    "10-Punt Homografie & Tracking"
])

# ----------------- TAB 1: Synchrone Video -----------------
with tab1:
    st.subheader("Synchrone 2D Minimap Radar & Broadcast Video")
    st.write(
        "Links: originele broadcast videostream (1920x1080, 30 fps) met multi-scale YOLO26s detecties voor spelers en de bal. "
        "Rechts: synchroon geprojecteerde 2D tactische radar via onze 10-punts RANSAC perspectief-homografie."
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Trackingdekking", "100.0%", "660 / 660 frames getrackt")
    kpi2.metric("Homografie Nauwkeurigheid", "2.76 px", "Sub-pixel herprojectiefout")
    kpi3.metric("mAP@50 Verbetering", "+25.7%", "0.726 vs 0.466 baseline")
    kpi4.metric("Bal Weergave", "Consistent Rood", "Zuiver BGR (0, 0, 255)")

    col1, col2 = st.columns([3, 1])

    with col1:
        if VIDEO_RADAR_PATH.exists():
            with open(VIDEO_RADAR_PATH, "rb") as vf:
                video_bytes = vf.read()
            st.video(video_bytes, format="video/mp4")
            st.caption(f"Bestand: {VIDEO_RADAR_PATH.name} (H.264 / AAC, {len(video_bytes)/(1024*1024):.1f} MB)")
        else:
            st.warning("De synchrone video is nog niet gegenereerd. Draai: python dimi/minimap_pipeline.py")

    with col2:
        st.subheader("Legende & Codering")
        st.markdown(
            """
            - **PSG (Team A):** Royal Navy Blue met witte ring en rode kern
            - **Inter Milan (Team B):** Fel Zonnegeel met zwarte ring
            - **Doelman:** Fluo Oranje met witte ring
            - **Scheidsrechter:** Fluo Gifgroen met zwarte ring
            - **Bal:** Zuiver felrood met witte contrastrand en dynamische komeetstaart
            """
        )
        st.markdown("---")
        st.subheader("Pipeline Eigenschappen")
        st.markdown("**Planar Assumption:** Voeten van spelers raken het gras (`x_center, y_max`).")
        st.markdown("**Dribble Carrier Hold:** Bij occlusie tussen benen blijft de bal verankerd aan de baldrager.")
        st.markdown("**Micro-Penaltystip Filter:** Straal van 20 px sluit de stip uit, maar laat passes ongehinderd door.")

# ----------------- TAB 2: Heatmap -----------------
with tab2:
    st.subheader("Spatial Occupancy Heatmap (2D Gaussiaanse KDE)")
    st.write(
        "Deze extensie visualiseert de ruimtelijke verdeling en veldbezetting van spelers over de volledige videoduur via "
        "2D Kernel Density Estimation (KDE) met Gaussiaanse kernels:"
    )
    st.latex(r"f(x, y) = \frac{1}{2\pi \sigma^2} \sum_{i=1}^N \exp\left( -\frac{(x - x_i)^2 + (y - y_i)^2}{2\sigma^2} \right)")

    if HEATMAP_PATH.exists():
        st.image(str(HEATMAP_PATH), caption="2D Gaussiaanse Territoriale Dominantie Heatmap", use_container_width=True)
    else:
        st.warning("Heatmap niet gevonden. Draai: python dimi/minimap_pipeline.py")

# ----------------- TAB 3: IoU & NMS Benchmark -----------------
with tab3:
    st.subheader("Systematische Evaluatie van IoU & NMS Drempelwaarden")
    st.write(
        "Om de optimale balans tussen precisie, recall, lokalisatiescherpte en kleine-baldetectie te bepalen, "
        "zijn 18 verschillende combinaties van Confidence (`conf`) en Non-Maximum Suppression (`iou`) geëvalueerd op de validatieset (RTX 4070 GPU):"
    )

    if BENCHMARK_CSV.exists():
        df_bench = pd.read_csv(BENCHMARK_CSV)

        # Highlight top metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Beste mAP@50", "0.7291", "conf=0.005, iou=0.60 (+10.0% vs baseline)")
        c2.metric("Beste mAP@50-95", "0.3201", "conf=0.005, iou=0.70")
        c3.metric("Beste F1-Score", "0.7115", "P=0.7534 | R=0.6740")
        c4.metric("Beste Bal mAP@50", "0.0676", "+35.2% winst t.o.v. baseline (0.050)")

        st.dataframe(df_bench, use_container_width=True)

        st.subheader("Visuele Drempelwaarde Vergelijkingen")
        chart_img = RUNS_DIR / "threshold_comparison_charts.png"
        if chart_img.exists():
            st.image(str(chart_img), caption="Systematische Vergelijking van 18 Threshold Configuraties (mAP, PR Trade-Off, Bal Gevoeligheid & IoU NMS)", use_container_width=True)

        st.subheader("Officiële Validatiecurven: PR-Curve, F1-Curve & Verwarringsmatrix")
        st.write(
            "Bij objectdetectie gebruiken we conform de opdracht uitsluitend de **Precision-Recall Curve (PR-Curve)** "
            "in plaats van een ROC-curve, omdat de achtergrondklasse ('geen bal/speler') zwaar oververtegenwoordigd is (extreme class imbalance)."
        )

        col_pr, col_f1 = st.columns(2)
        pr_path = RUNS_DIR / RUN_NAME / "BoxPR_curve.png"
        f1_path = RUNS_DIR / RUN_NAME / "BoxF1_curve.png"
        if pr_path.exists():
            col_pr.image(str(pr_path), caption="Precision-Recall Curve (Area Under Curve = mAP@50 per klasse)", use_container_width=True)
        if f1_path.exists():
            col_f1.image(str(f1_path), caption="F1-Confidence Curve (Piek F1 rond lage confidence thresholds)", use_container_width=True)

        col_p, col_cm = st.columns(2)
        p_path = RUNS_DIR / RUN_NAME / "BoxP_curve.png"
        cm_path = RUNS_DIR / RUN_NAME / "confusion_matrix_normalized.png"
        if p_path.exists():
            col_p.image(str(p_path), caption="Precision-Confidence Curve (Stabiele precisie bij toenemende drempel)", use_container_width=True)
        if cm_path.exists():
            col_cm.image(str(cm_path), caption="Genormaliseerde Verwarringsmatrix (YOLO26s op test/val)", use_container_width=True)

        st.subheader("Verantwoording & Analyse van de Drempelwaarden")
        st.markdown(
            r"""
            ### 1. De Winnaar voor Productie: `conf = 0.005` & `iou = 0.60`
            - **Hoogste mAP@50:** **0.7291** (een sprong van **+10.0%** ten opzichte van de standaard YOLO baseline van $0.6292$).
            - **Hoogste F1-Score:** **0.7115** met een robuuste Precision van $75.34\%$ en Recall van $67.40\%$.
            - **Hoogste Bal-detectie:** Bal mAP@50 stijgt van $0.0500$ naar **$0.0676$** (+35.2% winst).
            - **Waarom deze drempel?** Een kleine voetbal van slechts $16 \times 17\text{ px}$ op een 1080p broadcast video heeft door bewegingsonscherpte en occlusie vaak een lagere netwerkconfidence (tussen $0.01$ en $0.05$). Door `conf` op $0.005$ te zetten, worden deze vroege kandidaat-detecties niet weggegooid, waardoor de spatiale tracker 100% dekking behoudt.

            ### 2. De IoU Threshold Trade-off (NMS analyse)
            - **Strakke NMS (`iou = 0.50`):** Onderdrukt kaders te agressief. Wanneer twee spelers van hetzelfde team een duel uitvechten, wordt de tweede speler per ongeluk geëlimineerd als false negative ($mAP@50 = 0.7236$).
            - **Gebalanceerde NMS (`iou = 0.60` - WINNAAR):** Geeft de absolute piek in mAP@50 ($0.7291$) en behoudt een hoge precisie ($75.34\%$).
            - **Permissieve NMS (`iou = 0.70`):** Behaalt de hoogste strikte lokalisatie $mAP@50-95 = 0.3201$, maar verlaagt de precisie van $75.34\%$ naar $70.44\%$ omdat overlappende duplicaatkaders niet snel genoeg worden samengevoegd.

            ### 3. Falen van de Standaard YOLO Baseline (`conf = 0.25`)
            De standaard YOLO instelling (`conf = 0.25, iou = 0.60`) behaalt slechts een mAP@50 van **0.6292**. Hij mist systematisch de snelle bal en deels bedekte spelers aan de rand van het veld, wat bewijst dat domeinspecifieke threshold tuning onmisbaar is in sport-analyse.
            """
        )
    else:
        st.warning("Benchmark bestand niet gevonden.")

# ----------------- TAB 4: Multi-Scale & Pyramid Heads -----------------
with tab4:
    st.subheader("Multi-Scale Architectuur & Pyramid Heads")
    st.write(
        "Hoe slaagt het model erin om zowel een gigantisch speelveld, middelgrote spelers als een minuscule voetbal gelijktijdig te detecteren?"
    )

    st.markdown(
        """
        ### 1. Backbone: Feature Extractie (C3, C4, C5)
        Tijdens het doorlopen van de convolutielagen verliest het beeld spatiale resolutie, maar wint het aan semantische abstractie:
        - **C3 (Stride 8):** Hoge spatiale resolutie ($160 \times 160$ bij 1280p). Bevat fijne details en randen, maar weinig abstracte context.
        - **C4 (Stride 16):** Medium resolutie ($80 \times 80$) met gemiddelde semantiek.
        - **C5 (Stride 32):** Lage spatiale resolutie ($40 \times 40$), maar rijke semantische context (begrijpt complete concepten).
        """
    )

    st.markdown(
        """
        ### 2. Neck (Feature Pyramid Network): Upsampling & Fusie
        - Als we enkel op C5 detecteren, is de kleine voetbal ($16 \times 17$ px) allang weggeschaald.
        - Als we enkel op C3 detecteren, ontbreekt semantisch begrip (het netwerk verwart een witte schoen met de bal).
        - **De Neck lost dit op:** Hij neemt de diepe laag C5, upsamplet deze en voegt deze samen met laag C4. 
          Vervolgens upsamplet hij opnieuw en voegt samen met C3. Daarna voert een bottom-up pad (PAN) de gecombineerde informatie weer terug naar beneden.
        - **Resultaat:** Drie piramidelagen (**P3, P4, P5**) die **allebei bezitten: scherpe ruimtelijke details én rijke semantische context!**
        """
    )

    st.markdown(
        """
        ### 3. De Drie Predictie Heads (P3, P4, P5)
        Elke piramidelaag heeft zijn eigen predictiekop met een specifiek receptief veld:
        - **P3 Head ($160 \times 160$ grid):** Hoogste resolutie, kleinste ankers -> **Detecteert de kleine voetbal ($16 \times 17$ px)**.
        - **P4 Head ($80 \times 80$ grid):** Middelgrote kaders -> Detecteert spelers verder weg op het veld.
        - **P5 Head ($40 \times 40$ grid):** Grootste kaders -> Detecteert nabije spelers en doelgebieden.
        """
    )

# ----------------- TAB 5: Deep Learning Theorie & Examen Gids -----------------
with tab5:
    st.subheader("Deep Learning & Computer Vision Compendium")

    with st.expander("1. Big Data, Latente Ruimte & De Trainingslus"):
        st.markdown(
            r"""
            - **Big Data (5 V's):** Volume, Velocity, Variety, Veracity en Value. We schalen horizontaal (scale out) met meerdere GPU-nodes.
            - **Latente Ruimte:** Gecomprimeerde representatie waarin alleen essentiële kenmerken overeind blijven (Encoder -> Bottleneck -> Decoder).
            - **Neuron Wiskunde:** Lineair dot product $z = \mathbf{w}^T \mathbf{x} + b$ gevolgd door een niet-lineaire activatie (ReLU, Leaky ReLU, SiLU).
            - **Waarom niet greedy zoals Dijkstra?** Dijkstra kiest altijd de lokaal kortste stap. In deep learning leidt dit direct tot vastlopen in het eerste scherpe lokale minimum of zadelpunt. Mini-batch SGD zoekt stochastisch naar brede, vlakke bekkens die optimaal generaliseren.
            """
        )

    with st.expander("2. Activaties, Vanishing Gradients & Batch Normalization"):
        st.markdown(
            r"""
            - **Vanishing Gradients:** Bij Sigmoid/Tanh ($\le 0.25$) sterven gradiënten in diepe netwerken exponentieel uit ($0.25^{20} \to 0$).
            - **ReLU ($\max(0, z)$):** Afgeleide is exact 1 voor positieve waarden, waardoor gradiënten ongehinderd doorstromen.
            - **Batch Normalization:** Recentreert en herschaalt tussentijdse lagen ($\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$) om Internal Covariate Shift te neutraliseren.
            """
        )

    with st.expander("3. Regularisatie: L1/L2, Dropout & Data-Augmentatie"):
        st.markdown(
            """
            - **Anti-Superstar Neuronen:** Zonder regularisatie trekt één dominant neuron alle taken naar zich toe.
            - **Dropout:** Schakelt willekeurige neuronen tijdelijk uit tijdens training, wat representaties dwingt om redundant en gecrowdsourced te worden.
            - **L1 (Lasso) vs. L2 (Weight Decay):** L1 stimuleert spaarzaamheid; L2 bestraft grote gewichten kwadratisch.
            - **close_mosaic=10:** Schakelt samengestelde Mosaic augmentaties 10 epochs voor het einde uit zodat het model afstemt op natuurlijke beelden.
            """
        )

    with st.expander("4. Hard Labels vs. Soft Labels & Label Smoothing"):
        st.markdown(
            r"""
            - **Hard Labels:** $y_{hard} = [1, 0, 0, 0, 0]$ forceert logits naar $\pm \infty$. Dit veroorzaakt overconfidence en *entropy collapse*: bij wazige of deels occluded ballen klapt de doelkans direct in naar 0.
            - **Soft Labels:** We mengen het doel met een uniforme verdeling over $K=5$ klassen:
            $$y_{soft} = (1 - \epsilon) \cdot y_{hard} + \frac{\epsilon}{K}$$
            - **Task-Aligned Assigner in YOLO:** Berekent continue zachte doelen $t = s^\alpha \cdot \text{IoU}^\beta$ tussen 0 en 1, waardoor het netwerk gekalibreerde kansen leert.
            """
        )
        eps_slider = st.slider("Label Smoothing factor (ε):", min_value=0.0, max_value=0.25, value=0.10, step=0.01)
        k_classes = 5
        p_main = (1.0 - eps_slider) + (eps_slider / k_classes)
        p_sub = eps_slider / k_classes
        df_soft_vis = pd.DataFrame({
            "Klasse": ["ball (Doelklasse)", "player_A", "player_B", "player_B_gk", "referee"],
            "Hard Label": [1.0, 0.0, 0.0, 0.0, 0.0],
            f"Soft Label (ε = {eps_slider:.2f})": [round(p_main, 4), round(p_sub, 4), round(p_sub, 4), round(p_sub, 4), round(p_sub, 4)]
        })
        st.dataframe(df_soft_vis, use_container_width=True)

    with st.expander("5. Evaluatie: Waarom NOOIT een ROC-Curve bij Objectdetectie? [EXAMEN VRAAG]"):
        st.markdown(
            r"""
            - **FPR Formule:** $\text{FPR} = \frac{FP}{FP + TN}$.
            - In een 1080p broadcast frame zijn er honderdduizenden achtergrondvensters waar geen object staat ($TN > 100.000$).
            - Omdat $TN \to \infty$, is $\text{FPR} \approx 0$ ongeacht hoeveel valse detecties ($FP$) het model genereert.
            - De ROC-curve toont daardoor een misleidend perfecte score ($\text{AUC} \approx 0.999$).
            - **De Precision-Recall (PR) curve is de enige eerlijke metric:** Negeert $TN$ volledig en toetst zuiver op $TP$, $FP$ en $FN$.
            """
        )

    with st.expander("6. mAP Ontrafeld: Het Gemiddelde van de Gemiddelden [EXAMEN VRAAG]"):
        st.markdown(
            r"""
            - **Mean** en **Average** zijn synoniemen, maar opereren op twee hiërarchische niveaus:
              1. **Average Precision (AP) per klasse:** De oppervlakte onder de PR-curve (AUC-PR) over alle recall-drempels voor één specifieke klasse.
              2. **Mean Average Precision (mAP) over alle klassen:** Het rekenkundig gemiddelde (*Mean*) van die AP-waarden over alle $C$ unieke klassen.
            - **mAP@50 vs. mAP@50-95:** mAP@50 toetst met een tolerante drempel van $\text{IoU} \ge 0.50$; mAP@50-95 middelt over 10 drempels ($0.50, 0.55, \dots, 0.95$) en straft onnauwkeurige bounding boxes zwaar af.
            """
        )

    with st.expander("7. Model Zoo Evolutie: VGG, GoogLeNet (1x1 Convolutie) & ResNet"):
        st.markdown(
            r"""
            - **VGG (2014):** Stapelen van kleine $3\times 3$ filters vervangt grote filters met minder parameters.
            - **GoogLeNet / Inception (2014):** Introduceerde de **$1\times 1$ convolutie als bottleneck** om kanaaldiepte drastisch te reduceren zonder spatiale informatie te verliezen.
            - **ResNet (2015):** Skip connections ($F(x) + x$) laten signalen lagen overslaan, wat vanishing gradients oplost en netwerken $>100$ lagen diep toelaat.
            """
        )

    with st.expander("8. Semantische Segmentatie & U-Net Skip Connections"):
        st.markdown(
            r"""
            - **Semantic vs. Instance vs. Panoptic:** Semantic labelt categorieën (gras, lucht); Instance labelt individuele objecten (speler 1, speler 2); Panoptic combineert beide.
            - **Fully Convolutional Networks (FCNs):** Vervangen Flatten/Dense lagen door $1\times 1$ convoluties om de 2D spatiale kaart te behouden.
            - **U-Net:** Encoder pooling verliest scherpe randen; Decoder upsamplet via Transposed Convolutions; Skip Connections voeren originele hoge-resolutie features rechtstreeks naar de decoder.
            - **Loss bij Segmentatie:** Cross-Entropy wordt altijd gecombineerd met Shape Overlap Loss (Dice of Soft IoU) om $95\%$ achtergrondoverwicht te compenseren. Argmax wordt pas na training toegepast omdat de afgeleide 0 is.
            """
        )

    with st.expander("9. Foundation Models & Zelf-Gesuperviseerd Leren (MAE, DINO, CLIP, SAM)"):
        st.markdown(
            """
            - **MAE (Masked Auto-Encoder):** Vision Transformer die 75% van beeldpatches maskeert en dwingt te reconstrueren uit ruwe pixels.
            - **DINO:** Zelf-gesuperviseerde Teacher-Student distillatie zonder labels; de teacher wordt geüpdatet via een trage Exponential Moving Average (EMA).
            - **CLIP:** Koppelt beelden en tekst in een gedeelde latente ruimte via Contrastive Loss (maakt zero-shot classificatie mogelijk).
            - **SAM (Segment Anything Model):** Promptable Vision; segmenteert elk gewenst object op basis van een punt of box-prompt.
            """
        )

# ----------------- TAB 6: 10-Punt Homografie & Tracking -----------------
with tab6:
    st.subheader("10-Punt RANSAC Homografie & Invariant Tracking")

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        st.markdown(f"**Beschrijving:** {cfg['description']}")
        st.markdown(f"**Aantal Referentiepunten:** {cfg['num_points']}")
        st.markdown(f"**Gemiddelde Herprojectiefout (RMSE):** **{cfg['mean_reprojection_error_px']:.2f} pixels**")

        st.subheader("Calibratietabel (Camera vs. 2D Minimap Veldcoördinaten)")
        table_rows = []
        for i, (cp, mp, lbl) in enumerate(zip(cfg['camera_points'], cfg['minimap_points'], cfg['point_labels'])):
            table_rows.append({
                "Index": i + 1,
                "Referentiepunt": lbl.split('->')[0].strip(),
                "Camera Positie (px)": f"({cp[0]:.1f}, {cp[1]:.1f})",
                "2D Veld Positie (px)": f"({mp[0]:.1f}, {mp[1]:.1f})"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        st.subheader("Berekende 3x3 Homografie Matrix H")
        cam_arr = np.array(cfg['camera_points'], dtype=np.float32)
        map_arr = np.array(cfg['minimap_points'], dtype=np.float32)
        import cv2
        H_mat, _ = cv2.findHomography(cam_arr, map_arr, cv2.RANSAC, 5.0)
        st.code(str(np.round(H_mat, 6)), language="python")
    else:
        st.warning("Homografie configuratie niet gevonden.")
