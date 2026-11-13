# Sports Vision & Tactical Mapping — Dimi Workspace
### Minimap Project, Model Training, Error Analyse & Streamlit Dashboard

**Student:** Dmitrij Tchervonisjenko  
**Team:** Git Push and Pray (Dmitrij Tchervonisjenko, Kobe Vandenberk, Cisse Vandeweyer, Brent Van Gelder)  
**Opleiding:** Thomas More Geel — 3e Jaar Deep Learning & Computer Vision  
**Hardware & Stack:** NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM), CUDA 12.4, PyTorch 2.6, Ultralytics YOLO11, OpenCV, Streamlit

---

## Project Overzicht & Doelstelling

In deze map (`dimi/`) staat de volledige, zelfstandige end-to-end Computer Vision pipeline die een 20-seconden Full HD broadcast videostream ($1920 \times 1080$, 30 fps, 660 frames) van een Champions League wedstrijd (PSG vs. Inter Milan) omzet naar een **gesynchroniseerde 2D tactische minimap (radar)** met realtime telemetrie.

### Kernprestaties & Resultaten
1. **The Small Object Problem Opgelost:** Training op $1280\text{ px}$ op de RTX 4070 GPU i.p.v. 640p bracht de baldetectie van $0.000$ (nooit gevonden) naar betrouwbare detectie en verhoogde de overall mAP@50 met **+25.7%** ($0.726$ vs $0.466$).
2. **Multi-Scale Pyramid Heads:** P3 Head ($160 \times 160$ grid) detecteert de kleine voetbal ($16 \times 17\text{ px}$), P4 detecteert middelgrote spelers, en P5 detecteert nabije spelers en doelgebieden.
3. **100.0% Tracking Dekking (660 van de 660 frames):** De `InvariantBallTracker` elimineert elke teleportatie door spatiale gating, Dribble Carrier Hold en een verfijnde 20px penaltystip-filter.
4. **Sub-3-Pixel Homografie:** 10 RANSAC-gekalibreerde veldpunten met een gemiddelde herprojectiefout van slechts **$2.76\text{ px}$**.
5. **Interactief Dashboard:** Professioneel 6-tab Streamlit dashboard met native web-compatibele H.264 videoweergave, 2D Gaussiaanse veldbezetting heatmap, IoU/NMS benchmark en complete examengids.

---

## Schone & Geordende Mappenstructuur in `dimi/`

De map `dimi/` is strikt geordend volgens modulaire best practices:

```
dimi/
├── README.md                           # Dit document: overzicht, mondelinge verdediging & structuur
├── PROJECT_RAPPORT_DEEP_LEARNING.md    # Uitgebreid academisch rapport & examengids (Deel 1 t/m Deel 5)
├── app.py                              # Interactief Streamlit dashboard (6 tabs, web-compatibele H.264 video)
├── run_dashboard.bat                   # 1-klik starter voor het Streamlit dashboard
├── run_homography_tool.bat             # 1-klik starter voor de interactieve calibratietool
│
├── notebooks/                          # DE HOOFDMAP MET NOTEBOOKS (hoofdcode & analyse-experimenten)
│   ├── 01_model_training_and_tuning.ipynb   # Multi-Scale Backbone/Neck, One-Cycle LR, Soft Labels
│   ├── 02_error_analysis.ipynb              # PR vs ROC, mAP ontrafeld, IoU/NMS Winnaar
│   └── 03_homography_and_mapping.ipynb      # 10-punts homografie, FCN, U-Net & Invariant Tracker
│
├── src/                                # De onderliggende Python scripts & pipeline modules
│   ├── common.py                       # Centrale paden & automatische GPU/CUDA detectie
│   ├── minimap_pipeline.py             # End-to-end video pipeline: YOLO -> Homografie -> Tracker -> H.264
│   ├── train_ball.py                   # GPU-trainingscript met 1280p resolutie, One-Cycle LR en close_mosaic
│   ├── make_split.py                   # Tijdsgebaseerde train/val data split (154 frames)
│   ├── eval_ball.py                    # Validatiemetrics berekening (P, R, F1, mAP@50, mAP@50-95)
│   ├── eval_iou_nms_experiments.py     # Systematische benchmark van 8 IoU & NMS parametercombinaties
│   ├── calibrate_homography.py         # Visuele interactieve calibratietool met zoomvenster
│   ├── predict_video.py                # Standalone video voorspeller
│   └── generate_notebooks.py           # Generator voor de 3 reproduceerbare notebooks
│
├── config/                             # Configuratiebestanden, matrices en benchmarkdata
│   ├── homography_config.json          # 10 gekalibreerde referentiepunten & H matrix (RMSE = 2.76 px)
│   ├── data_dimi.yaml                  # YOLO dataset configuratie
│   ├── iou_nms_benchmark.csv           # Benchmark resultaten (8 geteste configuraties)
│   └── experiments.csv                 # Logboek van eerdere experimentele runs
│
├── data/                               # Dataset lijsten en referentiebeelden
│   ├── train.txt                       # Lijst met trainingspaden (104 beelden)
│   ├── val.txt                         # Lijst met validatiepaden (50 beelden)
│   ├── sample_field.jpg                # Referentieframe van de wedstrijd
│   └── gecalibreerde_punten_visueel.jpg # Visuele verificatie van calibratiepunten
│
├── runs/                               # Alle trainingsartefacten en media-outputs
│   ├── ball_yolo11_1280/               # Modelgewichten (weights/best.pt), curves & confusion matrix
│   └── tactical_radar_output/          # Output video (synced_tactical_radar.mp4) & heatmap
│
└── labels/                             # Roboflow annotatielabels
```

---

## Snelle Start (Quickstart)

### 1. Activeer de Virtuele Omgeving
```powershell
Set-Location "C:\Users\dimit\Documents\01 - Thomas More\Thomas More Jaar 3\01 - Semester 1\Deep Learning\Project 2 - CV\CV-Challenge"
.\venv\Scripts\Activate.ps1
```

### 2. Start het Streamlit Dashboard
```powershell
streamlit run dimi/app.py
```
*Of dubbelklik simpelweg op `dimi/run_dashboard.bat`.*

### 3. Pipeline of Scripts Uitvoeren
```powershell
# Volledige minimap video genereren (inclusief automatische H.264 conversie):
.\venv\Scripts\python dimi/src/minimap_pipeline.py

# Dataset split updaten:
.\venv\Scripts\python dimi/src/make_split.py

# Model evalueren op validatieset:
.\venv\Scripts\python dimi/src/eval_ball.py
```

---

## MVP & Extra Mile Punten (Examen & Verdediging)

### 1. Data & Annotation Strategy (25%)
- **Small Object Problem:** De voetbal ($16 \times 17\text{ px}$) verdwijnt bij standaard 640p resoluties. Training op $1280\text{ px}$ lost dit op.
- **Tijdsgebaseerde Split:** Split gemaakt op framenummer (Train $\le 494$, Val $\ge 525$) om data leakage tussen opeenvolgende videoframes te voorkomen.
- **Soft Labels (Label Smoothing $\epsilon = 0.10$):**
  $$y_{soft} = (1 - \epsilon) \cdot y_{hard} + \frac{\epsilon}{K}$$
  Voorkomt overconfidence en *entropy collapse* bij bewegingsonscherpte en tackles.

### 2. Modeling & Multi-Scale Architecture (35%)
- **Backbone (C3, C4, C5):** C3 vangt fijne details ($160\times 160$ grid bij 1280p), C4 medium features, C5 rijke semantische context.
- **FPN/PAN Neck:** Upsamplet en fuseert diepe context met hoge spatiale resolutie.
- **Pyramid Heads:** P3 Head detecteert onze kleine voetbal ($16\times 17\text{ px}$), P4 middelgrote spelers, P5 grote nabije spelers.
- **One-Cycle Policy (Leslie Smith):** Kneeling warm-up $\to$ exploratiepiek $\to$ cosine annealing cooldown.
- **AdamW & Momentum:** $v_t = \beta v_{t-1} + (1-\beta)g_t$. Traagheid dendert over zadelpunten heen; ontkoppelde weight decay voorkomt gewichtsvervorming.
- **Vanishing Gradient, ReLU & BatchNorm:** ReLU ($\max(0, x)$) heeft afgeleide 1; BatchNorm neutraliseert Internal Covariate Shift.
- **Regularisatie & close_mosaic=10:** Dropout voorkomt super-star neuronen; Mosaic schakelt 10 epochs voor het einde uit voor scherpe kaders.

### 3. Foutanalyse: PR-Curve vs. ROC & mAP Ontrafeld (20%)
- **Waarom NOOIT een ROC-curve bij Objectdetectie?**
  $$\text{FPR} = \frac{FP}{FP + TN}$$
  Omdat er $>100.000$ negatieve achtergrond-patches zijn ($TN \to \infty$), is $\text{FPR} \approx 0$. De ROC-curve toont een bedrieglijk perfecte $\text{AUC} \approx 0.999$. De **PR-curve** negeert $TN$ en evalueert zeldzame klassen eerlijk.
- **mAP ("Mean of Averages"):** Het gemiddelde (*mean*) over alle klassen van de Average Precision (AUC-PR).
- **IoU & NMS Benchmark Winnaar:** `conf=0.025, iou=0.60` behaalt **mAP@50 = 0.7256** (+8.3% over default 0.6427) en **mAP@50-95 = 0.3086**.

### 4. Homografie & 2D Minimap (10%)
- **Planar Homografie:** $H \in \mathbb{R}^{3\times 3}$ met 8 vrijheidsgraden.
- **Planar Assumption:** Voeten van spelers raken het gras (`x_center, y_max`).
- **10-Punt Ground Truth:** $RMSE = 2.76\text{ px}$ herprojectiefout via RANSAC.

### 5. Extra Mile: Invariant Tracking & Telemetrie (85%+ categorie)
- **InvariantBallTracker:** 100% video tracking (660/660 frames), Dribble Carrier Hold bij occlusie en 20px micro-penaltystip filter. 0 teleportaties.
- **Realtime Telemetrie:** Live balsnelheid in km/u ($v = \Delta d \cdot 10.8$), balbezitverdeling (PSG vs. Inter) en 2D Gaussiaanse KDE heatmap.
