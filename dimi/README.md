# ⚽ Sports Vision & Tactical Mapping — Dimi Workspace
### Minimap Project, Model Training, Error Analyse & Streamlit Dashboard

**Student:** Dmitrij Tchervonisjenko  
**Team:** Git Push and Pray (Dmitrij Tchervonisjenko, Kobe Vandenberk, Cisse Vandeweyer, Brent Van Gelder)  
**Opleiding:** Thomas More Geel — 3e Jaar Deep Learning & Computer Vision  
**Hardware & Stack:** NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM), CUDA 12.4, PyTorch 2.6, Ultralytics YOLO11, OpenCV, Streamlit

---

## 🎯 Project Overzicht & Doelstelling

In deze map (`dimi/`) staat de volledige, zelfstandige end-to-end Computer Vision pipeline die een 20-seconden Full HD broadcast videostream ($1920 \times 1080$, 30 fps, 660 frames) van een Champions League wedstrijd (PSG vs. Inter Milan) omzet naar een **gesynchroniseerde 2D tactische minimap (radar)** met realtime telemetrie.

### 🌟 Kernprestaties & Resultaten
1. **The Small Object Problem Opgelost:** Training op $1280\text{ px}$ op de RTX 4070 GPU i.p.v. 640p bracht de baldetectie van $0.000$ (nooit gevonden) naar betrouwbare detectie en verhoogde de overall mAP@50 met **+25.7%** ($0.726$ vs $0.466$).
2. **100.0% Tracking Dekking (660 van de 660 frames):** De `InvariantBallTracker` elimineert elke teleportatie door spatiale gating, Dribble Carrier Hold en een verfijnde 20px penaltystip-filter.
3. **Sub-3-Pixel Homografie:** 10 RANSAC-gekalibreerde veldpunten met een gemiddelde herprojectiefout van slechts **$2.76\text{ px}$**.
4. **Interactief Dashboard:** 5-tab Streamlit dashboard met native web-compatibele H.264 videoweergave, 2D Gaussiaanse veldbezetting heatmap, IoU/NMS benchmark en complete examengids.

---

## 📂 Bestandenstructuur in `dimi/`

```
dimi/
├── README.md                           # Dit document: overzicht, mondelinge verdediging & instructies
├── PROJECT_RAPPORT_DEEP_LEARNING.md    # Uitgebreid academisch rapport & examengids (alle formules & theorie)
├── app.py                              # Interactief Streamlit dashboard (5 tabs, web-compatibele H.264 video)
├── minimap_pipeline.py                 # End-to-end video pipeline: YOLO -> Homografie -> InvariantBallTracker -> H.264
├── train_ball.py                       # GPU-trainingscript met 1280p resolutie, One-Cycle LR en close_mosaic
├── make_split.py                       # Tijdsgebaseerde train/val data split (154 frames) zonder data leakage
├── eval_ball.py                        # Validatiemetrics berekening (P, R, F1, mAP@50, mAP@50-95)
├── eval_iou_nms_experiments.py         # Systematische benchmark van 8 IoU & NMS parametercombinaties
├── iou_nms_benchmark.csv               # Volledige benchmark resultaten (winnende config: conf=0.025, iou=0.60)
├── homography_config.json              # 10 gekalibreerde ground-truth puntenparen & H matrix (RMSE = 2.76 px)
├── calibrate_homography.py             # Visuele interactieve calibratietool met zoomvenster en auto-preset
├── notebooks/                          # Drie schone, reproduceerbare Jupyter notebooks:
│   ├── 01_model_training_and_tuning.ipynb   # Architectuur, One-Cycle LR, Soft Labels, Vanishing Gradient & BatchNorm
│   ├── 02_error_analysis.ipynb              # Evaluatiemetrics, PR vs ROC, mAP ontrafeld & IoU/NMS benchmark
│   └── 03_homography_and_mapping.ipynb      # Projectieve meetkunde, 10-punts homografie & tracking stabilisatie
└── runs/
    ├── ball_yolo11_1280/               # Trainingsartefacten (results.csv, PR-curves, confusion matrix, best.pt)
    └── tactical_radar_output/          # Output video (synced_tactical_radar.mp4) & heatmap (spatial_occupancy_heatmap.png)
```

---

## 🚀 Snelle Start (Quickstart)

### 1. Activeer de Virtuele Omgeving
```powershell
Set-Location "C:\Users\dimit\Documents\01 - Thomas More\Thomas More Jaar 3\01 - Semester 1\Deep Learning\Project 2 - CV\CV-Challenge"
.\venv\Scripts\Activate.ps1
```

### 2. Start het Streamlit Dashboard
```powershell
streamlit run dimi/app.py
```
*Het dashboard opent direct in je browser met 5 tabs: Video & Radar, Veldbezetting Heatmap, IoU & NMS Benchmark, Deep Learning Examen Gids, en 10-Punt Homografie.*

### 3. De Volledige Pipeline Opnieuw Uitvoeren
```powershell
# 1. Dataset split bijwerken op basis van alle beschikbare frames
.\venv\Scripts\python dimi/make_split.py

# 2. End-to-end video pipeline uitvoeren (genereert automatisch H.264 video en heatmap)
.\venv\Scripts\python dimi/minimap_pipeline.py
```

---

## 🏆 MVP & Extra Mile Punten (Examen & Verdediging)

### 1. Targeted Data Annotation (25%)
* **Small Object Problem:** De voetbal ($16 \times 17\text{ px}$) verdwijnt bij standaard 640p resoluties. Training op $1280\text{ px}$ lost dit op.
* **Tijdsgebaseerde Split:** Split gemaakt op framenummer (Train $\le 494$, Val $\ge 525$) om spatiale data leakage te voorkomen.
* **Soft Labels (Label Smoothing $\epsilon = 0.10$):**
  In plaats van harde one-hot labels ($[1, 0, 0, 0, 0]$) die logits naar $\pm \infty$ forceren en leiden tot overconfidence en *entropy collapse*, mengen we het doel met een uniforme verdeling:
  $$y_{soft} = (1 - \epsilon) \cdot y_{hard} + \frac{\epsilon}{K}$$
  Hierdoor behoudt een wazige of deels bedekte bal (bij tackles of schoten) een gekalibreerde doelkans ($0.92$ vs $0.02$).

### 2. Model Fine-Tuning & Evaluatiemetrics (35%)
* **One-Cycle Policy (Leslie Smith):** Warm-up ("kneeling") voorkomt beschadiging van gewichten; hoge piekleersnelheid ontsnapt uit zadelpunten en lokale kuiltjes; cosine annealing landt in brede vlakke dalen.
* **AdamW & Momentum:** $v_t = \beta v_{t-1} + (1-\beta) g_t$. Rolt als een zware kogel over zadelpunten heen. Ontkoppelde gewichtsafname voorkomt verstoring.
* **Batch Size Ruis ($batch=8$):** Injecteert stochastische ruis die overfitten in smalle kuiltjes voorkomt.
* **Vanishing Gradient & ReLU/BatchNorm:** ReLU ($\max(0, x)$) heeft afgeleide 1, waardoor gradiënten niet uitsterven. BatchNorm recentreert en herschaalt tussenlagen.
* **Regularisatie & close_mosaic=10:** Dropout voorkomt "super-star neurons" door kennis te crowdsourcen. Mosaic schakelt 10 epochs voor het einde uit zodat het model fijnstemt op natuurlijke beelden.

### 3. Foutanalyse: PR-Curve vs. ROC & mAP Ontrafeld (20%)
* **Waarom NOOIT een ROC-curve bij Objectdetectie?**
  $$\text{FPR} = \frac{FP}{FP + TN}$$
  Omdat de achtergrond van een 1080p frame $>100.000$ negatieve patches bevat ($TN \to \infty$), is $\text{FPR} \approx 0$. De ROC-curve toont een bedrieglijk perfecte score ($\text{AUC} \approx 0.999$) en verbergt valse detecties. De **PR-curve** negeert $TN$ en evalueert zeldzame klassen eerlijk.
* **mAP ("Mean of Averages"):**
  *Average Precision (AP)* is de oppervlakte onder de PR-curve voor 1 klasse. *Mean Average Precision (mAP)* is het gemiddelde over alle $C$ klassen.
* **IoU & NMS Benchmark Winnaar:**
  `conf = 0.025, iou = 0.60` behaalt de hoogste mAP@50 (**0.7256**, +8.3% over default 0.6427) en de hoogste mAP@50-95 (**0.3086**).

### 4. Homografie & 2D Minimap Projectie (10%)
* **Planar Homography:** Projectieve transformatie ($H \in \mathbb{R}^{3\times 3}$) met 8 vrijheidsgraden.
* **Planar Assumption:** Voeten van spelers raken het gras (`x_center, y_max`).
* **10-Punt Ground Truth:** $RMSE = 2.76\text{ px}$ herprojectiefout via RANSAC.

### 5. Extra Mile: Invariant Tracking & Tactische Telemetrie (10%+)
* **InvariantBallTracker:**
  * Fysische spatiale gating ($d_{cam} \le 170\text{ px}$, $d_{pitch} \le 160\text{ px}$).
  * Dribble Carrier Hold bij voetocclusie.
  * 20px micro-penaltystip filter (100% filtering van de stip, 0% verlies van passes).
  * **Resultaat:** 660/660 frames (100.0%), 0 teleportaties, altijd puur rood.
* **Realtime Telemetrie:** Live balsnelheid in km/u ($v = \Delta d \cdot 10.8$), balbezit per team en 2D Gaussiaanse KDE veldbezetting heatmap.
