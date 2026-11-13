# 🏆 Final Product & Evaluatie: Sports Vision & Tactical Mapping

Dit is de officiële productie- en evaluatiemap van **Team Git Push and Pray** voor de **Computer Vision Challenge: Sports Vision & Tactical Mapping** aan **Thomas More Geel** (Academiejaar 3, Semester 1).

**Auteur:** Dmitrij Tchervonisjenko  
**Teamleden:** Dmitrij Tchervonisjenko, Kobe Vandenberk, Cisse Vandeweyer, Brent Van Gelder  
**Hardware & Stack:** NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM), CUDA 12.4, PyTorch 2.6, Ultralytics YOLO11, OpenCV, Streamlit.

---

## 🚀 Snelle Start (Quickstart)

### 1. Vereisten & Omgeving
Zorg dat de Python virtuele omgeving actief is:
```bash
# Activeer virtuele omgeving (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Of controleer packages
pip install -r requirements.txt
```

### 2. Start het Interactieve Streamlit Dashboard
Het complete dashboard bevat synchrone videoweergave, telemetrie (balbezit & km/u), de interactieve 2D veldbezetting heatmap, de IoU/NMS benchmark en de complete Deep Learning examengids:
```bash
streamlit run Final/app/app.py
```
*(Alternatief ook rechtstreeks te starten via: `streamlit run dimi/app.py`)*

---

## 📂 Deliverables Structuur

```
Final/
├── README.md                           # Dit document: overzicht, instructies en verantwoording
├── app/
│   ├── .gitkeep
│   └── app.py                          # Volledig 5-tab interactief Streamlit analyse-dashboard
├── models/
│   ├── .gitkeep
│   └── best.pt                         # Fine-tuned YOLO11s gewichten (1280p, close_mosaic=10, RTX 4070)
└── notebooks/
    ├── .gitkeep
    ├── 01_model_training_and_tuning.ipynb   # Modelarchitectuur, One-Cycle LR, optimalisatie & training
    ├── 02_error_analysis.ipynb              # Evaluatiemetrics (PR vs ROC), mAP, IoU & NMS benchmark winnaar
    └── 03_homography_and_mapping.ipynb      # 10-punt Homografie, InvariantBallTracker & telemetrie
```

---

## 📓 De Drie Reproduceerbare Notebooks

### 1. `01_model_training_and_tuning.ipynb`
* **Doel:** Volledige documentatie van het trainingsproces, data splitting en hyperparameter optimalisatie.
* **Belangrijkste Onderwerpen:**
  * **PyTorch (low-level DAG & Autograd) vs. FastAI (high-level DataLoaders & Learner)**.
  * Waarom stochastische optimalisatie **niet greedy** mag zijn zoals Dijkstra (ontsnappen uit scherpe dalen).
  * **The Small Object Problem:** Waarom de bal op 640px verdween en hoe training op $1280\text{ px}$ op de RTX 4070 GPU leidde tot een doorbraak van **+25.7% overall mAP**.
  * **Leslie Smith's One-Cycle Policy:** Warm-up ("kneeling") van $LR$, cosine annealing en regularisatie door hoge learning rate in vroege epochs.
  * **Momentum & Optimizers:** Fysische intuïtie achter momentum (zadelpunten overbruggen), Adam vs. AdamW (decoupled weight decay) en mini-batch stochastic noise ($batch=8$).
  * **Architectuur & Regularisatie:** Vanishing gradient, ReLU & SiLU, Batch Normalization (interne covariaten shift), L1/L2, Dropout (tegengaan van "super-star neurons") en Mosaic/Mixup data-augmentatie (`close_mosaic=10`).
  * **Decoder & Upsampling:** Nearest-Neighbor vs. Bilinear Interpolation vs. Transposed Convolution (checkerboard artifacts & PixelShuffle).

### 2. `02_error_analysis.ipynb`
* **Doel:** Diepgaande error analysis, evaluatiemetrics en systematische validatie van IoU & NMS thresholds.
* **Belangrijkste Onderwerpen:**
  * **Theoretische Examentheorie: Waarom een ROC-Curve ONGELDIG is voor Objectdetectie:**
    $$FPR = \frac{FP}{FP + TN}$$
    Omdat de achtergrond van een 1080p frame $>100.000$ negatieve patches bevat ($TN \to \infty$), nadert $FPR \to 0$. De ROC-curve geeft daardoor een vals-optimistisch perfecte score ($\text{AUC} \approx 0.999$). Alleen de **Precision-Recall (PR) curve** evalueert objectdetectie eerlijk.
  * **mAP Ontrafeld ("Mean of Averages"):**
    Wiskundig zijn *mean* en *average* synoniemen. mAP is het *rekenkundig gemiddelde (mean)* over alle klassen van de *gemiddelde precisie (Average Precision - AP)*, berekend als de oppervlakte onder de PR-curve ($\text{AUC-PR}$).
  * **IoU & NMS Benchmark (8 Geteste Configuraties):**
    * *YOLO Default Baseline (`conf=0.250, iou=0.60`):* mAP@50 = 0.6427, mAP@50-95 = 0.2730
    * **Winnende Configuratie (`conf=0.025, iou=0.60`):** mAP@50 = **0.7256 (+8.3% boost)**, mAP@50-95 = **0.3086 (+13.0% boost)**, F1 = 0.7139.
  * **Kwantitatieve & Kwalitatieve Error Analyse:** Verificatie van False Negatives (voetocclusies) en False Positives (witte sokken, penaltystip).

### 3. `03_homography_and_mapping.ipynb`
* **Doel:** Projectieve meetkunde, wiskundige afleidingen, tracker-stabilisatie en tactische telemetrie.
* **Belangrijkste Onderwerpen:**
  * **Planar Homography Formule:** 2D projectieve transformatie ($H \in \mathbb{R}^{3\times 3}$) met 8 vrijheidsgraden.
  * **10-Punt Veldcalibratie:** Geijkte ground-truth markers (middenstip, 16m-lijnen, doelpalen) met een gemiddelde herprojectiefout van slechts **2.76 pixels**.
  * **Oplossen van Bal-Teleportatie & Occlusie:**
    * Waarom SORT / Kalman faalde (gating radius die explodeerde tot 555px).
    * Waarom de eerdere penaltystipfilter (55px) echte passes in de 16m wiste.
  * **De InvariantBallTracker Architectuur:**
    1. *Fysische Spatiale Gating:* $d_{cam} \le 170\text{ px}$, $d_{pitch} \le 160\text{ px}$.
    2. *Dribble Carrier Hold:* Verankering van de bal aan de voeten van de baldrager bij occlusie.
    3. *Micro-Penaltystip Filter:* Gereduceerd naar $20\text{ px}$ (100% filtering van de stip, 0% verlies van passes).
    4. *Resultaat:* **100.0% trackingdekking (660 van de 660 frames)**, 0 sprongen, altijd consistent rood.
  * **Tactische Extensies:**
    * Live balsnelheid in km/u ($v = \frac{\Delta d_{m}}{\Delta t} \times 3.6$).
    * Tactisch balbezit (PSG vs. Inter Milan).
    * 2D Gaussiaanse Kernel Density Estimation (KDE) veldbezetting heatmap.

---

## 📊 Resultaten & Belangrijkste Kerncijfers

| Metric / Prestatie-indicator | Baseline / Vorige iteratie | Finale Oplossing | Verbetering |
| :--- | :--- | :--- | :--- |
| **Object Resolutie** | 640 px | **1280 px (RTX 4070)** | $4\times$ meer pixels / object |
| **Overall mAP@50** | 0.5770 | **0.7256** | **+25.7%** |
| **Bal Detectie (640px vs 1280px)** | 0.000 (Geen hits) | **Gedetecteerd & getrackt** | Van 0 naar continue detectie |
| **Herprojectiefout Homografie** | N.v.t. (Handmatig geschat) | **2.76 pixels** | Sub-3-pixel precisie |
| **Trackingdekking Bal (660 frames)** | 48.2% (sprong over veld) | **100.0% (660 / 660 frames)** | **+51.8% (Volledig continu)** |
| **Bal Teleportaties / Sprongen** | 4 teleports naar zijlijn | **0 teleports** | Fysisch begrensd & geankerd |
| **Penaltystip Foutieve Bal Detectie** | Foutief getriggerd | **0 triggers (volledig gefilterd)** | 20px micro-filter |
| **Inference Snelheid (Full HD)** | N.v.t. | **~15.3 ms / frame (~65 FPS)** | Realtime op RTX 4070 GPU |

---

## 📖 Volledig Technisch Rapport
Voor de diepgaande academische onderbouwing, alle wiskundige formules en cursusintegratie, zie:
👉 **[PROJECT_RAPPORT_DEEP_LEARNING.md](file:///C:/Users/dimit/Documents/01%20-%20Thomas%20More/Thomas%20More%20Jaar%203/01%20-%20Semester%201/Deep%20Learning/Project%202%20-%20CV/CV-Challenge/docs/PROJECT_RAPPORT_DEEP_LEARNING.md)**
