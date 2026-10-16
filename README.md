# ⚽ Computer Vision Challenge: Sports Vision & Tactical Mapping

## 👥 Team: Git Push and Pray
* **Dmitrij Tchervonisjenko**
* **Kobe Vandenberk**
* **Cisse Vandeweyer**
* **Brent Van Gelder**

**Opleiding:** Thomas More Mechelen – Jaar 3  
**Vak:** Deep Learning & Computer Vision  
**Sport:** Voetbal (Soccer)

---

## 🎯 Projectdoel (MVP)
Het doel van dit project is een end-to-end computer vision pipeline bouwen die een 20-seconden broadcast/tactische video van een voetbalwedstrijd omzet naar een synchrone **2D tactische radar (minimap)**.

De pipeline bestaat uit 4 kerncomponenten:
1. **Targeted Data Annotation (25%):**
   * Pre-trained YOLOv8/v11 als basis voor spelers en scheidsrechters.
   * Fine-tuning op de bal (de data gap!) met 40–80 lastige frames (motion blur, occlusies).
   * Strikte team-afspraken en labeling protocol (zie `docs/labeling_protocol.md`).
2. **Model Fine-Tuning & Error Analysis (35% + 20%):**
   * YOLO fine-tunen en evalueren met mAP@50 en mAP@50:95.
   * Grondige analyse van False Positives en False Negatives.
3. **Homografie & 2D Minimap Projectie (10%):**
   * Vaste veldlijnen/hoekpunten identificeren met OpenCV (`cv2.findHomography`).
   * Coördinaten van gedetecteerde spelers en de bal projecteren op een top-down 2D veldsjabloon.
4. **Lokale Streamlit Dashboard:**
   * Lokaal interactief dashboard waarin de originele video met bounding boxes naast de dynamische 2D minimap loopt.

---

## 📁 Mappenstructuur

```text
CV-Challenge/
├── .gitignore
├── requirements.txt
├── README.md
│
├── docs/                      # Documentatie & Labeling Protocols
│   └── labeling_protocol.md   # Officiële labeling regels (25% rubric)
│
├── data/                      # Dataset beheer
│   ├── raw_video/             # Originele .mp4 clips
│   ├── frames/                # Geëxtraheerde frames voor labeling
│   ├── dataset_yolo/          # Geannoteerde dataset (train/val)
│   └── pitch_template/        # 2D top-down veld visualisaties
│
├── scripts/                   # Automatisering & Utilities
│   ├── extract_frames.py      # Script om gevarieerde frames te pollen
│   └── generate_pitch.py      # Genereert een top-down voetbalveld
│
├── Final/                     # 🏆 Definitieve productiecode & app
│   ├── app/                   # Streamlit dashboard
│   ├── models/                # Beste getrainde YOLO weights (.pt)
│   └── notebooks/             # Schone, gevalideerde notebooks
│
├── brent/                     # Individuele werkmap Brent
├── cisse/                     # Individuele werkmap Cisse
├── dimi/                      # Individuele werkmap Dimi
└── kobe/                      # Individuele werkmap Kobe
```

---

## 🚀 Aan de slag

### 1. Omgeving opzetten
```bash
python -m venv venv
# Activeer op Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Frames extraheren uit je MP4 video
Plaats je 20-seconden videoclip in `data/raw_video/input_match.mp4` en run:
```bash
python scripts/extract_frames.py --video data/raw_video/input_match.mp4 --num_frames 60
```

### 3. Streamlit Dashboard starten (zodra de app klaar is)
```bash
streamlit run Final/app/app.py
```
