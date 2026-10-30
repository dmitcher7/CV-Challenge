# ⚽ Dimi - Ball & Player Vision Model (RTX 4070 GPU)

In deze werkmap staat het geoptimaliseerde YOLO model dat specifiek is afgesteld op het **"Small Object Problem"** (het detecteren van de kleine voetbal).

---

## 🎯 Belangrijkste Doorbraak: Bal Detectie

In eerdere tests (Kobe & Brent) werd het model getraind op resolutie `imgsz=640`. Omdat een voetbal in Full HD ($1920 \times 1080$) slechts $\approx 16 \times 17$ pixels groot is, werd de bal bij 640p verkleind naar $\approx 5$ pixels—kleiner dan het fijnste detectierooster van YOLO—waardoor het model de bal **$0\times$ detecteerde (mAP@50 = 0.000)**.

Met Dimi's **NVIDIA GeForce RTX 4070 Laptop GPU** is getraind op **`imgsz=1280`** met `close_mosaic=10`. Hierdoor blijft de bal groot en scherp genoeg.

### 📊 Vergelijking: Baseline (CPU 640p) vs. Dimi's Model (GPU 1280p)

| Klasse | Baseline mAP@50 (Kobe/Brent 640p) | **Dimi mAP@50 (RTX 4070 1280p)** | Verbetering |
| :--- | :---: | :---: | :---: |
| **⚽ ball** | **0.000** (nooit gevonden) | **0.035** (precision 0.126) | **✅ Doorbraak: bal wordt gedetecteerd!** |
| **player_A** | 0.863 | **0.944** | +8.1% |
| **player_B** | 0.667 | **0.859** | +19.2% |
| **player_B_goal_keeper** | 0.781 | **0.953** | +17.2% |
| **referee** | 0.016 | **0.822** | **+80.6%** |
| **Gemiddeld (ALL)** | 0.466 | **0.723** | **+25.7%** |

---

## 📁 Bestanden in deze map

* `common.py`: Paden en automatische GPU/CUDA detectie.
* `make_split.py`: Splitst de 100 Roboflow frames in een strikte Train (68) / Val (32) split zonder leakage.
* `train_ball.py`: Traint YOLO11s op de GPU met `imgsz=1280` en `batch=8`.
* `eval_ball.py`: Berekent de evaluatiemetrics (P, R, mAP50, mAP50-95) en logt naar `experiments.csv`.
* `predict_video.py`: Draait inferentie op `data/raw_video/CV-video.mp4` en exporteert geannoteerde video.
* `experiments.csv`: Track record van alle runs.
* `runs/ball_yolo11_1280/weights/best.pt`: De getrainde weights.

---

## 🚀 Commando's om uit te voeren

```powershell
# 1. Dataset split controleren/genereren
.\venv\Scripts\python dimi/make_split.py

# 2. Model hertrainen op GPU (indien gewenst)
.\venv\Scripts\python dimi/train_ball.py --epochs 35 --imgsz 1280 --batch 8

# 3. Model evalueren op validatie-data
.\venv\Scripts\python dimi/eval_ball.py

# 4. Inferentie draaien op de wedstrijdvideo
.\venv\Scripts\python dimi/predict_video.py
```
