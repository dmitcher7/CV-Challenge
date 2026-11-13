# Sports Vision & Tactical Mapping Challenge
## Technisch Projectrapport & Examencompendium Deep Learning & Computer Vision

**Student:** Dmitrij Tchervonisjenko  
**Team:** Git Push and Pray (Dmitrij Tchervonisjenko, Kobe Vandenberk, Cisse Vandeweyer, Brent Van Gelder)  
**Opleiding:** Thomas More Geel – Academiejaar 3 (Semester 1)  
**Vak:** Deep Learning & Computer Vision  
**Hardware & Stack:** NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM), CUDA 12.4, PyTorch 2.6, Python 3.13, Ultralytics YOLO26, OpenCV, Streamlit  

---

## Inhoudsopgave
1. [Management Samenvatting & Project Scope](#1-management-samenvatting--project-scope)
2. [DEEL 1: Deep Learning Fundamentele Principes & Architecturen](#2-deel-1-deep-learning-fundamentele-principes--architecturen)
   - 1.1 Van Big Data naar de Latente Ruimte
   - 1.2 De Wiskunde van Leren (Dot Product & Activaties)
   - 1.3 Trainen & Optimaliseren (Non-Greedy vs. Dijkstra, Momentum, AdamW)
   - 1.4 Internal Covariate Shift & Batch Normalization
   - 1.5 Overfitting, Regularisatie & Dropout (Anti-Superstar Neuronen)
   - 1.6 Hard Labels vs. Soft Labels & Label Smoothing
3. [DEEL 2: Van Klassieke Computer Vision naar CNN's](#3-deel-2-van-klassieke-computer-vision-naar-cnns)
   - 2.1 Het Falen van Klassieke Vision (Verschil vs. Gelijkenis)
   - 2.2 De CNN Revolutie (AlexNet 2012 & Hierarchische Filters)
   - 2.3 Model Zoo Evolutie: VGG, GoogLeNet (1x1 Convolutie) & ResNet (Residuals)
   - 2.4 Evaluatiemetrics: PR-Curve vs. ROC-Curve & F1-Score
4. [DEEL 3: Object Detection (Multi-Scale, FPN Neck & Pyramid Heads)](#4-deel-3-object-detection-multi-scale-fpn-neck--pyramid-heads)
   - 3.1 Two-Stage (Faster R-CNN) vs. One-Stage (YOLO)
   - 3.2 De Multi-Scale Architectuur: Backbone (C3, C4, C5)
   - 3.3 De FPN/PAN Neck: Upsampling, Merging & Betekenis-Overdracht
   - 3.4 De 3 Predictie Heads: P3 (Small Ball), P4 (Medium) & P5 (Large)
   - 3.5 Post-Processing: Non-Maximum Suppression (NMS) & Distribution Focal Loss (DFL)
   - 3.6 mAP Ontrafeld ("Mean of Averages" - mAP@50 vs. mAP@50-95)
5. [DEEL 4: Semantische Segmentatie & 2D Homografie](#5-deel-4-semantische-segmentatie--2d-homografie)
   - 4.1 Semantic ("Stuff") vs. Instance ("Things") vs. Panoptic
   - 4.2 Fully Convolutional Networks (FCNs): 1x1 Convoluties ipv Flatten/Dense
   - 4.3 U-Net Architectuur: Transposed Convolutions, Checkerboard Artifacts & Skip Connections
   - 4.4 Verliesfuncties: Softmax vs. Argmax & Cross-Entropy + Shape Overlap (Dice / Soft IoU)
   - 4.5 2D Planar Homografie: Wiskundige Afleiding & Planar Assumption
6. [DEEL 5: Diepte, Zelf-Gesuperviseerd Leren & Foundation Models](#6-deel-5-diepte-zelf-gesuperviseerd-leren--foundation-models)
   - 5.1 Monocular Depth Estimation
   - 5.2 Zelf-Gesuperviseerd Pretrainen zonder Labels: MAE & DINO (Teacher-Student EMA)
   - 5.3 Multimodale Foundation Models: CLIP (Contrastive Latent Space) & SAM (Promptable Vision)
7. [DEEL 6: Praktische Experimenten, IoU/NMS Benchmark & Winnaar](#7-deel-6-praktische-experimenten-iou-nms-benchmark--winnaar)
8. [DEEL 7: Sports Vision Pipeline, Invariant Tracking & Telemetrie](#8-deel-7-sports-vision-pipeline-invariant-tracking--telemetrie)
9. [Conclusie & Verantwoording](#9-conclusie--verantwoording)

---

## 1. Management Samenvatting & Project Scope

Het project **Sports Vision & Tactical Mapping** transformeert een Full HD broadcast videostream ($1920 \times 1080$, 30 fps, 660 frames) van een Champions League wedstrijd (PSG vs. Inter Milan) naar een realtime **2D tactische minimap (radar)**.

### De 3 Grote Technische Uitdagingen:
1. **The Small Object Problem:** Een voetbal op 1080p broadcast video beslaat slechts $\approx 16 \times 17$ pixels ($<0.015\%$ van het beeldoppervlak). Bij standaard resolutie ($640\text{ px}$) kromp de bal naar 5 pixels en verdween hij na downsampling (0 detecties, $\text{mAP@50} = 0.000$).
2. **Occlusie & Spelersbenen:** Bij dribbels en duels verdwijnt de bal visueel tussen voeten, waardoor detectie-confidence keldert en naïeve trackers de bal verliezen of teleporteren.
3. **Perspectief en Veldmarkeringen:** Camerapanning en zoom vervormen veldlijnen; statische veldmarkeringen (zoals de witte penaltystip) lijken op een voetbal.

### Onze Gevalideerde Oplossingen:
- **1280p GPU-Training op NVIDIA RTX 4070:** Behoudt $4\times$ meer pixels per object. Overall mAP@50 steeg met **+25.7%** ($0.726$ vs $0.466$) en de bal werd continu gedetecteerd.
- **10-Punt RANSAC Homografie:** Berekende 2D projectie met een gemiddelde herprojectiefout van slechts **$2.76\text{ pixels}$**.
- **InvariantBallTracker:** Fysische spatiale gating ($d_{cam} \le 170\text{ px}$, $d_{pitch} \le 160\text{ px}$), Dribble Carrier Hold bij voetocclusie en een $20\text{ px}$ micro-penaltystip filter. **Resultaat: 660 van de 660 frames gevolgd (100.0% trackingdekking) zonder teleportaties.**

---

## 2. DEEL 1: Deep Learning Fundamentele Principes & Architecturen

### 1.1 Van Big Data naar de Latente Ruimte
Deep Learning drijft op de **5 V's van Big Data**:
- **Volume:** Enorme hoeveelheden trainingsframes.
- **Velocity:** Realtime videoverwerking op 30 frames per seconde.
- **Variety:** Wisselende camerastanden, lichtinvallen en speelstijlen.
- **Veracity:** Betrouwbare annotatie (correct omgaan met occlusie en bewegingsonscherpte).
- **Value:** Tactische meerwaarde voor coaches (balsnelheid in km/u, veldbezetting).

Bij Big Data schalen we **horizontaal (scale out)** door meerdere GPU/compute-nodes parallel te schakelen, in plaats van **verticaal (scale up)** door simpelweg één grotere CPU te kopen.

Het doel van een diep netwerk is het afbeelden van ruwe pixels naar een **Latente Ruimte (Latent Space)**: een gecomprimeerde laag-dimensionale representatie waarin enkel de semantisch relevante eigenschappen behouden blijven. Een **Encoder** comprimeert de invoer naar een 'bottleneck', waarna een **Decoder** of detectiekop de gewenste objecten lokaliseert.

### 1.2 De Wiskunde van Leren (Dot Product & Activaties)
Elk artificieel neuron berekent een lineair dot product:
$$z = \sum_{i=1}^n w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$
Om niet-lineaire relaties te leren, passeert $z$ een activatiefunctie:
- **ReLU ($\max(0, z)$):** Heeft afgeleide $1$ voor $z > 0$. Lost het vanishing gradient probleem op, maar kent het risico op *dying ReLU* (wanneer een neuron permanent negatief wordt en de gradiënt 0 blijft).
- **Leaky ReLU ($\max(\alpha z, z)$ met $\alpha \approx 0.01$):** Voorkomt dying ReLU door een kleine constante helling voor negatieve inputs toe te laten.
- **Sigmoid / Tanh:** Drukken waarden plat tussen $[0, 1]$ of $[-1, 1]$. Buiten het centrum nadert de afgeleide naar $0$, waardoor gradiënten in diepe netwerken uitsterven.

### 1.3 Trainen & Optimaliseren (Non-Greedy vs. Dijkstra, Momentum, AdamW)
Het netwerk leert via de cyclus: Forward pass $\to$ Loss berekening $\to$ Autograd backpropagation $\to$ Optimizer gewichtsupdate.

**Waarom mag een optimizer NIET greedy zijn zoals Dijkstra?**  
Dijkstra kiest op elk punt altijd de direct kortste afstand (*greedy choice*). In een hoog-dimensionaal niet-convex verliesoppervlak met miljoenen parameters leidt zo'n gulzige strategie direct tot vastlopen in het allereerste scherpe lokale minimum (*sharp minimum*) of een zadelpunt (*saddle point* waar $\nabla L \approx 0$). Stochastic Gradient Descent met mini-batches schat de globale gradiënt stochastisch, waardoor het model generaliseert naar brede, vlakke minima.

**Momentum & AdamW:**
- Klassiek SGD stuitert heen en weer tussen steile dalwanden. **Momentum** voegt fysische traagheid toe:
  $$v_t = \beta v_{t-1} + (1 - \beta) g_t, \quad w_{t+1} = w_t - \eta v_t$$
  Hierdoor rolt de optimizer als een zware stalen kogel over kleine bulten en zadelpunten heen.
- **AdamW** berekent adaptieve leersnelheden per parameter en ontkoppelt gewichtsverval (*weight decay*) van de gradiëntupdate, wat gewichtsvervorming voorkomt.

**Leslie Smith's One-Cycle Policy & Kneeling:**
1. *Warm-up ('Kneeling'):* Start met een zeer kleine $\eta_{\min}$ om willekeurig geïnitialiseerde gewichten niet te vernietigen.
2. *Exploratie (Piek):* Hoge $\eta_{\max}$ dwingt het model uit lokale kuiltjes te springen.
3. *Cosine Annealing:* Geleidelijke cosinus-afname naar 0 voor stabiele convergentie in het vlakke dal.

### 1.4 Internal Covariate Shift & Batch Normalization
Omdat de gewichten van eerdere lagen bij elke stap wijzigen, verandert de verdeling van de invoer voor diepere lagen voortdurend (*Internal Covariate Shift*).
**Batch Normalization** trekt het batch-gemiddelde af en deelt door de standaardafwijking:
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$
Dit centreert en stabiliseert de gradiënten over alle lagen, wat training drastisch versnelt.

### 1.5 Overfitting, Regularisatie & Dropout (Anti-Superstar Neuronen)
Wanneer training loss daalt maar validation loss stijgt, overfit het model.
- **L2 Regularisatie (Weight Decay):** $\frac{1}{2} \lambda \sum w_i^2$ bestraft excessief grote gewichten.
- **Dropout:** Schakelt tijdens training willekeurig neuronen uit ($20-50\%$). Dit voorkomt de vorming van dominante "super-star neuronen" en dwingt het netwerk om kennis te *crowdsourcen*: redundante, robuuste features leren.
- **Early Stopping:** Training staken zodra het validatieverlies niet meer verbetert.
- **close_mosaic=10:** Schakelt samengestelde mozaïek-beelden 10 epochs voor het einde uit zodat het model fijnstemt op natuurlijke, onvervormde scènes.

### 1.6 Hard Labels vs. Soft Labels & Label Smoothing
- **Hard Labels:** $y_{hard} = [1, 0, 0, 0, 0]$. Forceert logits naar $\pm \infty$ en veroorzaakt *overconfidence*. Bij een wazige of deels occluded bal klapt de detectiekans in naar 0 (*entropy collapse*).
- **Soft Labels (Label Smoothing $\epsilon = 0.10$):**
  $$y_{soft} = (1 - \epsilon) \cdot y_{hard} + \frac{\epsilon}{K}$$
  Voor $K=5$ klassen wordt het doel voor de bal $0.92$ en overige klassen $0.02$. Dit behoudt gekalibreerde kansen bij bewegingsonscherpte en tackles.

---

## 3. DEEL 2: Van Klassieke Computer Vision naar CNN's

### 2.1 Het Falen van Klassieke Vision (Verschil vs. Gelijkenis)
Vóór deep learning probeerde men objecten te vinden met wiskundige regels:
- **Op basis van verschil:** Intensiteitsdrempels (Otsu thresholding) en gradiëntfilters (Sobel, Canny edge detection).
- **Op basis van gelijkenis:** Region growing (Watershed) en kleurclustering (K-Means, Mean Shift).

**Waarom faalt klassieke vision bij voetbal?**  
*Pixels zijn geen objecten.* Witte sokken van spelers, kalklijnen op het veld en de penaltystip hebben exact dezelfde kleur en intensiteit als de bal. Klassieke filters missen semantisch contextbegrip en falen volledig bij occlusie, wisselende belichting en schaduwen.

### 2.2 De CNN Revolutie (AlexNet 2012)
In plaats van handmatige filters leert een Convolutional Neural Network zijn eigen filters via backpropagation:
- Vroege lagen: ruwe randen en kleurovergangen.
- Middelste lagen: texturen, strepen en patronen.
- Diepe lagen: semantische concepten (lichaamsdelen, balvorm).  
AlexNet combineerde in 2012 convolutionele lagen met GPU-rekenkracht, ReLU activaties en de ImageNet dataset.

### 2.3 Model Zoo Evolutie: VGG, GoogLeNet & ResNet
- **VGG (2014):** Toonde aan dat het stapelen van kleine $3\times 3$ convoluties effectiever is dan grote filters ($7\times 7$), omdat twee gestapelde $3\times 3$ lagen hetzelfde receptieve veld ($5\times 5$) bestrijken met $28\%$ minder parameters.
- **GoogLeNet / Inception (2014):** Introduceerde de **$1\times 1$ convolutie als bottleneck**. Dit is een wiskundige techniek om kanaaldiepte te reduceren (*dimensionality reduction*) zonder ruimtelijke afmetingen te verliezen, wat gigantisch veel rekenkracht bespaart.
- **ResNet (2015):** De absolute standaard. Introduceerde **skip connections (residual learning)**:
  $$y = \mathcal{F}(x, \{W_i\}) + x$$
  Het signaal mag lagen overslaan via een identiteitsverbinding. Hierdoor kunnen gradiënten tijdens backpropagation ongehinderd terugvloeien, waardoor netwerken voor het eerst meer dan 100 lagen diep konden worden zonder vanishing gradient.

### 2.4 Evaluatiemetrics: PR-Curve vs. ROC-Curve & F1-Score
- **Waarom NOOIT een ROC-curve bij Objectdetectie?**  
  De ROC-curve zet Recall uit tegen $\text{FPR} = \frac{FP}{FP + TN}$. In een 1080p frame zijn er miljoenen potentiële achtergrondkaders waar géén object staat ($TN > 100.000$). Omdat $TN \to \infty$, is $\text{FPR} \approx 0$. De ROC-curve toont daardoor een vals-optimistische score ($\text{AUC} \approx 0.999$) en verbergt valse detecties.
- **De PR-curve is eerlijk:** Negeert $TN$ volledig en toetst zuiver op $TP$, $FP$ en $FN$.
- **F1-Score:** Het harmonisch gemiddelde $F_1 = 2 \cdot \frac{P \cdot R}{P + R}$.

---

## 4. DEEL 3: Object Detection (Multi-Scale, FPN Neck & Pyramid Heads)

### 3.1 Two-Stage (Faster R-CNN) vs. One-Stage (YOLO)
- **Two-Stage (Faster R-CNN):** Fase 1 stelt potentiële kandidaatregio's voor via ankerboxen (Region Proposal Network - RPN). Fase 2 snijdt deze uit via RoI Align en classificeert ze. Zeer accuraat, maar traag (5-10 FPS).
- **One-Stage (YOLO):** Verwerkt het volledige frame in één enkele forward pass ("You Only Look Once"). Het beeld wordt opgedeeld in een multi-scale grid waarin elke cel parallel klassenkansen en coördinaten voorspelt. Realtime ($>60\text{ FPS}$ op RTX 4070).

### 3.2 De Multi-Scale Architectuur: Backbone (C3, C4, C5)
In de backbone doorloopt het beeld opeenvolgende downsampling stappen:
- **C3 (Stride 8):** Hoge spatiale resolutie ($160 \times 160$ bij 1280p input). Bevat fijne ruimtelijke details (lijnen, kleine stippen), maar lage semantische betekenis.
- **C4 (Stride 16):** Medium resolutie ($80 \times 80$) met gemiddelde semantische context.
- **C5 (Stride 32):** Lage spatiale resolutie ($40 \times 40$), maar rijke abstracte semantische representaties.

### 3.3 De FPN/PAN Neck: Upsampling, Merging & Betekenis-Overdracht
Als we alleen op C5 zouden detecteren, is de bal al lang weggeschaald. Als we alleen op C3 zouden detecteren, ontbreekt semantisch begrip (het netwerk ziet alleen een witte vlek en verwart die met een schoen).
**De Neck lost dit op:**
1. Hij neemt de diepe laag C5 (lage resolutie, hoge semantische betekenis).
2. Hij **upsamplet** deze laag en voegt hem samen (*feature merge / concatenation*) met laag C4.
3. Hij upsamplet opnieuw en voegt samen met de hoge-resolutie laag C3.
4. Vervolgens voert een bottom-up pad (Path Aggregation Network - PAN) de gecombineerde informatie weer terug naar beneden.

Het resultaat zijn 3 piramidelagen (**P3, P4, P5**) die **allebei bezitten: scherpe ruimtelijke details én rijke semantische context!**

### 3.4 De 3 Predictie Heads (P3, P4, P5)
Elke piramidelaag heeft zijn eigen predictiekop met een specifiek receptief veld:
- **P3 Head ($160 \times 160$ grid):** Hoogste resolutie, kleinste kaders $\to$ **DETECTEERT DE KLEINE VOETBAL ($16 \times 17\text{ px}$)**!
- **P4 Head ($80 \times 80$ grid):** Middelgrote kaders $\to$ detecteert spelers verder weg op het veld.
- **P5 Head ($40 \times 40$ grid):** Grootste kaders $\to$ detecteert grote spelers dichtbij de camera en doelgebieden.

### 3.5 Post-Processing: Non-Maximum Suppression (NMS) & DFL
- **NMS:** Sorteert alle predicties op confidence, behoudt de hoogste, en verwijdert alle overlappende boxen met $\text{IoU} > \text{threshold}$.
- **Distribution Focal Loss (DFL):** In plaats van een vast getal voorspelt YOLO26 een continue waarschijnlijkheidsverdeling voor elke boxrand, wat resulteert in strakkere lokalisatie bij occlusie.

### 3.6 mAP Ontrafeld ("Mean of Averages") [EXAMEN VRAAG]
Waarom heet het *Mean Average Precision* als *Mean* en *Average* hetzelfde betekenen?
Het is letterlijk **het gemiddelde van de gemiddelden**:
1. **Average Precision (AP) per klasse:** De oppervlakte onder de PR-curve (AUC-PR) over alle recall-drempels voor één specifieke klasse (bv. bal).
2. **Mean Average Precision (mAP) over alle klassen:** Het rekenkundig gemiddelde (*Mean*) van die AP-waarden over alle $C$ verschillende klassen.
- **mAP@50 vs. mAP@50-95:** mAP@50 toetst met een tolerante eis van $\text{IoU} \ge 0.50$. De COCO-standaard mAP@50-95 middelt over 10 drempels ($0.50, 0.55, \dots, 0.95$) en straft onnauwkeurige boxen zwaar af.

---

## 5. DEEL 4: Semantische Segmentatie & 2D Homografie

### 4.1 Typen Segmentatie
- **Semantic:** Labelt pixels van categorieën ("stuff": gras, lucht, tribunes).
- **Instance:** Labelt individuele objecten ("things": speler 1 vs speler 2).
- **Panoptic:** Combineert beide (alle pixels + individuele objecten).

### 4.2 Fully Convolutional Networks (FCNs)
Klassieke classificatienetwerken eindigen met Flatten- en Dense-lagen, wat alle 2D-ruimtelijke structuren vernietigt. FCN's vervangen Dense lagen door **$1\times 1$ convoluties**, waardoor de 2D-kaart behouden blijft en pixelgewijze voorspellingen mogelijk worden.

### 4.3 U-Net: Transposed Convolutions, Checkerboard Artifacts & Skip Connections
- **Encoder:** Comprimeert het beeld via pooling om context te vangen, maar verliest fijne randen.
- **Decoder:** Schaalt feature maps op via **Transposed Convolutions** (leerbare filters).
  * *Checkerboard Artifacts:* Door overlapping bij stride 2 ontvangen centrale pixels onevenredig veel energie. Dit wordt opgelost door PixelShuffle (Sub-Pixel Convolutie) of verliespenalisatie.
- **Skip Connections:** Verbinden encoderlagen rechtstreeks met de corresponderende decoderlagen. Hierdoor injecteert het netwerk ruwe, scherpe pixelranden direct in de decoder.

### 4.4 Verliesfuncties bij Segmentatie
- De netwerkoutput is een Softmax kansverdeling. **Argmax wordt pas na training gebruikt**, omdat de afgeleide van Argmax 0 is en backpropagation blokkeert.
- Omdat beelden vaak voor $95\%$ uit achtergrond (gras) bestaan, faalt standaard Cross-Entropy. Men combineert CE daarom altijd met een **Shape Overlap Loss (Dice Loss of Soft IoU)**:
  $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 |Y \cap \hat{Y}| + \epsilon}{|Y| + |\hat{Y}| + \epsilon}$$

### 4.5 2D Planar Homografie & Planar Assumption
Een homografie $H \in \mathbb{R}^{3\times 3}$ met 8 vrijheidsgraden transformeert punten tussen twee vlakken:
$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
- **Planar Assumption:** We nemen aan dat de voeten van alle spelers en de bal zich op het platte veldvlak bevinden ($Z=0$). Daarom projecteren we de onderkant van de bounding box (`x_center, y_max`).
- **10-Punt Calibratie:** Met 10 referentiepunten en RANSAC behaalden we een sub-3-pixel herprojectiefout ($2.76\text{ px}$).

---

## 6. DEEL 5: Diepte, Zelf-Gesuperviseerd Leren & Foundation Models

### 5.1 Monocular Depth Estimation
Schat diepte uit één enkel 2D-beeld op basis van visuele cues: perspectief (convergerende veldlijnen), occlusie (speler staat voor een andere speler) en textuurgradiënt (gras wordt fijner in de verte).

### 5.2 Zelf-Gesuperviseerd Pretrainen zonder Labels
- **MAE (Masked Auto-Encoder):** Verdeelt een beeld in patches via een Vision Transformer (ViT), maskeert **$75\%$ van de patches**, en dwingt het model om de ontbrekende pixels te reconstrueren. Het model leert zo noodgedwongen visuele structuren zonder handmatige labels.
- **DINO (Self-distillation with NO labels):** Gebruikt een Teacher-Student architectuur. De student ziet kleine crops, de teacher ziet de globale crop. De student leert de latente representatie van de teacher te voorspellen. De teacher wordt niet geüpdatet via backprop, maar via een langzame **Exponential Moving Average (EMA)** van de studentgewichten.

### 5.3 Multimodale Foundation Models
- **CLIP (Contrastive Language-Image Pre-training):** Koppelt een beeldencoder en een tekstencoder in een gedeelde latente ruimte via Contrastive Loss. Maakt *zero-shot* classificatie mogelijk zonder hertraining.
- **SAM (Segment Anything Model):** Introduceert **Promptable Vision**. Door simpelweg een punt of bounding box als prompt te geven, segmenteert SAM elk gewenst object zonder extra training. Dit hebben we benut tijdens het annoteren van uitdagende frames.

---

## 7. DEEL 6: Praktische Experimenten, IoU/NMS Benchmark & Winnaar

In `run_threshold_experiments.py` hebben we 18 configuraties van Confidence (`conf`) en Non-Maximum Suppression (`iou`) systematisch geëvalueerd op de RTX 4070 GPU:

| Configuratie Naam | Conf | IoU | Precision | Recall | F1-Score | mAP@50 | mAP@50-95 | Bal mAP@50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WINNAAR mAP@50 & F1** | **0.005** | **0.60** | **0.7534** | **0.6740** | **0.7115** | **0.7291** | **0.3143** | **0.0676** |
| WINNAAR mAP@50-95 | 0.005 | 0.70 | 0.7044 | 0.6481 | 0.6751 | 0.7224 | **0.3201** | 0.0674 |
| Ultra-Low | 0.005 | 0.50 | 0.7446 | 0.6806 | 0.7112 | 0.7236 | 0.3090 | 0.0676 |
| Ultra-High Recall | 0.010 | 0.60 | 0.7534 | 0.6740 | 0.7115 | 0.7280 | 0.3134 | 0.0659 |
| Ultra-High Recall | 0.010 | 0.50 | 0.7446 | 0.6806 | 0.7112 | 0.7219 | 0.3083 | 0.0659 |
| Ball-Tuned | 0.015 | 0.60 | 0.7534 | 0.6740 | 0.7115 | 0.7249 | 0.3127 | 0.0634 |
| Ball-Tuned | 0.020 | 0.60 | 0.7534 | 0.6740 | 0.7115 | 0.7239 | 0.3118 | 0.0634 |
| Ball-Optimized | 0.025 | 0.60 | 0.7534 | 0.6740 | 0.7115 | 0.7239 | 0.3112 | 0.0634 |
| Permissive NMS | 0.025 | 0.70 | 0.7044 | 0.6481 | 0.6751 | 0.7173 | 0.3169 | 0.0633 |
| Medium Confidence | 0.050 | 0.60 | 0.7534 | 0.6740 | 0.7115 | 0.7146 | 0.3076 | 0.0571 |
| Conservative | 0.100 | 0.60 | 0.7534 | 0.6740 | 0.7115 | 0.7044 | 0.3028 | 0.0541 |
| YOLO Default Baseline | 0.250 | 0.60 | 0.7721 | 0.6480 | 0.7046 | 0.6292 | 0.2758 | 0.0500 |
| Strict Baseline | 0.250 | 0.70 | 0.6929 | 0.6492 | 0.6703 | 0.6252 | 0.2782 | 0.0500 |
| High Confidence Baseline | 0.500 | 0.60 | 0.9749 | 0.3415 | 0.5058 | 0.3297 | 0.1566 | 0.0350 |

**Verantwoording van de Winnaar (`conf=0.005, iou=0.60`):**
- Behaalt de allerhoogste **mAP@50 ($0.7291$)**, een fenomenale winst van **+10.0%** ten opzichte van de standaard YOLO drempel ($0.6292$).
- Behaalt de hoogste **F1-Score ($0.7115$)** en piekt in baldetectie met **Bal mAP@50 = 0.0676** (+35.2% winst).
- **Waarom?** Een voetbal beslaat slechts $16 \times 17\text{ px}$. Bij snelle passes daalt de netwerkconfidence door bewegingsonscherpte tot onder $0.05$. Door `conf=0.005` in te stellen, blijven deze vroege detecties behouden, terwijl `iou=0.60` duplicaatboxen bij spelers elimineert zonder twee strijdende spelers in een duel per ongeluk te supprimeren.

### Visuele Grafieken & Officiële Curven:
1. **Vergelijking PR-Curven (Voor vs. Na Tuning):** `runs/pr_curve_comparison_before_after.png`  
   Zij-aan-zij vergelijking van de originele baseline (`mAP@50 = 0.629`) met het model na finetuning en parameter tuning (`mAP@50 = 0.729`). Toont direct hoe het oppervlak onder de curve (AUC) voor de bal en spelers aanzienlijk toeneemt.
2. **Volledige 4-in-1 Drempelvergelijking:** `runs/threshold_comparison_charts.png`  
   Toont de mAP barplot, Precision vs. Recall trade-off scatterplot, de logaritmische drempelcurve en de NMS IoU gevoeligheid.
3. **PR-Curve (Precision-Recall / AUC-PR):** `runs/ball_yolo26_1280/BoxPR_curve.png`  
   Conform de opdracht gebruiken we de PR-curve in plaats van een ROC-curve, omdat de achtergrondklasse ('geen bal') zwaar dominant is. De oppervlakte onder deze curve (AUC-PR) definieert de mAP@50 per klasse.
4. **F1-Confidence Curve:** `runs/ball_yolo26_1280/BoxF1_curve.png`  
   Visualiseert exact hoe de F1-score piekt bij lage confidence thresholds en instort bij klassieke hoge drempels ($>0.30$).
5. **Genormaliseerde Verwarringsmatrix:** `runs/ball_yolo26_1280/confusion_matrix_normalized.png`

---

## 8. DEEL 7: Sports Vision Pipeline, Invariant Tracking & Telemetrie

### De InvariantBallTracker Architectuur:
1. **Fysische Spatiale Gating:** Zoekstraal begrensd op $d_{cam} \le 170\text{ px}$ en $d_{pitch} \le 160\text{ px}$. Teleportatie naar de overkant is wiskundig uitgesloten.
2. **Dribble Carrier Hold:** Bij occlusie tijdens het dribbelen wordt de bal automatisch verankerd aan de voeten van de baldrager.
3. **Micro-Penaltystip Filter ($20\text{ px}$):** Sluit de statische stip op $(925, 340)$ betrouwbaar uit, maar laat passes dwars door de 16m ongehinderd door.
4. **Wrijving & Demping ($0.86$):** Clampt posities strikt binnen het veld $[0, 1050] \times [0, 680]$.
5. **Kleurconsistentie:** De bal blijft altijd puur rood (`BGR = [0, 0, 255]`) met witte contrastrand.

**Resultaat:** **660 van de 660 frames gevolgd (100.0% trackingdekking)**, 0 sprongen, realtime telemetrie (live km/u, team-balbezit) en 2D Gaussiaanse KDE veldbezetting heatmap.

---

## 9. Conclusie & Verantwoording

Met de ontwikkelde pipeline binnen de map `dimi/` voldoet het project integraal aan alle formele eisen en bereikt het de Extra Mile:
1. **Data & Annotatie:** 154 frames geordend gesplitst, het Small Object Problem opgelost via 1280p resolutie, en Soft Labels toegepast tegen overconfidence.
2. **Modellering & Multi-Scale:** Grondig begrip en toepassing van de Backbone (C3-C5), FPN Neck en P3 Pyramid Head voor kleine objecten.
3. **Foutanalyse:** Wiskundig bewezen waarom ROC ongeldig is voor objectdetectie en mAP ontleed als het gemiddelde van de gemiddelden.
4. **Homografie & Minimap:** 10 RANSAC punten met sub-3-pixel precisie ($2.76\text{ px}$).
5. **Deployment & Tracking:** 100.0% continue tracking en een strak, professioneel Streamlit dashboard zonder overdreven iconen.
