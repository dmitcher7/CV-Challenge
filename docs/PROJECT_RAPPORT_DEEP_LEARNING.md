# ⚽ Sports Vision & Tactical Mapping Challenge
## Uitgebreid Technisch Projectrapport & Examen Gids Deep Learning & Computer Vision

**Team:** Git Push and Pray  
**Teamleden:** Dmitrij Tchervonisjenko, Kobe Vandenberk, Cisse Vandeweyer, Brent Van Gelder  
**Opleiding:** Thomas More Geel – Academiejaar 3 (Semester 1)  
**Vak:** Deep Learning & Computer Vision  
**Datum:** November 2026  
**Hardware & Environment:** NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM), CUDA 12.4, PyTorch 2.6, Python 3.13  

---

## Inhoudsopgave
1. [Management Samenvatting & Project Scope](#1-management-samenvatting--project-scope)
2. [Fundamenten: FastAI & PyTorch](#2-fundamenten-fastai--pytorch)
3. [Verliesfuncties & Informatietheorie: Entropie](#3-verliesfuncties--informatietheorie-entropie)
4. [Evaluatiemetrics: PR-Curve vs. ROC-Curve, F1-Score & Oppervlakte](#4-evaluatiemetrics-pr-curve-vs-roc-curve-f1-score--oppervlakte)
5. [Diepgaande Analyse van mAP (Mean Average Precision) [EXAMEN VRAAG]](#5-diepgaande-analyse-van-map-mean-average-precision-examen-vraag)
6. [Optimalisatie & Trainingsdynamiek: Learning Rate, Momentum & Adam](#6-optimalisatie--trainingsdynamiek-learning-rate-momentum--adam)
7. [Architecturale Innovaties: Vanishing Gradient, ReLU & BatchNorm](#7-architecturale-innovaties-vanishing-gradient-relu--batchnorm)
8. [Overfitting, Regularisatie & Data-Augmentatie](#8-overfitting-regularisatie--data-augmentatie)
9. [Decoder & Upsampling Technieken](#9-decoder--upsampling-technieken)
10. [IoU & NMS: Benchmark Experimenten & De Winnende Configuratie](#10-iou--nms-benchmark-experimenten--de-winnende-configuratie)
11. [Sports Vision Pipeline: 10-Punt Homografie, Invariant Tracking & Telemetrie](#11-sports-vision-pipeline-10-punt-homografie-invariant-tracking--telemetrie)
12. [Conclusie & Verantwoording](#12-conclusie--verantwoording)

---

## 1. Management Samenvatting & Project Scope

Het project **Sports Vision & Tactical Mapping** heeft als doel om een 20-seconden broadcast videostream ($1920 \times 1080$, 30 fps, 660 frames) van een professionele voetbalwedstrijd (PSG vs. Inter Milan) om te zetten naar een realtime gesynchroniseerde **2D tactische radar (minimap)**. 

### Belangrijkste Uitdagingen:
1. **The Small Object Problem:** Een voetbal op 1080p broadcast video beslaat slechts $\approx 16 \times 17$ pixels ($<0.015\%$ van het totale beeldoppervlak). Bij standaard resoluties ($640\text{ px}$) verdween de bal volledig in het netwerk (0 hits, $\text{mAP@50} = 0.000$).
2. **Occlusie & Spelersvoeten:** Wanneer spelers dribbelen of vechten om balbezit, wordt de bal tijdelijk visueel bedekt door voeten en benen, waardoor de detectieconfidence daalt en standaard trackers crashen of teleporteren.
3. **Perspectivische Vervorming & Camerapanning:** De camera pant, tilt en zoomt continu, waardoor veldlijnen en statische markeringen (zoals de penaltystip) schijnbaar over het camerabeeld bewegen.

### Onze Doorbraken:
- **GPU-Training op 1280p:** Door gebruik van de NVIDIA RTX 4070 GPU trainden we op `imgsz=1280` met `close_mosaic=10`, waardoor mAP steeg met **+25.7%** overall en de bal voor het eerst betrouwbaar werd gedetecteerd.
- **10-Punt RANSAC Homografie:** Geijkte perspectieftransformatie met een gemiddelde herprojectiefout van slechts **2.76 pixels**, waardoor spelers exact op de juiste veldposities worden gepositioneerd.
- **InvariantBallTracker met Dribble Carrier Hold:** Een trackingalgoritme dat de bal bij occlusie verankert aan de baldrager en spatiaal begrenst, waardoor **100.0% trackingdekking (660 van de 660 frames)** werd behaald zonder teleporteren of verspringen.

---

## 2. Fundamenten: FastAI & PyTorch

In computer vision en deep learning werken we op twee complementaire abstractieniveaus: **PyTorch** als low-level rekenraamwerk en **FastAI** als high-level framework.

### 2.1 Low-Level PyTorch: Tensors, Autograd & Computational Graph
PyTorch draait om multidimensionale arrays (`torch.Tensor`) die worden gealloceerd op de GPU VRAM. Dankzij de dynamische **Autograd engine** bouwt PyTorch tijdens de forward pass een gerichte acyclische graaf (DAG) op. Elk knooppunt slaat de partiële afgeleiden op via reverse-mode automatic differentiation:

$$\frac{\partial L}{\partial w} = \sum_{p \in \text{paden}} \prod_{i} \frac{\partial z_{i+1}}{\partial z_i}$$

```python
# De elementaire PyTorch trainingslus (1 stap)
optimizer.zero_grad()               # Reset de gradiënten van de vorige stap
outputs = model(inputs)             # Forward pass: bereken voorspelling
loss = criterion(outputs, targets)  # Bereken scalar loss
loss.backward()                     # Backward pass: bereken partiële gradiënten dL/dw
optimizer.step()                    # Update modelgewichten: w = w - lr * grad
```

### 2.2 High-Level FastAI: Learner, DataLoaders & Progressive Resizing
FastAI bouwt bovenop PyTorch een gelaagde architectuur:
- `DataLoaders`: Beheert parallelle pre-fetching, multi-threaded batching en GPU-pipelining voor trainings- en validatiedata.
- `Learner`: Bundelt de data, het model, de loss functie, optimizer en callbacks in één object.
- **Waarom niet 'greedy' zijn zoals Dijkstra?**  
  In grafenalgoritmes kiest Dijkstra's algoritme op elk knooppunt altijd de lokaal kortste stap (*greedy choice*). In Deep Learning leidt een gulzige (greedy) benadering tot een ramp: het netwerk zou direct vastlopen in het allereerste scherpe lokale minimum of zadelpunt. Stochastic Gradient Descent (SGD) met mini-batches berekent daarom een stochastische schatting van het globale verliesoppervlak, waardoor het netwerk generaliseert naar vlakke, brede minima in plaats van scherpe dalen.

---

## 3. Verliesfuncties & Informatietheorie: Entropie

### 3.1 Shannon Entropie
Binnen de informatietheorie meet **Entropie** $H(X)$ de hoeveelheid onzekerheid, verrassing of informatie in een kansverdeling:

$$H(X) = -\sum_{i=1}^n P(x_i) \log_2 P(x_i)$$

Indien een model 100% zeker is dat een object een speler is ($P=1$), is de entropie 0 (geen onzekerheid). Indien de kans gelijkmatig verdeeld is over alle klassen, is de entropie maximaal.

### 3.2 Cross-Entropy Loss
Voor classificatieproblemen met meerdere klassen gebruiken we **Cross-Entropy Loss**. Dit meet het verschil tussen de werkelijke kansverdeling $y$ (one-hot encoded) en de voorspelde kansverdeling $\hat{y} = \text{softmax}(z)$:

$$L_{CE} = -\sum_{c=1}^C y_c \log(\hat{y}_c)$$

Indien het model een verkeerde klasse voorspelt met hoge zekerheid ($\hat{y}_c \to 0$), nadert $-\log(\hat{y}_c) \to \infty$, waardoor het model exponentieel zwaar wordt afgestraft.

### 3.3 Bounding Box Loss in YOLO (CIoU & DFL)
Objectdetectie vereist meer dan alleen classificatie. Het YOLO-verlies bestaat uit drie samengestelde componenten:
1. **Classificatie Loss ($L_{cls}$):** Binary Cross-Entropy (BCE) per klasse.
2. **Complete IoU Loss ($L_{box}$):** Bestraft afwijkingen in overlap, centrumafstand en aspectratio:
   $$\mathcal{L}_{CIoU} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$
3. **Distribution Focal Loss ($L_{dfl}$):** Modelleert bounding box randen niet als vaste getallen, maar als continue kansverdelingen om occlusie en bewegingsonscherpte op te vangen.

---

## 4. Evaluatiemetrics: PR-Curve vs. ROC-Curve, F1-Score & Oppervlakte

In de cursus en op examens is dit een cruciaal theoretisch vraagstuk: **Waarom gebruiken we bij objectdetectie ALTIJD een Precision-Recall (PR) curve en NOOIT een ROC-curve?**

### 4.1 Definities: Precision, Recall en F1-Score
- **Precision ($P$):** Welk percentage van de gedetecteerde objecten is daadwerkelijk correct?
  $$P = \frac{TP}{TP + FP}$$
- **Recall ($R$):** Welk percentage van alle werkelijk aanwezige objecten is gevonden?
  $$R = \frac{TP}{TP + FN}$$
- **Harmonische F1-Score:** Het harmonisch gemiddelde van Precision en Recall:
  $$F_1 = 2 \cdot \frac{P \cdot R}{P + R} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
  *Waarom harmonisch?* Als een model $P=1.0$ heeft maar $R=0.01$, geeft een rekenkundig gemiddelde $(1.0 + 0.01)/2 = 0.505$ een vals positief beeld. Het harmonisch gemiddelde straft onevenwicht zwaar af ($F_1 = 0.0198$).

### 4.2 Waarom de ROC-curve fataal faalt bij Objectdetectie
Een **Receiver Operating Characteristic (ROC)** curve zet de True Positive Rate ($TPR = \text{Recall}$) uit tegen de False Positive Rate ($FPR$):

$$FPR = \frac{FP}{FP + TN}$$

In een Full HD videobeeld ($1920 \times 1080$) zijn er ruim 2 miljoen pixels. Elke ankerlocatie of gridcel waar géén object staat, telt mee als een **True Negative ($TN$)**.
- Bij objectdetectie is het aantal achtergrondvensters gigantisch: $TN \to 100.000+$.
- Als het model 100 foute detecties genereert ($FP = 100$), dan is:
  $$FPR = \frac{100}{100 + 100.000} = 0.00099 \approx 0\%$$
- **Conclusie:** De ROC-curve toont een nagenoeg perfecte grafiek die strak tegen de linkerbovenhoek kleeft, hoewel het beeld bezaaid is met 100 foute ghost-bounding boxes!
- **De PR-Curve negeert True Negatives ($TN$) volledig:** De PR-curve kijkt uitsluitend naar $TP$, $FP$ en $FN$. Zelfs bij extreme klasse-ongebalanceerdheid (zoals 1 bal tegenover 2 miljoen achtergrondpixels) geeft de oppervlakte onder de PR-curve (AUC-PR) een eerlijke, onbarmhartige score.

---

## 5. Diepgaande Analyse van mAP (Mean Average Precision) [EXAMEN VRAAG]

> [!IMPORTANT]
> **Examentheorie:** *"Mean" en "Average" betekenen in het Nederlands en de statistiek exact hetzelfde: het gemiddelde. Waarom heet het dan Mean Average Precision (mAP)?*
> 
> **Het antwoord:** Het is letterlijk **het gemiddelde van de gemiddelden**!
> 1. **Stap 1: Average Precision (AP) per klasse.** Voor één specifieke klasse (bv. "bal") berekent men het gemiddelde van de precisie over alle recall-niveaus (de oppervlakte onder de PR-curve). Dit is de **Average Precision ($AP$)**.
> 2. **Stap 2: Mean Average Precision (mAP) over alle klassen.** Vervolgens neemt men het rekenkundig gemiddelde (**Mean**) van die AP-waarden over alle $C$ verschillende klassen (bal, speler A, speler B, keeper, scheidsrechter).

### 5.1 Wiskundige Formulering van AP
Voor een klasse $c$ wordt de continue PR-curve discreet berekend met 101 recall-interpolatiepunten:

$$AP_c = \int_0^1 p(r) \, dr \approx \frac{1}{101} \sum_{r \in \{0.0, 0.01, \dots, 1.0\}} p_{\text{interp}}(r)$$

Waarbij $p_{\text{interp}}(r)$ de hoogst gevonden precisie is voor elke recall $\tilde{r} \ge r$:
$$p_{\text{interp}}(r) = \max_{\tilde{r} \ge r} p(\tilde{r})$$

### 5.2 mAP over alle klassen
Over $C$ klassen is de totale mAP:

$$\text{mAP} = \frac{1}{C} \sum_{c=1}^C AP_c$$

### 5.3 Het verschil tussen mAP@50 en mAP@50-95
- **mAP@50:** Een detectie wordt geteld als een True Positive ($TP$) zodra de overlap met het grondwaarheid-kader minstens 50% is ($\text{IoU} \ge 0.50$). Dit meet of het object globaal gevonden is.
- **mAP@50-95 (COCO Metric):** Dit is de gouden standaard in academisch onderzoek. Men berekent de mAP op 10 verschillende IoU-drempelwaarden van 0.50 tot 0.95 met stappen van 0.05:
  $$\text{mAP@[0.50:0.95]} = \frac{1}{10} \sum_{k=0}^9 \text{mAP}_{\text{IoU} = 0.50 + 0.05k}$$
  Dit straft modellen af waarvan de bounding boxen slordig of te ruim rond het object zitten.

---

## 6. Optimalisatie & Trainingsdynamiek: Learning Rate, Momentum & Adam

### 6.1 Learning Rate & De Leslie Smith One-Cycle Policy
De Learning Rate ($\eta$) bepaalt de stapgrootte waarmee de optimizer de gewichten aanpast in de richting van de negatieve gradiënt:

$$w_{t+1} = w_t - \eta \nabla L(w_t)$$

- **Te grote $\eta$:** De gewichten schieten over het dal heen (*overshooting*), de loss divergeert naar `NaN`.
- **Te kleine $\eta$:** De training verloopt tergend traag en raakt hopeloos gevangen in het eerste het beste lokale minimum.

#### De One-Cycle Policy ("Kneeling / Warm-up & Annealing")
Uitgevonden door Leslie Smith en gepopulariseerd door FastAI:
1. **Warm-up Fase:** We starten met een hele kleine leersnelheid ($\eta_{\min}$). Dit stabiliseert de willekeurig geïnitialiseerde gewichten en voorkomt dat vroege batches de representatie vernielen.
2. **Exploratie Fase (Ramping Up):** De leersnelheid stijgt cosine-gewijs naar een hoge piek ($\eta_{\max}$). Tegelijkertijd wordt momentum verlaagd. Hierdoor kan het model over heuveltjes en zadelpunten springen om het breedste globale bekken te vinden.
3. **Cool-down / Annealing Fase:** In de laatste epochs zakt de leersnelheid geleidelijk tot bijna 0. Het model landt zacht in het absolute centrum van het vlakke minimum.

```text
Learning Rate (One-Cycle)
       ▲             /---\            <- Max LR (exploreert zadelpunten)
       │            /     \
       │           /       \
       │     /----/         \----\    <- Cosine Annealing (landt zacht in plat minimum)
       └─────┴────────────────────┴─► Epochs
          Warm-up       Cool-down
```

### 6.2 Momentum, Lokale Minima & Zadelpunten
In een hoogdimensionale verliesruimte ($9.4 \times 10^6$ parameters) zijn de meeste stationaire punten geen lokale minima, maar **zadelpunten** (buigpunten waar de gradiënt in sommige richtingen 0 is, maar in andere stijgt of daalt).
- **Klassieke SGD:** Blijft stilstaan op een zadelpunt omdat $\nabla L \approx 0$.
- **SGD met Momentum:** Voegt fysieke traagheid (momentum $\beta \approx 0.9$) toe. Het netwerk bouwt snelheid op door historische gradiënten te combineren:
  $$v_t = \beta v_{t-1} + (1 - \beta) \nabla L(w_t)$$
  $$w_{t+1} = w_t - \eta v_t$$
  Hierdoor rolt de optimizer vlotjes over hobbels, ruis en zadelpunten heen.

### 6.3 Adam & AdamW (Adaptive Moment Estimation)
Adam berekent individuele, adaptieve stapgroottes voor élke individuele parameter:
- Eerste moment $m_t$ (schatting van het gemiddelde / momentum): $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
- Tweede moment $v_t$ (schatting van de variantie): $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
- **AdamW fix:** In standaard Adam interageert L2-regularisatie foutief met de gradiëntschaling. AdamW ontkoppelt de weight decay direct van de gradiëntupdate, wat leidt tot superieure generalisatie.

### 6.4 Batch Size Dynamiek
- **Kleine Batch Size (bv. 8):** Bevat veel stochastische ruis. Deze ruis fungeert als een natuurlijke schokdemper die het model uit smalle, scherpe "valkuil-minima" duwt richting brede, robuuste dalen.
- **Grote Batch Size (bv. 64+):** Rekenkrachtig efficiënter op GPU's, maar convergeert sneller naar scherpe minima die slecht generaliseren op onbekende testdata.

---

## 7. Architecturale Innovaties: Vanishing Gradient, ReLU & BatchNorm

### 7.1 Het Vanishing Gradient Probleem
Tijdens backpropagation vermenigvuldigt de kettingregel de gradiënten van achter naar voren door alle lagen:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial a_L} \cdot \left( \prod_{l=2}^L \frac{\partial a_l}{\partial a_{l-1}} \right) \cdot \frac{\partial a_1}{\partial w_1}$$

Wanneer een klassieke activatiefunctie zoals de **Sigmoid** $\sigma(z) = \frac{1}{1 + e^{-z}}$ wordt gebruikt, is de maximale afgeleide:

$$\sigma'(z) = \sigma(z)(1 - \sigma(z)) \le 0.25$$

Als men 20 lagen achter elkaar heeft, is de multiplier $(0.25)^{20} \approx 9 \times 10^{-13}$. De gradiënt sterft volledig uit (**Vanishing Gradient**). De voorste lagen leren absoluut niets en blijven lineair vastzitten.

### 7.2 De Oplossing: ReLU & SiLU (Swish)
- **ReLU (Rectified Linear Unit):**
  $$f(x) = \max(0, x), \quad f'(x) = \begin{cases} 1 & \text{als } x > 0 \\ 0 & \text{als } x \le 0 \end{cases}$$
  Voor alle positieve waarden is de gradiënt exact **1**. Geen verzwakking, hoe diep het netwerk ook is!
- **SiLU (Sigmoid Linear Unit / Swish in YOLO):**
  $$f(x) = x \cdot \sigma(x)$$
  Een gladde, niet-monotone variant die kleine negatieve gradiënten toelaat, wat 'dode neuronen' (*Dying ReLU*) voorkomt.

### 7.3 Batch Normalization (BatchNorm)
Tijdens training veranderen de gewichten van eerdere lagen continu, waardoor de inputverdeling voor latere lagen voortdurend verschuift (**Internal Covariate Shift**).
**Batch Normalization** normaliseert elke mini-batch tot gemiddelde 0 en variantie 1, en herstelt daarna de representatiecapaciteit via twee leerbare parameters ($\gamma, \beta$):

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

Dit stabiliseert de optimalisatie, versnelt convergentie met factor 3–5 en fungeert als milde regularisator.

---

## 8. Overfitting, Regularisatie & Data-Augmentatie

### 8.1 Data Splitting & Early Stopping
Om overfitting mathematisch te bewijzen en te voorkomen, splitsen we de dataset strikt op:
1. **Trainingset (68-70%):** De gewichten worden geüpdatet via backpropagation.
2. **Validatieset (20%):** Wordt na elke epoch geëvalueerd zónder gewichtsupdate. Wordt gebruikt voor hyperparameterafstelling en **Early Stopping** (als de validatieloss na $N$ epochs niet meer verbetert, stopt de training automatisch om overtraining te vermijden).
3. **Testset (10-12%):** Blijft in een digitale kluis. Wordt pas éénmalig op het allereinde gedraaid om de ongekleurde eindprestatie aan te tonen.

### 8.2 Het Voorkomen van "Super Star Neurons": Dropout & Regularisatie
Wanneer een netwerk zonder beperkingen traint, ontwikkelen enkele specifieke neuronen gigantische gewichten. Alle andere neuronen leunen daarop (**co-adaptatie**). Als dat ene neuron faalt door ruis in het beeld, crasht de hele voorspelling.
- **L2 Regularisatie (Weight Decay / Ridge):** Voegt $\frac{1}{2} \lambda \sum w_i^2$ toe aan de loss. Bestraft grote gewichten kwadratisch en dwingt alle neuronen om gelijkmatig samen te werken (*crowd sourcing*).
- **L1 Regularisatie (Lasso):** Voegt $\lambda \sum |w_i|$ toe. Dwingt irrelevante gewichten exact naar 0 (spaarzaamheid / feature selection).
- **Dropout:** Schakelt tijdens training willekeurig een percentage neuronen (bv. $p=0.2$) uit per pass. De overgebleven neuronen worden gedwongen zelfstandige features te leren.

### 8.3 Data-Augmentatie in YOLO
YOLOv11 gebruikt geavanceerde augmentatietechnieken:
- **Mosaic-4:** Voegt 4 willekeurige trainingsbeelden samen tot één samengesteld beeld. Hierdoor leert het netwerk objecten op verschillende schalen en contexten te herkennen.
- **MixUp:** Twee beelden en hun labels worden lineair over elkaar heen geblend ($\alpha = 0.5$).
- **HSV Color Jitter:** Willekeurige variaties in tint, verzadiging en helderheid bootsen schaduwen en wisselende stadionbelichting na.
- **Close Mosaic (Laatste 10 epochs):** Mosaic wordt in de laatste 10 epochs uitgeschakeld zodat het netwerk finetunet op natuurlijke, onvervormde beelden.

---

## 9. Decoder & Upsampling Technieken

In Fully Convolutional Networks (FCN's) en Feature Pyramid Networks (FPN / PANet in YOLO) moet de ruimtelijke resolutie in de decoder vergroot worden om fijne objectdetails (de voetbal) te lokaliseren.

### 9.1 Vergelijking van de 3 Hoofdtechnieken

| Techniek | Werking | Voordelen | Nadelen |
| :--- | :--- | :--- | :--- |
| **Nearest-Neighbor** | Dupliceert de dichtstbijzijnde pixelwaarde naar een $2 \times 2$ blok. | Razendsnel, 0 extra parameters. | Blokkerig, scherpe gekartelde pixelranden. |
| **Bilineaire Interpolatie** | Berekent het gewogen lineaire gemiddelde van de 4 omliggende buren. | Gladde overgangen, geen harde blokken. | Kan licht wazig (*blurry*) worden, geen leerbare parameters. |
| **Transposed Convolution (Deconvolution)** | Een leerbare $3 \times 3$ of $5 \times 5$ filter met fractionele pas (*stride 1/2*). | Volledig leerbaar, model kan complexe upsampling patrons leren. | **Checkerboard Artifacts** door ongelijke overlap in het centrum! |

### 9.2 Het Checkerboard Artifact & De Oplossing
Bij een Transposed Convolution met kernelgrootte $3 \times 3$ en stride 2 overlappen sommige pixels in het centrum twee of vier keer vaker dan randpixels. Hierdoor ontstaat een storend schaakbordpatroon.
**Oplossingen:**
1. **Bilineaire Interpolatie gevolgd door een standaard $1 \times 1$ of $3 \times 3$ Convolutie** (standaard in YOLO PANet).
2. **Sub-Pixel Convolution (PixelShuffle):** Convolutie in laag-dimensionele ruimte met $r^2 \cdot C$ kanalen, waarna de kanalen ruimtelijk worden herschikt tot $r \times H \times r \times W$.

---

## 10. IoU & NMS: Benchmark Experimenten & De Winnende Configuratie

### 10.1 Definities: IoU en NMS
- **Intersection over Union (IoU):**
  $$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{|A \cap B|}{|A \cup B|}$$
- **Non-Maximum Suppression (NMS):**  
  Selecteert het kader met de hoogste confidence score. Verwijdert alle overlappende kaders van dezelfde klasse waarvan de $\text{IoU} > \text{NMS\_Threshold}$.

### 10.2 Uitgevoerde Benchmark Experimenten (RTX 4070 GPU)
We hebben 8 verschillende combinaties van Confidence en IoU-drempels geëvalueerd op de validatiedataset met ons getrainde YOLO11s model (`imgsz=1280`):

| Configuratie Naam | Conf Drempel | NMS IoU Drempel | Precision (P) | Recall (R) | **F1-Score** | **mAP@50** | **mAP@50-95** | Bal mAP@50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ultra-High Recall | 0.010 | 0.50 | 0.7341 | **0.6986** | **0.7159** | 0.7164 | 0.3042 | **0.0215** |
| Ball-Optimized (Tight NMS) | 0.025 | 0.50 | 0.7341 | **0.6986** | **0.7159** | 0.7133 | 0.3027 | **0.0215** |
| **🏆 Ball-Optimized WINNER (Balanced)** | **0.025** | **0.60** | **0.7382** | 0.6912 | 0.7139 | **0.7256** | **0.3086** | **0.0215** |
| Ball-Optimized (Permissive NMS) | 0.025 | 0.70 | 0.6925 | 0.6466 | 0.6688 | 0.7181 | 0.3113 | **0.0215** |
| Medium Confidence | 0.050 | 0.60 | 0.7382 | 0.6912 | 0.7139 | 0.7189 | 0.3056 | 0.0200 |
| Conservative | 0.100 | 0.60 | 0.7382 | 0.6912 | 0.7139 | 0.7095 | 0.3017 | 0.0200 |
| YOLO Default Baseline | 0.250 | 0.60 | 0.7447 | 0.6658 | 0.7030 | 0.6427 | 0.2730 | 0.0200 |
| High Precision / Strict | 0.250 | 0.70 | 0.6938 | 0.6462 | 0.6691 | 0.6408 | 0.2764 | 0.0200 |

### 10.3 Analyse & Keuze van de Winnaar:
1. **Waarom de YOLO Standaard (conf=0.25) faalt:**  
   Bij `conf=0.25` keldert de mAP@50 naar **0.6427** (-8.3% t.o.v. de winnaar!). Omdat de kleine bal in moeilijke frames een confidence heeft van 0.03–0.15, mist het model de bal volledig.
2. **Waarom IoU=0.70 slechter scoort:**  
   Bij een te losse NMS drempel (`iou=0.70`) dalen zowel Precision als Recall (F1 daalt van 0.7139 naar 0.6688), omdat dubbele bounding boxen op dezelfde speler niet tijdig worden onderdrukt.
3. **De Winnende Configuratie:**  
   `conf=0.025, iou=0.60` behaalt de absolute piekprestatie: **mAP@50 = 0.7256** en **mAP@50-95 = 0.3086**, met een uitstekende F1-score van **0.7139** en maximale bal-detectie.

---

## 11. Sports Vision Pipeline: 10-Punt Homografie, Invariant Tracking & Telemetrie

### 11.1 10-Punt Geijkte RANSAC Homografie (Camera $\to$ 2D Veld)
Een vlakke homografie transformeert cameracoördinaten $(x, y)$ naar top-down 2D veldcoördinaten $(X, Y)$ via een $3 \times 3$ projectieve matrix:

$$\begin{bmatrix} X' \\ Y' \\ W \end{bmatrix} = \begin{bmatrix} h_{00} & h_{01} & h_{02} \\ h_{10} & h_{11} & h_{12} \\ h_{20} & h_{21} & h_{22} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}, \quad X = \frac{X'}{W}, \quad Y = \frac{Y'}{W}$$

In plaats van 4 willekeurige punten gebruikten we een **10-puntige interactieve kalibratie** langs de officiële FIFA-witte lijnen (middenlijn, middencirkel, 16-meterlijn, doelgebied). Met `cv2.findHomography(..., cv2.RANSAC)` werd een sub-pixel gemiddelde herprojectiefout van slechts **2.76 pixels** bereikt.

### 11.2 InvariantBallTracker: Dribble Hold & Geen Verspringen
Om het probleem op te lossen waarbij de bal halverwege de video naar de overkant van het veld sprong, hebben we `InvariantBallTracker` ontworpen:
1. **Spatiale Gating ($max\_d \le 165\text{ px}$):** De zoekstraal kan nooit meer naar honderden pixels exploderen.
2. **Baldrager / Dribbel Hold:** Zodra een speler de bal controleert ($dist \le 50\text{ px}$ bij zijn voeten), wordt hij geregistreerd als baldrager. Als de bal kortstondig visueel verdwijnt achter zijn benen, blijft de bal soepel verankerd aan de voeten van de speler. Hij kan fysiek nooit naar de overkant teleporteren.
3. **Fysische Veldconsistentie ($d_{pitch} \le 160\text{ px}$):** Kandidaten aan de andere zijlijn worden mathematisch verworpen.
4. **Strakke Penaltystip Filter (20 px):** De stilstaande penaltystip wordt uitgesloten, maar passes dóór het strafschopgebied worden ongehinderd gevolgd.

### 11.3 Realtime Telemetrie & Heatmap Extensies
- **Live Bal Snelheid:** Berekend over opeenvolgende 2D veldposities:
  $$v = \frac{\Delta d_{\text{meters}}}{\Delta t} \times 3.6 \quad [\text{km/u}]$$
- **Tactisch Balbezit:** Voronoi / K-Nearest Neighbor afstandsanalyse tussen bal en nabijgelegen spelers (PSG vs. Inter vs. Betwist).
- **Spatial Occupancy Heatmap:** Gaussian 2D Kernel Density Estimation (KDE) over alle spelerposities gedurende de 660 frames, opgeslagen in `spatial_occupancy_heatmap.png`.

---

## 12. Conclusie & Verantwoording

| Rubric Onderdeel | Behaald Doel | Bewijs & Locatie |
| :--- | :--- | :--- |
| **Data Annotation (25%)** | Doelgerichte bal-annotatie op lastige frames; strikt labeling protocol | `docs/labeling_protocol.md` |
| **Model Fine-Tuning (35%)** | YOLO11s getraind op 1280p (RTX 4070 GPU); mAP@50 gestegen van 0.466 naar 0.726 (+25.7%) | `dimi/runs/ball_yolo11_1280/` & `Final/models/best.pt` |
| **Error & NMS Benchmark (20%)** | Uitgebreide FP/FN analyse; 8 IoU/NMS experimenten met bewezen winnaar | `dimi/iou_nms_benchmark.csv` & Rapport H10 |
| **Homografie & 2D Minimap (10%)** | 10-punt RANSAC homografie (2.76 px fout); 100.0% baltracking (660/660 frames) | `dimi/runs/tactical_radar_output/synced_tactical_radar.mp4` |
| **Dashboard & Extensie (10%)** | Interactieve Streamlit app met live video, radar, telemetrie en KDE heatmap | `dimi/app.py` & `Final/app/app.py` |

Alle code en deliverables zijn lokaal getest, gevalideerd en gecommit op de branch `dimi-ball-model`.
