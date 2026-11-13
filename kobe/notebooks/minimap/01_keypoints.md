# Minimap 01 – Veldpunten aanklikken voor de homografie

Dit document legt uit hoe we van het camerabeeld naar een **2D-veld van bovenaf** gaan, en hoe je met
`click_keypoints.py` de punten aanklikt die daarvoor nodig zijn. Het klikken gebeurt in een apart venster
(OpenCV): dat werkt niet goed in een notebook, daarom is dit een script met deze uitleg erbij.

## 1. Het idee: homografie

YOLO vindt spelers en de bal **in het beeld** (pixels). Voor de minimap willen we weten waar ze **op het veld**
staan (meters). Het veld is een vlak, en de afbeelding van een vlak door een camera is een **homografie**:
een 3×3-matrix **H** die elk punt van het grasvlak in het beeld omzet naar zijn plaats op het veld.

```
[x_veld, y_veld, 1]  ∝  H · [x_pixel, y_pixel, 1]
```

H heeft 8 onbekenden, en elk punt dat we in beide systemen kennen geeft 2 vergelijkingen. Met **4 punten** is H dus
precies bepaald; met meer punten kiest `cv2.findHomography` de H die het best bij alle punten past.

**Vlakke-veld-aanname:** H klopt alleen voor punten óp het gras. Daarom gebruiken we van een speler zijn
**voeten** (onderkant van de box), niet zijn hoofd. Een bal in de lucht ligt niet op het vlak en komt op de
minimap te ver van de camera terecht.

**De camera beweegt:** in onze clip draait en zoomt de camera mee met het spel. H verschilt dus per frame. We klikken
punten op **keyframes** en berekenen H voor de frames ertussen door interpolatie (zie sectie 5).

## 2. Het veld in meters (`pitch.py`)

FIFA-afmetingen: 105 × 68 m (Allianz Arena, München). Oorsprong linksboven, zoals in het broadcastbeeld:

```
(0,0) ────────────── x = 105 m ──────────────►
  │   verre zijlijn (boven in beeld)
  │ ┌──────┐            │            ┌──────┐
  y │ L_box│      middencirkel      │R_box │
  │ └──────┘            │            └──────┘
  ▼   nabije zijlijn (onder in beeld)
 68 m
```

`pitch.py` bevat 31 benoemde punten waarvan de positie in meters vastligt, bv. `R_box_T` (hoek van het rechter
strafschopgebied, bovenaan), `R_penalty_spot`, `mid_T` (middenlijn × verre zijlijn), `circle_B` (middencirkel
onderaan), `R_arc_T` (waar de boog het strafschopgebied raakt). `L_` = linker helft, `R_` = rechter helft,
`_T` = bovenkant (verre zijlijn), `_B` = onderkant.

## 3. Punten aanklikken (`click_keypoints.py`)

Vanuit de hoofdmap van het project, met de gewone `venv`:

```powershell
venv\Scripts\python kobe\notebooks\minimap\click_keypoints.py                 # elk 30e frame
venv\Scripts\python kobe\notebooks\minimap\click_keypoints.py --frames 45 75   # extra keyframes
venv\Scripts\python kobe\notebooks\minimap\click_keypoints.py --step 15        # elk 15e frame
```

Er opent een venster: bovenaan het frame, onderaan links een minimap, in het midden de status, rechts een vergrootglas.

| Actie | Wat |
|---|---|
| klik een punt op de **minimap** (of `n` / `p`) | kies welk veldpunt je gaat aanklikken (rode ring) |
| **linkerklik** in het frame | plaats dat punt; het vergrootglas volgt de muis voor precisie |
| **rechterklik** | verwijder het dichtstbijzijnde punt |
| `x` / `c` | verwijder het geselecteerde punt / alle punten van dit frame |
| `spatie` / `enter`, `a` | volgende / vorige keyframe |
| `q` / `esc` / venster sluiten | stoppen (alles wordt na elke klik bewaard) |

**Wat je ziet terwijl je klikt:**
- vanaf **4 punten** worden de **cyaan veldlijnen** over het beeld getekend: ze moeten op de echte lijnen liggen,
  ook waar je niet geklikt hebt. Liggen ze ernaast, dan is een punt fout of verkeerd benoemd;
- vanaf **5 punten** krijgt elk punt zijn **fout in pixels**: groen < 5, oranje < 10, rood ≥ 10. Rood = misklik
  of verkeerde naam: verwijder of klik opnieuw;
- op een nieuw keyframe staan de lijnen van het vorige keyframe in **grijs** als hulp.

**Tips:**
- klik **5–8 punten verspreid over het zichtbare veld**, niet allemaal op één lijn (punten op één lijn bepalen geen vlak);
- goede punten in deze clip: hoeken van het rechter strafschop- en doelgebied, de strafschopstip, waar de boog het
  strafschopgebied raakt, de middenlijn met de zijlijn en de middencirkel;
- het script start bij het eerste keyframe met minder dan 4 punten: je kunt stoppen en later verdergaan.

**Resultaat:** `keypoints.json` (wordt gecommit: het is annotatiedata, net zoals de labels):

```json
{"pitch": "...", "frames": {"0": {"mid_T": [480.2, 240.1], "R_box_T": [1250.0, 255.3], ...}, "30": {...}}}
```

## 4. Controleren waar het nog niet klopt (`check_homography.py`)

```powershell
venv\Scripts\python kobe\notebooks\minimap\check_homography.py
```

Voor elk frame projecteert het de veldlijnen in het beeld en meet het welk deel van die lijnpunten op een **witte
pixel** ligt (± 4 px). Op een keyframe is dat het best haalbare (spelers en schaduwen verbergen een deel van de
lijnen). Tussen keyframes betekent een veel lagere score dat de geïnterpoleerde H **afwijkt**.

Het script schrijft `output/homography_check.png` (score per frame) en drukt kant-en-klare commando's af:
- **extra keyframes** waar de score onder 75 % van de keyframes zakt;
- **keyframes die zelf laag scoren**: hun kliks nakijken (bv. een keyframe met maar 4 punten, waarvan de fout niet te meten is).

Werkwijze: klikken → `check_homography.py` → de voorgestelde frames klikken → opnieuw controleren.

## 5. Hoe de homografie berekend wordt (`homography.py`)

- **Per keyframe** met ≥ 4 punten: `cv2.findHomography(beeldpunten, veldpunten)`. Met precies 4 punten is de
  oplossing exact. Vanaf 5 punten gebruiken we **RANSAC**: een punt dat meer dan 1 m afwijkt wordt genegeerd, zodat
  één misklik H niet scheeftrekt.
- **Tussen twee keyframes a en b:** een raster van 6×5 veldpunten wordt met H_a en H_b in het beeld geprojecteerd,
  die beeldposities worden lineair in de tijd geïnterpoleerd, en op dat geïnterpoleerde raster wordt H opnieuw
  berekend. Vóór het eerste en na het laatste keyframe gebruiken we het dichtstbijzijnde keyframe.
- Gebruik in code:

```python
from homography import Homographies
hs = Homographies()                                   # leest keypoints.json
feet_m = hs.image_to_pitch([(960, 700)], frame_idx=120)   # pixels -> meters
pixels = hs.pitch_to_image([(52.5, 34.0)], frame_idx=120) # meters -> pixels (bv. de middenstip)
```

## Bestanden in deze map

| Bestand | Wat |
|---|---|
| `pitch.py` | het veld in meters: keypoints, lijnen, minimap-afbeelding |
| `click_keypoints.py` | het klikvenster (sectie 3) |
| `keypoints.json` | de aangeklikte punten per keyframe |
| `homography.py` | H per frame (sectie 5) |
| `check_homography.py` | controle op drift (sectie 4) |
| `tracking.py` | ByteTrack en de Kalman-filter voor de bal (gebruikt in `02_minimap_video.ipynb`) |
| `render.py` | het tekenen van de video (infobalk, boxen, minimap) |
| `02_minimap_video.ipynb` | de volgende stap: de minimapvideo maken |
