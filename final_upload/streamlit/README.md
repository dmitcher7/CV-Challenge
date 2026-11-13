# Streamlit-app en analysecode: PSG vs Inter

Deze map bevat de interactieve app en alle code erachter: video -> detectie (YOLO26) -> tracking (ByteTrack) ->
homografie -> positie in meters op het veld -> balbezit -> statistieken. In de app pas je elke instelling van die
keten aan en zie je meteen het effect op de beelden, de minimap en de cijfers.

Alles werkt met paden die vanaf `final_upload/` berekend worden: je kan die map naar een andere computer of schijf
kopiëren en alles blijft werken. Het model, de clip, de aangeklikte veldpunten en de ground truth staan buiten deze
map (zie "Mapstructuur"). Een uitleg van de opbouw staat in `../uitleg/09_streamlit_app.md`.

## Installatie

Python 3.11 of nieuwer (getest met 3.13). In een virtuele omgeving, vanuit deze map:

```powershell
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
```

Op een laptop zonder GPU draait alles op de CPU. Dat is genoeg voor de app en de scripts: de zware detectie van
de hele clip zit al in `cache/`.

## De app starten

```powershell
python -m streamlit run app/app.py
```

Dit werkt ook vanuit een andere map (geef dan het volledige pad naar `app/app.py`), want de paden worden berekend
vanaf het bestand zelf. De app opent op `http://localhost:8501`.

Bij het opstarten staat alles op de finale keuze: model `psg_vs_inter_CV.pt`, de hele clip (frames 0 tot 659),
stride 1, imgsz 1280, NMS-IoU 0,7. Voor die combinatie bestaat al een detectiecache, dus de pagina staat binnen
enkele seconden klaar zonder dat YOLO draait.

## De zijbalk

De instellingen zijn verdeeld in drie groepen, naar wat er opnieuw berekend moet worden als je ze wijzigt.

| Groep | Instelling | Wat het doet | Kost bij wijzigen |
|---|---|---|---|
| Video | Video | `match.mp4` (de clip van de opdracht) of een eigen mp4. Voor een eigen video klopt de homografie niet, want de veldpunten zijn voor `match.mp4` aangeklikt (de app waarschuwt daarvoor). | Nieuwe detectie nodig |
| 1. Model | Eigen .pt uploaden | Bewaart het bestand in `models/` en zet het in de lijst. | Geen |
| 1. Model | Gewichten | Keuze uit de `.pt`-bestanden in `models/`; het finale model staat standaard geselecteerd. Met het vinkje "Toon ook experimentele modellen" komen de bestanden uit `models/experimenten/` erbij (nu de mislukte bal-fine-tune `best_ball_ft.pt`, alleen om te vergelijken). | Nieuwe detectie |
| 1. Model | Inferentie-imgsz (640, 960, 1280) | Resolutie waarop YOLO het beeld ziet. Hoger vindt de kleine bal vaker. 960 is de trainingsresolutie. | Nieuwe detectie |
| 1. Model | NMS IoU (0,3 tot 0,9) | Overlapdrempel waarboven YOLO twee boxen samenvoegt. | Nieuwe detectie |
| 1. Model | Framebereik, Stride | Welk deel van de clip en welke frames (elke n-de). | Nieuwe detectie |
| 1. Model | Knop Detecteer | Draait YOLO eenmalig op confidence 0,05 en bewaart alle boxen op schijf. | Traag op CPU, daarna uit de cache |
| 2. Post-processing | Confidence spelers | Minimale confidence om een spelertrack te starten. | Snel (ongeveer 4 s voor de hele clip) |
| 2. Post-processing | Confidence bal | Aparte drempel voor balkandidaten (standaard 0,2). | Snel |
| 2. Post-processing | Tracker: lost buffer, matching-drempel | Hoe lang een verloren track bewaard blijft en hoe makkelijk een box aan een track gekoppeld wordt. | Snel |
| 2. Post-processing | Bal: max. grootte, max. gat, confidence om een baltrack te starten | Filters en interpolatie voor de baltrack. | Snel |
| 2. Post-processing | Herdetectie in crops | Zoekt de bal opnieuw met YOLO in een uitsnede rond de verwachte positie, alleen in frames zonder bal. Start pas na een klik op "Herdetecteer bal". | Traag de eerste keer (YOLO op uitsneden), daarna uit de cache |
| 3. Weergave | Preview-breedte | Grootte van de afbeelding in de pagina. | Direct |

Detectie en post-processing zijn bewust gescheiden: YOLO is traag, het filteren, tracken en berekenen van posities
niet. Daarom wordt alleen de detectie op schijf gecachet en wordt de rest bij elke wijziging opnieuw berekend. Heeft
een combinatie uit groep 1 geen cache, dan toont de zijbalk een waarschuwing en gebeurt er niets tot je op
**Detecteer** klikt.

## De tabs

- **Frame**: schuif door de frames. Je ziet het beeld met boxen, track-ID's en veldlijnen, met de 2D-minimap
  (spelers per team, bal, balbezit tot dat frame).
- **Video**: de knop "Render video" schrijft een H.264-mp4 (beeld naast minimap) en biedt hem aan als download.
  Het resultaat wordt in `output/` bewaard en hergebruikt zolang de parameters gelijk blijven.
- **Statistieken**: aantal frames en tracks, percentage frames met een gedetecteerde bal, balbezit per team,
  downloads van `tracks.csv` en `frames.csv`, en een tabel met de gebruikte parameters.
- **Analyses**: aanvalsrichting (stemming van drie signalen), aanvallen en ingangen van het laatste derde en het
  16 m-gebied, snelheidszones per team, kilometers (ondergrens) en heatmaps per team met de aanvalsrichting.
  Berekend op het huidige resultaat; bij een kort framebereik zijn de cijfers zeer grof.

## Losse scripts

Draai de scripts vanuit deze map. `<model>` in de bestandsnamen is de naam van het gewichtenbestand zonder `.pt`,
dus `psg_vs_inter_CV`.

| Script | Wat het doet |
|---|---|
| `run_all.py [--no-video]` | Alles in één keer: minimap en csv's, analyses, evaluatie tegen de ground truth en het rapport `output/RAPPORT_<model>.md`. Met de cache duurt het ongeveer 5 s met `--no-video` en ongeveer 30 s met video. |
| `run_minimap.py` | De minimapvideo (`output/minimap_<model>.mp4`) en de csv's. Opties: `--start`, `--end`, `--stride`, `--imgsz`, `--no-video`, `--ball-refine`. |
| `run_analytics.py` | Analyses op de bestaande csv's, naar `output/analytics/`. Standaard `--model psg_vs_inter_CV`. |
| `evaluate_system.py` | Vergelijkt de csv's met `../data/ground_truth.json` (`--anchor body`) of `../data/ground_truth_feet.json` (`--anchor feet`); schrijft `output/evaluation/`. |
| `check_homography.py` | Controleert hoe goed de homografie de aangeklikte punten terugprojecteert; schrijft `output/homography_check.csv`. |
| `click_keypoints.py` | Hulpmiddel om veldpunten op keyframes aan te klikken; schrijft `../data/keypoints.json`. Interactief (opent een venster). |
| `label_ground_truth.py` | Hulpmiddel om spelers en bal op frames aan te klikken; schrijft `../data/ground_truth.json`. Interactief (opent een venster). |

Let op: een combinatie van imgsz, IoU, framebereik, stride of gewichten zonder cache start YOLO. Voor de hele clip
is dat op een CPU te traag; gebruik de combinaties die in `cache/` zitten (hieronder) of een kort framebereik.

## Mapstructuur

```
final_upload/
├── models/psg_vs_inter_CV.pt        het finale model (experimenteel: models/experimenten/best_ball_ft.pt)
├── data/
│   ├── raw/match.mp4                de clip (1920x1080, 660 frames)
│   ├── keypoints.json               aangeklikte veldpunten per keyframe (voor de homografie)
│   ├── ground_truth.json            22 frames, personen op het lichaam geklikt
│   └── ground_truth_feet.json       6 frames, voeten geklikt
└── streamlit/                       deze map
    ├── app/app.py                   alleen UI: zijbalk en tabs
    ├── core/                        de keten: paths, video, detection, teams, tracking, homography, pitch,
    │                                possession, ball_refine, pipeline, render
    ├── analytics/                   afstand, snelheid, zones, kilometers, aanvalsrichting, heatmaps, rapport
    ├── evaluation/                  ground-truthformaat en metrics (F1, positiefout, balbezit, ID-wissels)
    ├── tests/                       77 tests (pytest), waarvan 1 smoke-test met het echte model
    ├── cache/                       detecties op schijf (zie hieronder)
    ├── output/                      csv's, rapport, analyses, evaluatie en minimapvideo van het finale model
    └── run_*.py, evaluate_system.py, check_homography.py, click_keypoints.py, label_ground_truth.py
```

Alle paden staan op één plek: `core/paths.py`.

### De cache

`cache/` bevat de detecties van de hele clip (frames 0 tot 659, stride 1) voor het finale model bij: imgsz 1280 met
NMS-IoU 0,7 (de standaard van de app), imgsz 960 met IoU 0,7, imgsz 960 met IoU 0,5 en imgsz 960 zonder NMS (de
NMS-vrije kop van YOLO26). Daarnaast staan er detecties voor `best_ball_ft.pt` (960 en 1280) en de uitsneden voor
de bal-herdetectie. De sleutel in de bestandsnaam komt uit naam, grootte en md5 van model en video plus de
parameters, niet uit een absoluut pad of tijdstip: een gekopieerde map hergebruikt zijn cache, een ander model met
dezelfde bestandsnaam niet. Een cachebestand verwijderen is veilig; het wordt bij de volgende detectie (met YOLO)
opnieuw gemaakt.

## Datacontract van de csv's

Coördinaten zijn in meters op het veld: x van 0 tot 105 langs de zijlijn (oorsprong: verste zijlijn, linker doellijn),
y van 0 tot 68 dwars erop. `time_s` is het framenummer gedeeld door de fps van de video.

`output/tracks_<model>.csv`: één rij per getrackte persoon per verwerkt frame.

| Kolom | Betekenis |
|---|---|
| `frame`, `time_s` | framenummer en tijd in seconden |
| `track_id` | ID van de ByteTrack-track |
| `group` | `player_A`, `player_B`, `player_B_goal_keeper` of `referee` (bij een COCO-model `team 1`, `team 2`, `other`) |
| `team` | `A` (PSG), `B` (Inter, keeper inbegrepen) of `-` (scheidsrechter of overig) |
| `x_m`, `y_m` | positie van de voeten (onderkant-midden van de box) op het veld |
| `conf` | detectieconfidence |
| `box_x1`, `box_y1`, `box_x2`, `box_y2` | box in beeldpixels |

`output/frames_<model>.csv`: één rij per verwerkt frame.

| Kolom | Betekenis |
|---|---|
| `frame`, `time_s` | zoals hierboven |
| `ball_x_m`, `ball_y_m` | positie van de bal (leeg als er geen bal is) |
| `ball_state` | `detected`, `interpolated`, `refined` (gevonden in een uitsnede) of `none` |
| `possession_team` | `A`, `B` of `-` (gladgestreken) |
| `possession_track_id` | track-ID van de speler het dichtst bij de bal terwijl zijn team balbezit heeft, anders leeg |

## Resultaten van het finale model (uit `output/`)

Hele clip, stride 1, imgsz 1280: 660 frames, 41 tracks, bal gedetecteerd in 308 van 660 frames en na interpolatie
aanwezig in 499. Balbezit PSG 92 %, Inter 8 %. Tegen de handmatige ground truth: speler-F1 0,986 (22 frames, op het
lichaam geklikt) en 0,989 (6 frames, op de voeten geklikt, gemiddelde positiefout 1,09 m). Het volledige rapport
staat in `output/RAPPORT_psg_vs_inter_CV.md`.

## Tests

```powershell
venv\Scripts\python -m pytest -q                  # alle 77 tests, ongeveer 20 s
venv\Scripts\python -m pytest -q -m "not smoke"   # zonder de smoke-test (76 tests)
```

De smoke-test draait het echte model op 10 frames en wordt overgeslagen als model, video of `keypoints.json` ontbreekt.

## Beperkingen

- De homografie geldt alleen voor `match.mp4` (de keyframes zijn aangeklikt, daartussen wordt geïnterpoleerd). Een
  eigen video geeft een onjuiste minimap.
- De clip duurt 22 s. Kilometers, snelheidszones en "per 90 minuten" zijn daardoor indicatief. Kilometers zijn een
  ondergrens: ID-wissels verdelen een speler over meerdere tracks en korte tracks (onder 1 s) vallen weg.
- Tracking heeft veel ID-wissels (36 bij 75 personen in de voeten-evaluatie, IDF1-lite 0,37): het aantal tracks (41)
  is groter dan het aantal personen op het veld.
- Balbezit is getest op 22 en 6 frames (0,77 en 0,67 juist). Met zo weinig frames zijn die cijfers zeer onzeker.
- De bal is klein: in 161 van 660 frames is er geen bal (noch gedetecteerd, noch geïnterpoleerd).
- Schoten worden niet geteld (te onbetrouwbaar in trackingdata).
- Een geüploade `.pt` komt in `models/`, een geüploade video in `cache/uploads/`.
