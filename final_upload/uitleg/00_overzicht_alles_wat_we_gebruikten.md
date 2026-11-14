# 00. Overzicht: alles wat we in dit project gebruikten

Dit bestand somt op wat er in het project gebruikt is: data, modellen, technieken, instellingen, bibliotheken en
hulpmiddelen. Bij elk onderdeel staat wat het is, hoe wij het gebruikten en waar je het terugvindt. De volgorde volgt de
pijplijn, van video tot dashboard. Voor de diepere uitleg verwijst elke sectie naar het uitleg-bestand en het notebook
van dat onderwerp.

Onderdelen die wel in de map zitten maar in de eindresultaten **niet** gebruikt worden, staan apart in sectie 12.

## 1. Het project in één alinea

Een clip van 22 seconden uit PSG tegen Inter (660 frames, 30 beelden per seconde, 1920 x 1080) wordt omgezet naar een
2D-minimap. Een eigen YOLO26-model vindt per frame de bal, de spelers van beide teams, de doelman van Inter en de
scheidsrechters. ByteTrack geeft elke speler een vast nummer, een Kalman-filter volgt de bal, en een homografie zet de
posities om van pixels naar meters op een veld van 105 x 68 m. Daaruit berekenen we balbezit en wedstrijdstatistieken,
en een Streamlit-app toont alles naast de video.

## 2. Data

| Wat | Uitleg | Hoe wij het gebruikten | Waar |
|---|---|---|---|
| **De clip** | Eén videofragment, één camerastandpunt dat meebeweegt met het spel | De enige bron van het hele project | `data/raw/match.mp4` |
| **Frames uitpakken** | Elk beeld van de video opslaan als losse afbeelding | 660 frames, genummerd zodat elk label terug op de tijdlijn te plaatsen is | notebook 01, sectie 2.1 |
| **Roboflow** | Online tool om boxen op beelden te tekenen en als dataset te exporteren | Het hele team labelde hierin; vier exports (51, 100, 118 en 136 beelden) | `data/datasets/` |
| **YOLO-labelformaat** | Per beeld een tekstbestand met per object: klasse, midden x, midden y, breedte, hoogte (genormaliseerd op 0-1) | Het formaat van alle datasets | `data/datasets/*/labels/` |
| **Vijf klassen** | `ball`, `player_A` (PSG), `player_B` (Inter), `player_B_goal_keeper`, `referee` | 3 305 boxen in 136 beelden: 105 ballen, 1 351 PSG, 1 348 Inter, 134 doelman, 367 scheidsrechter | `uitleg/02` |
| **Labelprotocol** | Afspraken zodat iedereen hetzelfde labelt | Strakke box zonder schaduw; bewegingsonscherpte meenemen; bij afdekking minstens 30 % zichtbaar; aan de rand meer dan de helft; alle objecten in een beeld labelen; kruiscontrole door een teamgenoot | `data/results/training/labeling_protocol.md` |
| **Split per tijdsblok** | Aaneengesloten stukken van de clip gaan in hun geheel naar train, validatie of test, omdat buurframes bijna identiek zijn | Gebruikt in de eerste runs en in de bal-fine-tune. Het finale model trainde op alle 136 beelden | notebook 01, sectie 2.5 |
| **Geklikte veldpunten** | Herkenbare punten van het veld (hoeken van het strafschopgebied, penaltystip, middellijn), aangeklikt in het beeld | 305 klikken op 39 keyframes; invoer voor de homografie | `data/keypoints.json` |
| **Ground truth** | Met de hand gelabelde waarheid om het hele systeem tegen te meten | 22 frames met een klik op het lichaam, 6 frames met een klik op de voeten; per frame ook de bal en het balbezit | `data/ground_truth.json`, `data/ground_truth_feet.json` |

## 3. Het detectiemodel

| Wat | Uitleg | Hoe wij het gebruikten | Waar |
|---|---|---|---|
| **YOLO** | Objectdetector die in één doorgang door één netwerk boxen, klassen en een zekerheid (confidence) voorspelt | De basis van de hele pijplijn | `uitleg/01` |
| **YOLO26** | De versie van ultralytics die wij gebruikten. Het netwerk heeft een backbone (kenmerken berekenen), een neck (schalen combineren) en een head (boxen voorspellen op stride 8, 16 en 32) | Vier groottes geprobeerd: n, s, m en l | notebook 01, sectie 3 |
| **YOLO26l** | De "large"-variant, 26,2 miljoen parameters | Het finale model | `models/psg_vs_inter_CV.pt` |
| **COCO-gewichten** | Gewichten van een model dat al getraind is op de COCO-dataset (ongeveer 118 000 beelden, 80 klassen) | Startpunt van elke training | `models/basis/yolo26l.pt` |
| **Transfer learning / fine-tunen** | Vertrekken van al getrainde gewichten en verder trainen op eigen data. Het netwerk kent dan al randen, vormen en personen | Daardoor volstaan 136 beelden. Ultralytics vervangt alleen de klassenuitgang (80 naar 5) | `uitleg/01`, sectie 3 |
| **Verliesfuncties** | `box_loss` (overlap van de box), `cls_loss` (juiste klasse), `l1_loss` (afstand van de boxranden) | Per epoch gevolgd in de trainingscurves | notebook 01, sectie 3.4 |

### De trainingsinstellingen van het finale model

| Instelling | Waarde | Wat het doet |
|---|---|---|
| `imgsz` | 960 | Beeldgrootte tijdens de training. Hoger geeft de kleine bal meer pixels, maar kost meer geheugen |
| `epochs` / `patience` | 100 / 25 | Maximaal 100 rondes over de data; vroeger stoppen als de score 25 rondes niet verbetert (early stopping) |
| `batch` | 2 | Twee beelden per stap; de gradiënten worden opgeteld tot een nominale batch van 64 |
| Optimizer | AdamW, learning rate 0,001111 | Door ultralytics zelf gekozen (`optimizer=auto`) |
| Warm-up | 3 epochs | De learning rate groeit eerst geleidelijk, zodat de COCO-gewichten niet meteen verstoord worden |
| `cos_lr` | aan | Daarna daalt de learning rate volgens een cosinus |
| `mosaic` | uit | Mosaic plakt vier beelden samen en maakt objecten kleiner, nadelig voor de bal |
| `mixup` | 0,15 | 15 % van de beelden wordt half doorzichtig gemengd met een ander beeld (tegen overfitting) |
| `scale` / `translate` | 0,2 / 0,1 | Willekeurig zoomen en verschuiven |
| `fliplr` | 0,5 | De helft van de beelden horizontaal gespiegeld |
| HSV-augmentatie | 0,015 / 0,7 / 0,4 | Kleine wijziging van tint, grotere van verzadiging en helderheid. De tint blijft bijna gelijk, want de shirtkleur bepaalt het team |

Er zijn negen trainingsruns gereconstrueerd (Kobe, Cisse en Brent, beeldgroottes 640 tot 1920). Het finale model is de
vierde run van Cisse. De training staat uitvoerbaar in notebook 01, sectie 6.

## 4. Het model beoordelen en afstellen

| Wat | Uitleg | Hoe wij het gebruikten | Waar |
|---|---|---|---|
| **IoU** | Intersection over union: de overlap van twee boxen gedeeld door hun unie | Een voorspelling telt als juist bij IoU ≥ 0,5 | `uitleg/03` |
| **Precision, recall, F1** | Welk deel van de meldingen juist is, welk deel van de echte objecten gevonden is, en het harmonisch gemiddelde van beide | Per klasse en per model | notebook 02, sectie 2 |
| **PR-curve en AP** | Precision tegenover recall bij elke confidence-drempel; AP is de oppervlakte eronder | Vergelijking van COCO, ons model en de bal-fine-tune | notebook 02, sectie 2 |
| **mAP50 en mAP50-95** | AP gemiddeld over de klassen bij IoU 0,5, en gemiddeld over IoU 0,5 tot 0,95 | Finale model: mAP50 0,895 (op de trainingsbeelden). Runs met aparte validatie: 0,67 tot 0,78 | notebook 01 en 02 |
| **Confusion matrix** | Tabel van echte klasse tegenover voorspelde klasse | Zelf berekend op de 15 testbeelden: 321 van 324 personen met de juiste rol | notebook 02 |
| **Confidence-drempel** | Boxen onder deze zekerheid worden weggegooid | Personen 0,3; bal 0,2. De baldrempel is gekozen op de hele pijplijn, niet op de F1-curve | notebook 02, sectie 3 en 7 |
| **NMS** | Non-maximum suppression: van overlappende boxen op hetzelfde object blijft alleen de zekerste | IoU-drempel 0,7. Een sweep van 0,3 tot 0,8 gaf geen verschil | notebook 02, sectie 4 |
| **Beeldgrootte bij inferentie** | De grootte waarop YOLO het frame bekijkt, los van de trainingsgrootte | 1280 px in plaats van 960: van 16 naar 19 van de 21 gelabelde ballen gevonden. De grootste winst van de afstelling | notebook 02, sectie 5 |
| **Vergelijking met het basismodel** | Hetzelfde testbeeld door het COCO-model en door ons model | COCO vindt geen enkele bal (AP50 0,000 tegenover 0,195) en kent geen teams | notebook 02, sectie 2 |
| **Data leakage** | Informatie uit de testset die in de training zat | Het finale model valideerde op zijn eigen trainingsbeelden en de testframes zaten in de trainingsdata. De detectiecijfers zijn dus te optimistisch | `uitleg/02`, `uitleg/03` |

## 5. Tracking van spelers

| Wat | Uitleg | Hoe wij het gebruikten | Waar |
|---|---|---|---|
| **Tracking-by-detection** | YOLO bekijkt elk frame apart; een tracker koppelt de boxen van opeenvolgende frames | Elke speler krijgt een vast track-ID | `uitleg/06` |
| **ByteTrack** | Tracker die boxen koppelt op overlap, in twee rondes: eerst de zekere detecties, daarna de zwakke | De versie uit ultralytics. 41 track-ID's voor 22 spelers en de scheidsrechters | `streamlit/core/tracking.py` |
| **Kalman-filter per track** | Voorspelt uit positie en snelheid waar een box in het volgende frame staat | Onderdeel van ByteTrack | notebook 04, sectie 2 |
| **Hongaars algoritme** | Zoekt de koppeling tussen twee groepen met de laagste totale kost | Koppelt voorspelde boxen aan detecties (bibliotheek `lap`) | notebook 04, sectie 2 |
| **Dubbelenfilter** | Van twee persoonsboxen met IoU boven 0,6 blijft alleen de zekerste | Voorkomt twee boxen op één speler | `DUPLICATE_IOU` |
| **Veldfilter** | Alleen personen met de voeten binnen het veld plus 3 m marge tellen mee | Publiek en bankspelers vallen weg | `PITCH_MARGIN` |

Instellingen: zekere detectie vanaf 0,3, zwakke vanaf 0,1, een verloren track blijft 30 frames (1 s) bewaard, maximale
koppelkost 0,8.

## 6. Teams en rollen

| Wat | Uitleg | Hoe wij het gebruikten | Waar |
|---|---|---|---|
| **Klasse van het model** | Het model voorspelt zelf PSG, Inter, doelman of scheidsrechter | De bron van het team | `streamlit/core/teams.py` |
| **Stemming per track** | Elke track krijgt de klasse met de hoogste som van confidences over al zijn frames | Eén fout frame draait het team niet om. 0,96 % van de detecties wijkt af van de gekozen groep | notebook 04, sectie 3 |

## 7. De bal

| Wat | Uitleg | Hoe wij het gebruikten | Waar |
|---|---|---|---|
| **Kandidaten filteren** | Niet elke baldetectie is een bal | Confidence ≥ 0,2, hoogstens 40 px groot, op het veld, en minstens 1 m van de middenstip en de penaltystippen. Van 922 detecties blijven 398 kandidaten over | `ball_candidate` |
| **Kalman-filter in meters** | Schat positie en snelheid uit onzekere metingen door afwisselend te voorspellen en te corrigeren | Eigen filter met toestand (x, y, vx, vy) in meters, zodat een snelheid overal op het veld hetzelfde betekent | `BallKalman` in `tracking.py` |
| **Gate (poort)** | Alleen kandidaten dicht genoeg bij de voorspelling tellen mee | 2,0 m, plus 1,5 m per frame zonder meting | notebook 04, sectie 4 |
| **Gate-fix en herstart** | Na een harde trap viel de bal buiten de oorspronkelijke poort | Poort verruimd, track blijft 30 frames bestaan, en een zekere kandidaat buiten de poort mag de track herstarten als de volgende frames hem bevestigen | notebook 04, sectie 4 en 5 |
| **Lineaire interpolatie** | Een kort gat tussen twee metingen opvullen met een rechte lijn | Gaten tot 15 frames (0,5 s) | `MAX_BALL_GAP` |

Resultaat met de standaardinstellingen: de bal is gedetecteerd in 308 frames, geïnterpoleerd in 191 en afwezig in 161.

## 8. Homografie en minimap

| Wat | Uitleg | Hoe wij het gebruikten | Waar |
|---|---|---|---|
| **Homografie** | Een matrix van 3 x 3 die een vlak (het beeld) afbeeldt op een ander vlak (het grasveld). Acht vrijheidsgraden, dus minstens vier puntparen nodig | Eén matrix per frame, omdat de camera beweegt | `uitleg/05` |
| **Homogene coördinaten** | Een punt (u, v) schrijven als (u, v, 1), zodat perspectief met een matrix te beschrijven is | De wiskunde achter de homografie | `uitleg/05`, sectie 2 |
| **`cv2.findHomography`** | De functie van OpenCV die de matrix schat uit puntparen | Per keyframe | `streamlit/core/homography.py` |
| **DLT** | Direct Linear Transform: de matrix oplossen als een lineair stelsel | Zelf nagebouwd in het notebook ter controle | notebook 03, sectie 2 |
| **RANSAC** | Schat het model op willekeurige deelverzamelingen en negeert uitschieters | Vanaf 5 punten, drempel 1,0 m op het veld. Geen enkele echte klik werd verworpen | notebook 03, sectie 2 |
| **Veldmodel** | De echte maten van een voetbalveld (105 x 68 m) met 31 benoemde punten | De doelcoördinaten van de homografie | `streamlit/core/pitch.py` |
| **Interpolatie tussen keyframes** | Een rooster van 30 veldpunten wordt in de twee omliggende keyframes geprojecteerd en lineair in de tijd gemengd; daarop wordt een nieuwe matrix gefit | Zo heeft elk van de 660 frames een matrix | notebook 03, sectie 4 |
| **Reprojectiefout** | De afstand tussen een omgerekende klik en de echte positie van dat punt | Mediaan 0,16 m per keyframe | notebook 03, sectie 5 |
| **Voetpunt** | Het midden van de onderrand van een box | De positie van een speler op het veld. Een homografie klopt alleen op het gras | `uitleg/05`, sectie 6 |
| **`cv2.warpPerspective`** | Een heel beeld omvormen met een homografie | Bovenaanzicht van een frame, als visueel bewijs | notebook 03, sectie 6 |
| **Minimap** | Getekend veld met een stip per speler en de bal | Getekend met OpenCV, naast het videobeeld | `streamlit/core/render.py` |

## 9. Balbezit en wedstrijdanalyse

| Wat | Uitleg | Hoe wij het gebruikten | Waar |
|---|---|---|---|
| **Balbezitregel** | De dichtste speler binnen een straal heeft de bal | Straal 2 m | `streamlit/core/possession.py` |
| **Hysterese** | Een wissel telt pas als ze een tijd aanhoudt | Wissel na 5 frames, bezit vervalt na 15 frames zonder speler bij de bal | notebook 04, sectie 6 |
| **Afvlakken** | Voortschrijdend gemiddelde over de posities | Venster van 0,5 s. Zonder afvlakken wordt de afstand 59 % overschat | notebook 05, sectie 2 |
| **Afstand en snelheid** | Som van de stappen tussen afgevlakte posities; snelheid over een span van 0,5 s | Per speler en per team | `streamlit/analytics/stats.py` |
| **Snelheidszones** | Stilstaand (< 2 km/h), wandelen (2-7), joggen (7-15), lopen (15-20), sprint (≥ 20) | Tijd en afstand per zone; een sprint duurt minstens 1 s | `streamlit/analytics/zones.py` |
| **Heatmap** | Posities tellen in vakken van 1 m en vloeiend maken met een gaussiaans filter | Per team en van de bal | `streamlit/analytics/plots.py` |
| **Aanvalsrichting** | Stemming van drie signalen: positie van de doelman, gemiddelde positie van de teams, beweging van de bal | PSG valt aan naar rechts | `streamlit/analytics/direction.py` |
| **Aanvallen** | De bal komt binnen 35 m van de doellijn (laatste derde) of in het 16-meter-gebied | PSG 5 ingangen in het laatste derde, 4 in het 16-meter-gebied | notebook 05, sectie 5 |

## 10. Systeemevaluatie

| Wat | Uitleg | Hoe wij het gebruikten | Waar |
|---|---|---|---|
| **Systeemevaluatie** | Het eindresultaat op het veld meten, niet alleen de boxen | Tegen de eigen ground truth van 28 frames | `uitleg/08` |
| **Labeltool** | Eigen programma om per frame personen, bal en balbezit aan te klikken | Maakte de ground truth | `streamlit/label_ground_truth.py` |
| **Koppelen met het Hongaarse algoritme** | Elke aangeklikte persoon krijgt hoogstens één box | Binnen 3 m (voetklik) of 0,6 boxhoogte (lichaamsklik) | `streamlit/evaluation/metrics.py` |
| **Gemeten** | Personen gevonden, juiste rol, positiefout, bal gevonden, balfout, balbezit | F1 0,99; juiste rol 99,2 %; positiefout gemiddeld 1,09 m; 19 van 21 ballen | notebook 06, sectie 4 |
| **Foutenanalyse** | Elke fout apart bekijken en de oorzaak zoeken | Twee gemiste ballen, één valse bal op 5,9 m, één ID-wissel | notebook 06, sectie 5 |

## 11. De app en de software eromheen

| Wat | Uitleg | Hoe wij het gebruikten | Waar |
|---|---|---|---|
| **Streamlit** | Bibliotheek om van een Python-script een webpagina met knoppen en schuifjes te maken | Het dashboard, met de tabs Frame, Video, Statistieken en Analyses | `streamlit/app/app.py` |
| **Scheiding van UI en logica** | De app tekent alleen; het rekenwerk staat in gewone functies | Dezelfde code draait in de app, de notebooks en de tests | `streamlit/core/` |
| **Detectiecache** | De YOLO-detecties worden één keer berekend en als json op schijf bewaard | De sleutel is een hash van de inhoud van model en video, zodat de cache meeverhuist naar een andere computer | `streamlit/cache/` |
| **Lage detectiedrempel** | YOLO draait één keer op confidence 0,05; de echte drempels komen daarna | Drempels aanpassen zonder YOLO opnieuw te draaien | `streamlit/core/detection.py` |
| **Tests** | Automatische controles van de code | 77 tests met pytest | `streamlit/tests/` |
| **Jupyter-notebooks** | Documenten met tekst, code en figuren door elkaar | Zes notebooks, uitgevoerd opgeslagen | `notebooks/` |

### Bibliotheken

| Bibliotheek | Waarvoor | Versie waarmee getest is |
|---|---|---|
| Python | De programmeertaal | 3.13 |
| ultralytics | YOLO26 trainen en gebruiken, ByteTrack | 8.4.182 |
| PyTorch | Het deep-learning-framework onder ultralytics | 2.14 |
| OpenCV (`opencv-python`) | Video lezen en schrijven, homografie, tekenen | 5.0 |
| NumPy | Rekenen met matrices | 2.5 |
| SciPy | Gaussiaans filter voor de heatmaps, Hongaars algoritme in de evaluatie | 1.18 |
| pandas | Tabellen en csv-bestanden | 3.1 |
| matplotlib | Grafieken | 3.11 |
| lap | Lineaire toewijzing voor ByteTrack | 0.5 |
| imageio-ffmpeg | De minimapvideo als H.264-mp4 schrijven | 0.7 |
| Streamlit | De app | 1.66 |
| pytest | De tests | 9.2 |

### Hardware en hulpmiddelen

- **GPU's voor de training:** de trainingen draaiden op de computers van de teamleden. De evaluatierun en de
  bal-fine-tune draaiden op een NVIDIA RTX 4070 Laptop (27 minuten). De hardware van de finale trainingsrun is niet
  genoteerd.
- **CUDA:** de laag waarmee PyTorch op een NVIDIA-GPU rekent.
- **Laptop zonder GPU:** de app, de notebooks en alle afstelling draaien op een processor, dankzij de detectiecache.
- **Git en GitHub:** versiebeheer en samenwerking, met een eigen werkmap en branch per teamlid.
- **Roboflow:** labelen en exporteren van de dataset.
- **PowerPoint:** de presentatie in `powerpoint/`.
- **AI-assistenten:** bij het programmeren, het uitwerken van de notebooks en de documentatie is hulp van
  AI-assistenten gebruikt.

## 12. Geprobeerd of aanwezig, maar niet gebruikt in de eindresultaten

| Wat | Uitleg | Waarom niet gebruikt | Waar |
|---|---|---|---|
| **Bal-fine-tune** | Het finale model verder getraind op 1280 px met klassengewichten (`cls_pw 0,5`), zodat de zeldzame bal zwaarder telt | De beste score viel in epoch 1 en zakte daarna; in de pijplijn niet beter dan het finale model | `models/experimenten/best_ball_ft.pt`, notebook 02, sectie 6 |
| **NMS-vrije kop** | YOLO26 kan ook zonder NMS één box per object geven | Gaf slechtere rollen en slechter balbezit in de pijplijn | notebook 02, sectie 4 |
| **Bal herdetecteren in uitsneden** | YOLO opnieuw draaien op een uitsnede rond de verwachte balpositie | Ingebouwd als optie, staat standaard uit; voegde in de test één bal toe | `streamlit/core/ball_refine.py` |
| **Extrapolatie en visuele trackers voor de bal** | De bal verder voorspellen zonder detectie, of volgen op uiterlijk | Werkten niet of gaven valse ballen | notebook 04, sectie 5 |
| **K-means op shirtkleur** | Spelers in twee teams verdelen door hun shirtkleur te clusteren | Alleen een terugval voor een model dat geen teams kent. Ons model voorspelt de teams zelf, dus deze code draait nooit | `assign_kmeans` in `teams.py` |
| **Kleinere modellen en andere beeldgroottes** | YOLO26n, s en m; training op 640, 1280, 1600 en 1920 px | Eerdere runs; het finale model is YOLO26l op 960 px | notebook 01, sectie 3 |

## 13. Waar vind je de uitleg per onderwerp?

| Onderwerp | Uitleg | Notebook |
|---|---|---|
| YOLO en transfer learning | `01_yolo_en_transfer_learning.md` | 01 |
| Dataset en labeling | `02_dataset_en_labeling.md` | 01 |
| Evaluatiematen | `03_evaluatie_metrics.md` | 02 |
| Fine-tunen en inferentie-instellingen | `04_fine_tuning_en_inferentie_instellingen.md` | 02 |
| Homografie | `05_homografie.md` | 03 |
| Tracking, bal en balbezit | `06_tracking_en_balbezit.md` | 04 |
| Wedstrijdanalyse | `07_wedstrijdanalyse.md` | 05 |
| Systeemevaluatie | `08_systeemevaluatie.md` | 06 |
| De Streamlit-app | `09_streamlit_app.md` | – |
