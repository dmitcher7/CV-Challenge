# CV-Challenge: PSG vs Inter, van video tot 2D-minimap

Inlevermap van team **Git Push and Pray** (Dmitrij Tchervonisjenko, Kobe Vandenberk, Cisse Vandeweyer,
Brent Van Gelder), Thomas More Geel, vak Deep Learning & Computer Vision.

Een clip van ±22 s (660 frames) wordt omgezet naar een 2D-minimap: detectie met een eigen YOLO26-model,
tracking, homografie naar meters, balbezit en wedstrijdanalyses, met een Streamlit-app erbovenop.

## Inhoud

| Map | Wat |
|---|---|
| `notebooks/` | Zes uitgevoerde notebooks met uitleg, grafieken en vergelijkingen (zie hieronder) |
| `uitleg/` | Een overzicht van alles wat gebruikt is (`00_...`) en negen uitleg-bestanden per onderwerp, met illustraties |
| `powerpoint/` | De presentatie (9 dia's, met sprekersnotities) |
| `models/psg_vs_inter_CV.pt` | Het finale detectiemodel (YOLO26l, imgsz 960, 100 epochs, 5 klassen) |
| `models/experimenten/` | De mislukte bal-fine-tune, alleen ter vergelijking |
| `models/basis/yolo26l.pt` | Het COCO-basismodel waarvan de training vertrekt |
| `data/raw/match.mp4` | De clip |
| `data/` | Veldpunten (`keypoints.json`), ground truth, de datasets en de resultaatbestanden die de notebooks inlezen (`results/`) |
| `streamlit/` | De app en de volledige pijplijn (`core/`, `analytics/`, `evaluation/`), met eigen `README.md` |

## Notebooks

| Notebook | Onderwerp |
|---|---|
| `01_model_training.ipynb` | Dataset, labeling, alle trainingsruns, hoe het finale model tot stand kwam, en de training zelf (sectie 6) |
| `02_fine_tuning.ipynb` | PR-curves (COCO, ons model, fine-tune), drempels, NMS, beeldgrootte, de bal-fine-tune |
| `03_homografie_en_minimap.ipynb` | Van pixels naar meters, reprojectiefout, drift, de minimap |
| `04_tracking_en_balbezit.ipynb` | ByteTrack, teams, de bal (Kalman, interpolatie, experimenten), balbezit |
| `05_match_analyse.ipynb` | Afstand, snelheid, zones, heatmaps, aanvalsrichting |
| `06_systeemevaluatie.ipynb` | Het hele systeem gemeten tegen handgelabelde ground truth, met foutenanalyse |
| `07_eindresultaat_video.ipynb` | Draait de hele pijplijn en schrijft de minimapvideo, de csv's en het rapport weg |

De notebooks zijn uitgevoerd opgeslagen: alle uitvoer en figuren staan erin.

## De notebooks als keten

Voer de notebooks in volgorde uit, van 01 tot 07. Ze werken alle op één bestand, het **actieve model**
`models/psg_vs_inter_CV.pt`:

- **Notebook 01** traint het model vanaf de COCO-gewichten, met de trainingsdata (`data/datasets/final_train_v4/`, de
  136 gelabelde beelden) en het basismodel uit deze map. Met een NVIDIA-GPU is dat de volledige training (100 epochs,
  ongeveer een uur); het nieuwe model wordt daarna het actieve model en het meegeleverde model wordt bewaard in
  `models/referentie/`. Zonder GPU draait een demonstratierun van 1 epoch die niets vervangt.
- **Notebook 02 tot 06** rekenen met het actieve model en melden bovenaan welk model dat is. Voor een nieuw model
  berekenen ze de YOLO-detecties zelf opnieuw (seconden per instelling op een GPU, minuten op een CPU); voor het
  meegeleverde model komen de detecties uit `streamlit/cache/`, zodat alles ook zonder GPU draait.
- **Notebook 07** schrijft het eindresultaat weg in `streamlit/output/`: de minimapvideo, de csv's en het rapport.

De begeleidende tekst in de notebooks beschrijft het meegeleverde model. Met een hertraind model wijken de cijfers
in de uitvoer daar licht van af. De opgeslagen uitvoer komt van een run op een laptop zonder GPU, dus met het
meegeleverde model; een volledige hertraining vanuit deze map is daar niet uitgevoerd.

Notebook 02 toont daarnaast de opgeslagen resultaten van de GPU-run (`data/results/gpu_run/`: de PR-curves, de
NMS-sweep en de bal-fine-tune). Die horen bij het meegeleverde model en worden niet opnieuw berekend.

## Gebruik

```powershell
python -m venv venv
venv\Scripts\pip install -r requirements.txt

# de app
cd streamlit
streamlit run app/app.py

# een notebook opnieuw uitvoeren (werkmap = notebooks/)
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_model_training.ipynb
```

## Belangrijkste resultaten

- Inferentie-instellingen: imgsz 1280, baldrempel 0,2. Daarmee vindt het systeem 19 van de 21 handgelabelde ballen.
- Spelers: F1 0,99 tegen de ground truth; gemiddelde positiefout 1,09 m.
- Homografie: reprojectiefout mediaan 0,16 m op de controleerbare keyframes.

## Eerlijke kanttekeningen

- Het finale model valideerde op zijn eigen trainingsbeelden en de testframes zaten in de trainingsdata
  (leakage): de mAP-cijfers zijn te optimistisch. Zie notebook 01 en 02.
- Eén korte clip van één wedstrijd, met de hand geklikte veldpunten: geen algemene uitspraken.
- De oorspronkelijke logbestanden en `args.yaml`'s in `data/results/` zijn ongewijzigde kopieën van de
  trainingsruns en bevatten daardoor de paden van de computers waarop getraind is.
