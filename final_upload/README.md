# CV-Challenge: PSG vs Inter, van video tot 2D-minimap

Inlevermap van team **Git Push and Pray** (Dmitrij Tchervonisjenko, Kobe Vandenberk, Cisse Vandeweyer,
Brent Van Gelder), Thomas More Geel, vak Deep Learning & Computer Vision.

Een clip van ±22 s (660 frames) wordt omgezet naar een 2D-minimap: detectie met een eigen YOLO26-model,
tracking, homografie naar meters, balbezit en wedstrijdanalyses, met een Streamlit-app erbovenop.

## Inhoud

| Map | Wat |
|---|---|
| `notebooks/` | Zes uitgevoerde notebooks met uitleg, grafieken en vergelijkingen (zie hieronder) |
| `uitleg/` | Negen uitleg-bestanden per onderwerp (theorie + hoe wij het toepasten) |
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

De notebooks zijn uitgevoerd opgeslagen: alle uitvoer en figuren staan erin. Ze draaien opnieuw zonder GPU,
omdat de YOLO-detecties van de clip in `streamlit/cache/` staan.

## Het model opnieuw trainen

De trainingsdata (`data/datasets/final_train_v4/`, de 136 gelabelde beelden van het finale model), het basismodel en
de trainingscode (in notebook 01) zitten in deze map.

Notebook 01, sectie 6, bevat de volledige trainingscode: met een GPU draait de volledige training, zonder GPU de
demonstratierun. De opgeslagen uitvoer in het notebook komt van een demonstratierun; de volledige training vanuit
deze map is niet uitgevoerd omdat er geen GPU beschikbaar was. Het resultaat komt in `models/runs/`;
`models/psg_vs_inter_CV.pt` wordt niet overschreven.

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
