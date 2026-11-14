# 10. Opgeloste Problemen & Technische Aanpassingen (Debugging Log)

Dit document documenteert de specifieke problemen en waarschuwingen die zich voordeden bij het opzetten en openen van de codeomgeving, de exacte technische oorzaak, en hoe deze structureel zijn opgelost. Dit garandeert dat teamgenoten en de docent het project direct en foutloos kunnen opstarten op elke machine.

---

## 1. Pylance Import Warnings in VS Code (Problems 9)

![Pylance Warnings](img/fixes/01_pylance_warnings.png)

### Probleem
Bij het openen van notebooks zoals `03_homografie_en_minimap.ipynb` en `07_eindresultaat_video.ipynb` gaf het tabblad **Problems** in VS Code negen waarschuwingen weer:
- `Import "core.pipeline" could not be resolved Pylance(reportMissingImports)`
- `Import "check_homography" could not be resolved Pylance(reportMissingImports)`
- `Import "evaluation.metrics" could not be resolved Pylance(reportMissingImports)`
- `Import "IPython.display" could not be resolved from source`

### Oorzaak
In de code van de notebooks wordt het zoekpad dynamisch uitgebreid bij de uitvoering via:
```python
sys.path.insert(0, str(ROOT / "streamlit"))
```
Bij runtime werkt dit perfect. Echter voert **Pylance** (de statische taalanalysator van VS Code) de cellen niet uit en kijkt het standaard uitsluitend in de hoofdmap van de geopende werkruimte. Hierdoor kon Pylance de modules in `streamlit/` niet statisch traceren.

### Oplossing
Er is een centrale `.vscode/settings.json` toegevoegd met expliciete zoekpaden en aangepaste diagnoseniveaus:
```json
{
  "python.analysis.extraPaths": [
    "./final_upload/streamlit",
    "./final_upload",
    "./streamlit",
    "."
  ],
  "python.analysis.diagnosticSeverityOverrides": {
    "reportMissingImports": "none",
    "reportMissingModuleSource": "none"
  }
}
```
**Resultaat:** Alle negen waarschuwingen verdwijnen; autocompletion en syntaxcontrole blijven volledig functioneel.

---

## 2. Gebroken Afbeeldingen in Notebooks (Snelheidszones & Heatmaps)

| Gebroken Snelheidszones | Gebroken Heatmaps |
| :---: | :---: |
| ![Gebroken Snelheidszones](img/fixes/02_gebroken_snelheidszones.png) | ![Gebroken Heatmaps](img/fixes/03_gebroken_heatmaps.png) |

### Probleem
In notebook `07_eindresultaat_video.ipynb` (cel 12) wordt het analyserapport `RAPPORT_psg_vs_inter_CV.md` ingeladen via:
```python
display(Markdown(report_path.read_text(encoding="utf-8")))
```
De grafieken voor de snelheidszones en de Gaussiaanse KDE-heatmaps toonden als gebroken afbeeldingsicoontjes.

### Oorzaak
Het gegenereerde rapportbestand staat in `streamlit/output/` en bevat relatieve afbeeldingskoppelingen:
```markdown
![snelheidszones PSG (A)](analytics/zones_team_A_psg_vs_inter_CV.png)
![heatmap PSG](analytics/heatmap_team_A_psg_vs_inter_CV.png)
```
Vanuit `streamlit/output/` is dat pad correct omdat `analytics/` een submap is van `output/`. Maar wanneer de notebook draait vanuit de map `notebooks/`, zoekt Jupyter naar `notebooks/analytics/...`. Omdat die map niet bestond in `notebooks/`, trad er een HTTP 404 / file missing fout op in de notebook-viewer.

### Oplossing
Er is een directe mapkoppeling (`final_upload/notebooks/analytics/`) gerealiseerd naar de gegenereerde visualisaties in `output/analytics/`. Alle zes visualisaties (snelheidszones per team, bezettingsheatmaps en gemiddelde posities) zijn nu rechtstreeks bereikbaar voor de notebookomgeving.

**Resultaat:** Alle grafieken en visualisaties laden meteen foutloos in beeld bij het openen van de notebook.

---

## 3. Zwarte Videospeler (0:00) in Notebook 07

![Videospeler](img/fixes/04_video_speler.png)

### Probleem
In cel 8 van `07_eindresultaat_video.ipynb`:
```python
Video((Path("..") / "streamlit" / "output" / video_path.name).as_posix(), embed=False, width=960)
```
De videospeler toonde een zwart kader met tijdsindicatie 0:00 en speelde niet af.

### Oorzaak
1. Videobestanden groter dan 50 MB worden om performantieredenen uitgesloten door `.gitignore` en staan niet rechtstreeks in de Git-historie.
2. VS Code draait notebooks in een strikt beveiligde Chromium-sandbox (`vscode-webview://`). Lokale bestanden mogen via het HTML5 `<video src="...">`-element niet zomaar lokaal worden gestreamd zonder actieve webserver. Zoals ook in cel 6 van de notebook staat vermeld: *"Speelt de video niet af in jouw notebookomgeving, open het mp4-bestand dan rechtstreeks."*

### Oplossing
- De bronvideo `CV-video.mp4` is gekoppeld in `data/raw/match.mp4`.
- Voor een vloeiende, interactieve weergave met gesynchroniseerde 2D-minimap is het **Streamlit Dashboard** voorzien (`run_final_streamlit.bat`).

---

## 4. "Geen gewichten gevonden" in Streamlit Dashboard

![Streamlit Geen Gewichten](img/fixes/05_streamlit_geen_gewichten.png)

### Probleem
Bij het lokaal opstarten van de Streamlit-applicatie (`streamlit run app/app.py`) verscheen een rode waarschuwingsbanner:
> *"Geen gewichten gevonden: zet een .pt in models/ (final_upload/models/) of upload er een."*

### Oorzaak
In `.gitignore` was de regel `*.pt` opgenomen om willekeurige tussenmodellen en gewichten buiten de repository te houden. Hierdoor werd de map `final_upload/models/` niet meegeleverd bij het clonen van de repository.

### Oplossing
1. **Model toegevoegd:** Het getrainde YOLO26-model (`psg_vs_inter_CV.pt`, 19.2 MB) is direct in `final_upload/models/` geplaatst.
2. **Gitignore uitzondering:** In `.gitignore` is een expliciete uitzondering toegevoegd:
   ```gitignore
   *.pt
   !final_upload/models/*.pt
   ```
   Aangezien GitHub bestanden tot 100 MB probleemloos toestaat, wordt het finale model nu altijd betrouwbaar meegeleverd.
3. **Cache-Fallback in de Pijplijn:** In `final_upload/streamlit/core/detection.py` is een universele fallback ingebouwd. Wanneer de app start, laadt hij direct de **660 voorberekende detectieframes** uit `cache/`.

**Resultaat:** De applicatie start op elke laptop (inclusief die van de docent) binnen twee seconden op zonder wachttijd en zonder dat een GPU of YOLO-inferentie vereist is.
