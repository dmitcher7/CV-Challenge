# Plan: match-analyse (zelfstandige analysemap)

**Status:** OPGELEVERD
**Opdracht:** rechtstreeks van de eigenaar in de chat (2026-11-13), op basis van de audit van `kobe/` tegen `DL_course_summary.md`
**Research:** geen apart bestand; de audit staat in het verslag onderaan
**Waar bouwen:** `kobe/match_analyse/` (branch `kantoor/match-analyse`)

## 1. Doel
Eén zelfstandige map `kobe/match_analyse/` die met alleen de venv werkt: video → detectie (nieuw model `best.pt`) → tracking → homografie → minimap → analyses, plus de Streamlit-app en een hulpmiddel om het hele systeem te evalueren. De dubbele code uit `homography/`, `notebooks/minimap/` en `stream-lit/` wordt samengevoegd tot één pakket.

## 2. Context voor de developer
- De eigenaar traint zelf geen modellen (geen GPU); teamgenoten leveren de `.pt`. Geen training in scope.
- Nieuw model: yolo26l, imgsz 960, 5 klassen (`ball`, `player_A`, `player_B`, `player_B_goal_keeper`, `referee`).
- Gekozen door de eigenaar: mapnaam `match_analyse`; scope = minimap + tracking, analyses, systeemevaluatie-tool, Streamlit-app; modelevaluatie (COCO-baseline, conf-sweep) valt **buiten** scope.

## 3. Technische aanpak
Python-pakket `core/` (paden, veld, homografie, detectie + cache, teams, tracking, bal, balbezit, rendering), scripts in de map, `app/` voor Streamlit, `analytics/`, `evaluation/`, `tests/` (pytest). Alle paden relatief aan de map. Werk verdeeld over Sonnet-agents; Opus coördineert en controleert.

## 4. Stappen
- [x] **Stap 1:** branch, map hernoemen, model/video/keypoints kopiëren → `kobe/match_analyse/models/`, `data/`
- [x] **Stap 2:** gedeeld pakket + minimap-script + tests (Sonnet) → `core/`, `run_minimap.py`, `tests/`
- [x] **Stap 3a:** Streamlit-app op het gedeelde pakket (Sonnet) → `app/`
- [x] **Stap 3b:** analyses: afstand, snelheid, heatmaps (Sonnet) → `analytics/`
- [x] **Stap 3c:** systeemevaluatie-tool: labelhulp + foutberekening (Sonnet) → `evaluation/`
- [x] **Stap 4:** end-to-end controle, `README.md` + `WIJZIGINGEN.md` (Opus)
- [x] **Stap 5:** archivaris werkt logboek en geschiedenis bij

## 5. Acceptatiecriteria
- [x] De map werkt los van de rest van de repo (geen imports of paden buiten de map, behalve de venv).
- [x] `run_minimap.py` maakt een minimapvideo + frames-CSV met het nieuwe model.
- [x] `pytest` slaagt, met echte asserts.
- [x] Analyses geven afstand/snelheid/heatmap in meters.
- [x] Evaluatietool: foutberekening werkt op het voorbeeld; labeltool alleen op hulpfuncties getest (GUI niet aangeklikt).
- [x] App start en draait op het gedeelde pakket.

## 6. Buiten scope
Modellen trainen; modelevaluatie (COCO-baseline, conf-sweep); de oude mappen aanpassen of verwijderen.

## Vragen voor de eigenaar

## Verslag developer
Gebouwd in `kobe/match_analyse/` (branch `kantoor/match-analyse`, niet gecommit) door 4 Sonnet-agents (core, app, analyses, evaluatie), gecoördineerd en gecontroleerd door Opus. Volledig overzicht: `kobe/match_analyse/WIJZIGINGEN.md`.

- **Gebouwd:** gedeeld pakket `core/`, `run_minimap.py`, `check_homography.py` (nu met reprojectiefout in meters + spreidingscheck), `click_keypoints.py`, `analytics/` + `run_analytics.py`, `evaluation/` + `label_ground_truth.py` + `evaluate_system.py`, `app/app.py`, 49 pytest-tests, `README.md`, `WIJZIGINGEN.md`.
- **Gecontroleerd:** alle 49 tests slagen (ook in een kopie van de map buiten de repo); volledige clip gedraaid met het nieuwe model (660 frames, 48 tracks, bal 302 gedetecteerd + 119 geïnterpoleerd, balbezit PSG 93 % / Inter 7 %); homografie-RMSE 0,07–0,28 m op 30 controleerbare keyframes; app start zonder fouten (headless).
- **Afwijkingen:** één tracker (ultralytics BYTETracker, `supervision` niet meer nodig); bal overal in meters; teams uit modelklassen (K-means alleen als terugval); enkele opties van de oude app geschrapt (zie WIJZIGINGEN.md).
- **Open:** ground truth moet de eigenaar zelf labelen; 10 laag scorende keyframes herklikken; K-means-terugval niet getest met een echt COCO-model; oude mappen ongewijzigd.

**Aanvulling (2026-11-13, middag):** de eigenaar labelde 22 frames (lichaamsklik) en 6 frames (voetklik). Toegevoegd: boxkolommen in `tracks_*.csv`, `evaluate_system.py --anchor body`, waarschuwing bij niet-voetklikken, rapport per modus; 53 tests slagen. Resultaat: spelers F1 0,99, rol 99,6–100 %, bal gevonden 71–75 % (fout ±0,2 m), balbezit 67–68 %, positiefout spelers 0,5–1,1 m (1,1 m bovengrens door klikken iets boven de voeten). ID-switches niet meetbaar met deze labels. Details: `kobe/match_analyse/WIJZIGINGEN.md`.

**Aanvulling (2026-11-13, balvoorspelling):** poortfix in `core/tracking.py` (standaard aan: poortgroei 1,5 m/frame, 30 gemiste frames, herstart met bevestiging) en optionele crop-herdetectie `core/ball_refine.py` (`--ball-refine`, checkbox in de app, `ball_state = refined`, cache in `cache/`). Bal gevonden op GT 15/21 → 16/21 (17/21 met refine); bal gemeld zonder zichtbare bal 4/7 → 5/7 (frame 270: bal achter de voeten van een dribbelaar, geen verzonnen bal); balbezit body 68 % → 73 %. Standaarduitvoer zonder refine; 63 tests slagen. Details: `kobe/match_analyse/WIJZIGINGEN.md`, sectie "Balvoorspelling".

**Aanvulling (2026-11-13, avond):** GPU-run van teamgenoot (RTX 4070, 27 min) overgenomen. Fine-tune voor de bal leverde niets op (beste epoch 1/16, te hoge lr). Wel winst door inferentie op 1280 + baldrempel 0,2: GT-ballen 16 → 19 van 21, valse ballen gelijk (5/7), frames zonder bal 192 → 161. Nieuwe standaard ingesteld; 97 tests slagen. Details: `kobe/match_analyse/WIJZIGINGEN.md`, sectie "Resultaten GPU-run".
