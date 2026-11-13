# Samenvatting: Nieuw model van Brent (brent_v2) in minimap-notebook

**Datum:** 2026-11-09  
**Status:** AFGEROND (losse opdracht, geen plan)  

## Overzicht
Nieuw detectiemodel van Brent (YOLO26m, 100 epochs, imgsz 1600, 5 klassen: ball, player_A, player_B, player_B_goal_keeper, referee) geïntegreerd in `kobe/notebooks/minimap/02_minimap_video.ipynb`.

## Belangrijke beslissingen & stappen
- Model geplaatst in `kobe/homography/pretrained/brent_v2/best.pt` en geëxporteerd naar OpenVINO (FP16, imgsz 1600) in `kobe/homography/pretrained/brent_v2/best_openvino_model/`.
- Afzonderlijke map `brent_v2` gekozen om overschrijving van het oude Brent-model te voorkomen (en cache-conflicten met `detections_brent_1600.json` te vermijden).
- Notebook cel 3 aangepast (`USE_MODEL = brent_v2`, keuzes `own` / `brent` / `brent_v2`); `RUN`-naam volgt modelkeuze.

## Resultaten & Testrun
- Volledige testrun succesvol: `kobe/notebooks/minimap/output/minimap_brent_v2.mp4`, `minimap_brent_v2_frames.csv`, en `detections_brent_v2_1600.json`.
- Bal gedetecteerd in 292 frames en geïnterpoleerd in 119 van de 660 frames; balbezit PSG 87% / Inter 13%.
- Openstaand punt: in controlecel (cel 17) vindt het model de bal in de test-split 0 keer op de juiste plek (train/val wel in orde); nader te onderzoeken.
- Bestanden staan buiten git (`pretrained/` en `output/` zijn genegeerd).
