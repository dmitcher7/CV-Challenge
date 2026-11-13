
## Conclusie

(Montages van de voorbeeldplaatjes: `montage_M3a.png`, `montage_M3d.png`, `montage_M2a.png`, `montage_M4a.png`; alle crops zijn bekeken.)

1. **Het werkt, maar vooral als reparatie van de tracker, niet als echte voorspelling.** De grootste winst komt van
   **M5 (gate-fix)**: gat 204-218 was geen gemiste bal maar een weggegooide track (bal 1,4 m/frame na een trap, poort 2 m + 1 m per
   gemist frame). Met `growth=1.5`, `max_missed=30` en herstart op een zekere kandidaat (>= 0,4) die door een volgende kandidaat binnen 3
   frames bevestigd wordt, gaat `detected` van 302 naar 332, `none` van 239 naar 196 (47 frames erbij, inclusief interpolatie), GT-frame 210
   wordt gevonden op 7 px, zonder extra valse ballen op GT-frames. Kost: niets (0 s).
2. **Extrapolatie en lange interpolatie (M1) vullen veel frames maar worden snel onbetrouwbaar.** Interpolatie <= 60 frames: mediaan 9 px bij
   20 frames gat, 17 px bij 40 (maar slechts 60-73 % binnen 25 px). Extrapolatie vooruit: bij 5 frames mediaan 4,7 px, bij 10 frames 8 px
   (81 % binnen 25 px), bij 20 frames 17 px (61 %), bij 40 frames 34-38 px (39 %). Op echte gaten komen er bij M1d 172 frames bij maar
   op de GT-frames loopt het aantal valse ballen op van 4 naar 7 (bal in het net / bij de keeper, frames 360 en 630) en 3 "fout" op de
   zichtbare frames. Alleen interpolatie tot ~30 frames en extrapolatie tot ~10 frames is verdedigbaar.
3. **M2 (zwakke detecties binnen een poort)**: +42 (strak) tot +77 (ruim) frames, nauwkeurig (mediaan 2-3 px in de simulatie; in de simulatie
   zijn de echte ballen wel verborgen, dus dit meet vooral dat er weinig valse treffers zijn). Op de plaatjes: 212 en 217 echt (zelfde gat
   als M5), 251/260/263/273 waarschijnlijk de bal bij de voeten, 269 en 437 twijfelachtig. Het is een bescheiden extra, valse treffers zijn niet uit te sluiten.
4. **M3 (YOLO op een crop)**: belangrijkste bevinding is dat het model **schaalgevoelig** is. Opgeschaalde crops (x2/x3, wat je intuitief zou doen
   om een kleine bal te vergroten) vinden de bal **niet**: 0/12 treffers bij 320 px x3, terwijl de bal in het volledige frame wel gevonden wordt.
   De bal moet op dezelfde schaal blijven als bij de training (halve resolutie): crop 640 px met `imgsz=320` (z=0,5) vond 12/12, 0,26-0,4 s per crop op CPU
   (opgeschaald x3: 1,4 s). Op de echte gaten voegt M3a 36 crop-treffers toe (M3b 42, tau 0,30: 22); GT-frames 210 (5 px) en 390 (1-2 px)
   worden gevonden, 600 via extrapolatie. Op de plaatjes zijn 204-217 en 387/397 de echte bal, maar M3d (met gate-fix) pakt ook
   **valse treffers** op de zijlijn (334-337, 629, conf 0,15-0,25). tau 0,15 is te laag; gebruik >= 0,3 of eis een bevestiging in het volgende frame.
   Crop-simulatie is optimistisch (de bal was al gevonden in het volle frame); de containment was 100 % (de voorspelde crop bevatte de bal altijd).
5. **M4 (visuele trackers) werken niet.** Template matching en TrackerMIL vullen 170-180 frames, maar op de plaatjes (f250 deels, f275, f358, f383, f585, f635, f659)
   zit de cirkel op gras, lijn of doelnet: de bal wordt niet vervolgd, er is geen goede stopcriterium voor een object van ~10 px. In de simulatie
   zien ze er goed uit (mediaan 1,6 px bij template matching) omdat het ballenpatroon hetzelfde is, maar op echte gaten geven ze veel valse ballen
   (GT 4 -> 5-6 vals, 120-140 px fout op 210/360/390). Niet aanbevolen.
6. **Wat de bovengrens is:** de visuele analyse schat 80-85 echt zichtbare ballen in de 239 'none' frames; M5a + M3 (tau 0,3) + zwakke
   detecties kunnen realistisch ~60-80 daarvan dekken. De rest (bal in het doel 571-601, keeper houdt de bal 607-630) is niet zichtbaar:
   daar zouden extrapolatie / lange interpolatie alleen valse ballen toevoegen (GT-frames 630, 360).

## Aanbeveling voor `core/`

Wel doen (klein, goedkoop, geen extra YOLO-tijd):
- `core/tracking.py`: poortgroei `BALL_GATE_GROWTH` van 1.0 naar 1.5 m/frame, `BALL_MAX_MISSED` van 15 naar 30, en in `track_ball` de herstart-regel
  uit `track_ball_fixed()` (confident >= 0,4 buiten de poort, bevestigd door een kandidaat binnen 3 frames op <= 1,5 m/frame). De code staat klaar in `run_experiments.py`.
  Winst: +30 gedetecteerde en +47 gevulde frames, GT-frame 210 erbij, geen extra valse ballen op GT-frames.
- Optioneel: `MAX_BALL_GAP` in `pipeline.py` van 15 naar 30 (interpolatie over gaten tot 1 s, ook over segmentgrenzen heen mits snelheid <= 1,2 m/frame).
  Geen extrapolatie voorbij ~10 frames.
- Optioneel (beste tweede stap): M3 alleen op frames zonder bal, crop 640 px, `imgsz=320`, tau >= 0,3: ~0,3 s per frame, voor ~240 frames ~1-2 min op CPU.
  Voeg nooit upscaling toe; dat breekt de detectie.

Niet doen: visuele trackers (M4), langere extrapolatie, zwakke detecties met ruime poort zonder extra bevestiging.
