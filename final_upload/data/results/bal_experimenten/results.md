# Resultaten: bal voorspellen uit eerdere frames

Automatisch gegenereerd door `run_experiments.py`. Conclusie onderaan komt uit `conclusie.md`.

## E1: gesimuleerde gaten

Waarheid = frames waar de pipeline de bal met conf >= 0,3 detecteert (195 frames). In een venster van k frames worden alle balkandidaten met conf >= 0,1 verborgen (zwakkere blijven zichtbaar, zoals in een echt gat). Recall = aandeel waarheidsframes in het venster waarvoor de methode een bal geeft; fout = pixelafstand tot de verborgen detectie; goed = recall met fout <= 25 px. `w10/w25` = aandeel van de gevonden frames binnen 10/25 px.

### k = 5 frames (40 geschikte vensters)

| Methode | vensters | frames | recall | goed | mediaan px | p90 px | w10 | w25 |
|---|---|---|---|---|---|---|---|---|
| M0 basis | 40 | 166 | 90% | 90% | 1.8 | 7.6 | 94% | 99% |
| M1a interp<=30 | 40 | 166 | 96% | 95% | 2.0 | 7.5 | 94% | 99% |
| M1b interp<=60 | 40 | 166 | 98% | 96% | 2.0 | 8.7 | 93% | 98% |
| M1c interp60+extrap15 (v const) | 40 | 166 | 100% | 98% | 2.0 | 9.5 | 91% | 98% |
| M1d interp60+extrap30 (decay .9) | 40 | 166 | 100% | 98% | 2.0 | 9.5 | 91% | 98% |
| M1e alleen extrap vooruit 40 (v const) | 40 | 166 | 100% | 94% | 4.7 | 17.3 | 74% | 94% |
| M1f alleen extrap vooruit 40 (decay .9) | 40 | 166 | 100% | 96% | 4.5 | 17.0 | 75% | 96% |
| M5a gate-fix (groei 1.5, 30 gemist, restart) | 40 | 166 | 96% | 95% | 2.0 | 7.5 | 94% | 99% |
| M5b gate-fix + interp60 + extrap15 | 40 | 166 | 100% | 98% | 2.0 | 9.5 | 91% | 98% |
| M2a zwak 0.05 strak gate (1+0.3d m) | 40 | 166 | 96% | 93% | 1.9 | 7.6 | 94% | 97% |
| M2b zwak 0.05 ruim gate (2+0.5d m) | 40 | 166 | 96% | 93% | 1.9 | 7.6 | 94% | 97% |
| M2c zwak 0.05, size<=60, strak | 40 | 166 | 96% | 93% | 1.9 | 7.6 | 94% | 97% |
| M4a template matching | 25 | 108 | 98% | 94% | 1.6 | 4.5 | 92% | 95% |
| M4b OpenCV TrackerMIL | 25 | 108 | 100% | 98% | 2.1 | 20.4 | 89% | 98% |

### k = 10 frames (40 geschikte vensters)

| Methode | vensters | frames | recall | goed | mediaan px | p90 px | w10 | w25 |
|---|---|---|---|---|---|---|---|---|
| M0 basis | 40 | 294 | 85% | 84% | 3.0 | 12.3 | 87% | 100% |
| M1a interp<=30 | 40 | 294 | 96% | 95% | 3.0 | 11.3 | 88% | 100% |
| M1b interp<=60 | 40 | 294 | 100% | 96% | 3.2 | 15.1 | 84% | 96% |
| M1c interp60+extrap15 (v const) | 40 | 294 | 100% | 96% | 3.2 | 15.1 | 84% | 96% |
| M1d interp60+extrap30 (decay .9) | 40 | 294 | 100% | 96% | 3.2 | 15.1 | 84% | 96% |
| M1e alleen extrap vooruit 40 (v const) | 40 | 294 | 100% | 81% | 8.3 | 44.7 | 56% | 81% |
| M1f alleen extrap vooruit 40 (decay .9) | 40 | 294 | 100% | 80% | 8.4 | 36.6 | 56% | 80% |
| M5a gate-fix (groei 1.5, 30 gemist, restart) | 40 | 294 | 94% | 94% | 3.0 | 11.7 | 88% | 100% |
| M5b gate-fix + interp60 + extrap15 | 40 | 294 | 100% | 96% | 3.2 | 15.1 | 84% | 96% |
| M2a zwak 0.05 strak gate (1+0.3d m) | 40 | 294 | 96% | 94% | 2.7 | 10.8 | 90% | 99% |
| M2b zwak 0.05 ruim gate (2+0.5d m) | 40 | 294 | 96% | 94% | 2.7 | 9.3 | 91% | 99% |
| M2c zwak 0.05, size<=60, strak | 40 | 294 | 96% | 94% | 2.7 | 10.8 | 90% | 99% |
| M4a template matching | 25 | 181 | 100% | 91% | 1.6 | 23.7 | 87% | 91% |
| M4b OpenCV TrackerMIL | 25 | 181 | 100% | 90% | 2.9 | 25.7 | 86% | 90% |
| M3a crop 640 x0.5 (native schaal) | 6 | 45 | 100% | 100% | 0.6 | 1.2 | 100% | 100% |
| M3b crop 960 x0.5 | 6 | 45 | 100% | 100% | 0.6 | 1.2 | 100% | 100% |
| M3x crop 320 x3 (opgeschaald) | 6 | 45 | 0% | 0% | - | - | 0% | 0% |

### k = 20 frames (40 geschikte vensters)

| Methode | vensters | frames | recall | goed | mediaan px | p90 px | w10 | w25 |
|---|---|---|---|---|---|---|---|---|
| M0 basis | 40 | 515 | 0% | 0% | - | - | 0% | 0% |
| M1a interp<=30 | 40 | 515 | 69% | 59% | 6.1 | 29.1 | 63% | 85% |
| M1b interp<=60 | 40 | 515 | 100% | 73% | 8.9 | 49.4 | 52% | 73% |
| M1c interp60+extrap15 (v const) | 40 | 515 | 100% | 73% | 8.9 | 49.4 | 52% | 73% |
| M1d interp60+extrap30 (decay .9) | 40 | 515 | 100% | 73% | 8.9 | 49.4 | 52% | 73% |
| M1e alleen extrap vooruit 40 (v const) | 40 | 515 | 100% | 61% | 17.3 | 95.9 | 34% | 61% |
| M1f alleen extrap vooruit 40 (decay .9) | 40 | 515 | 100% | 60% | 19.3 | 68.7 | 34% | 60% |
| M5a gate-fix (groei 1.5, 30 gemist, restart) | 40 | 515 | 0% | 0% | - | - | 0% | 0% |
| M5b gate-fix + interp60 + extrap15 | 40 | 515 | 100% | 78% | 7.5 | 43.0 | 57% | 78% |
| M2a zwak 0.05 strak gate (1+0.3d m) | 40 | 515 | 63% | 61% | 2.6 | 12.6 | 88% | 97% |
| M2b zwak 0.05 ruim gate (2+0.5d m) | 40 | 515 | 64% | 62% | 2.6 | 12.6 | 88% | 97% |
| M2c zwak 0.05, size<=60, strak | 40 | 515 | 63% | 61% | 2.6 | 12.6 | 88% | 97% |
| M4a template matching | 25 | 326 | 94% | 81% | 1.6 | 35.8 | 83% | 85% |
| M4b OpenCV TrackerMIL | 25 | 326 | 96% | 73% | 4.1 | 53.5 | 68% | 75% |
| M3a crop 640 x0.5 (native schaal) | 6 | 84 | 99% | 99% | 0.6 | 1.3 | 100% | 100% |
| M3b crop 960 x0.5 | 6 | 84 | 100% | 100% | 0.7 | 1.2 | 100% | 100% |
| M3x crop 320 x3 (opgeschaald) | 6 | 84 | 0% | 0% | - | - | 0% | 0% |

### k = 40 frames (40 geschikte vensters)

| Methode | vensters | frames | recall | goed | mediaan px | p90 px | w10 | w25 |
|---|---|---|---|---|---|---|---|---|
| M0 basis | 40 | 880 | 0% | 0% | - | - | 0% | 0% |
| M1a interp<=30 | 40 | 880 | 0% | 0% | - | - | 0% | 0% |
| M1b interp<=60 | 40 | 880 | 95% | 60% | 17.0 | 87.2 | 25% | 63% |
| M1c interp60+extrap15 (v const) | 40 | 880 | 96% | 60% | 17.1 | 87.6 | 25% | 63% |
| M1d interp60+extrap30 (decay .9) | 40 | 880 | 98% | 60% | 17.5 | 97.5 | 25% | 61% |
| M1e alleen extrap vooruit 40 (v const) | 40 | 880 | 100% | 39% | 38.3 | 203.8 | 21% | 39% |
| M1f alleen extrap vooruit 40 (decay .9) | 40 | 880 | 100% | 39% | 33.6 | 174.9 | 21% | 39% |
| M5a gate-fix (groei 1.5, 30 gemist, restart) | 40 | 880 | 0% | 0% | - | - | 0% | 0% |
| M5b gate-fix + interp60 + extrap15 | 40 | 880 | 96% | 60% | 17.1 | 87.6 | 25% | 63% |
| M2a zwak 0.05 strak gate (1+0.3d m) | 40 | 880 | 62% | 60% | 2.3 | 7.4 | 91% | 96% |
| M2b zwak 0.05 ruim gate (2+0.5d m) | 40 | 880 | 68% | 65% | 2.6 | 7.3 | 92% | 96% |
| M2c zwak 0.05, size<=60, strak | 40 | 880 | 62% | 60% | 2.3 | 7.4 | 91% | 96% |
| M4a template matching | 25 | 554 | 91% | 81% | 1.7 | 38.1 | 88% | 88% |
| M4b OpenCV TrackerMIL | 25 | 554 | 99% | 52% | 22.7 | 106.3 | 46% | 52% |
| M3a crop 640 x0.5 (native schaal) | 5 | 107 | 93% | 93% | 0.9 | 1.8 | 100% | 100% |
| M3b crop 960 x0.5 | 5 | 107 | 98% | 98% | 0.8 | 1.5 | 100% | 100% |
| M3x crop 320 x3 (opgeschaald) | 5 | 107 | 0% | 0% | - | - | 0% | 0% |

Opmerking M3 (crop): in de simulatie ziet het model in de crop het echte frame, waarin de bal wel gevonden was. M3-recall is dus optimistisch; `contain` (aandeel frames waar de echte bal binnen de voorspelde crop lag) meet de voorspelling zelf: M3a crop 640 x0.5 (native schaal) k=10: 100%; M3a crop 640 x0.5 (native schaal) k=20: 100%; M3a crop 640 x0.5 (native schaal) k=40: 100%; M3b crop 960 x0.5 k=10: 100%; M3b crop 960 x0.5 k=20: 100%; M3b crop 960 x0.5 k=40: 100%; M3x crop 320 x3 (opgeschaald) k=10: 100%; M3x crop 320 x3 (opgeschaald) k=20: 100%; M3x crop 320 x3 (opgeschaald) k=40: 85%

## E2: echte gaten (660 frames)

| Methode | bal totaal | bij 'none' erbij | detected | interp | extrap | weak | crop | tracker | none | tijd (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 basis | 421 | 0 | 302 | 119 | 0 | 0 | 0 | 0 | 239 | 0.0 |
| M1a interp<=30 | 417 | 0 | 302 | 115 | 0 | 0 | 0 | 0 | 243 | 0.0 |
| M1b interp<=60 | 484 | 67 | 302 | 182 | 0 | 0 | 0 | 0 | 176 | 0.0 |
| M1c interp60+extrap15 | 548 | 127 | 302 | 182 | 64 | 0 | 0 | 0 | 112 | 0.0 |
| M1d interp60+extrap30 decay | 593 | 172 | 302 | 182 | 109 | 0 | 0 | 0 | 67 | 0.0 |
| M5a gate-fix | 464 | 47 | 332 | 132 | 0 | 0 | 0 | 0 | 196 | 0.0 |
| M5b gate-fix + interp60 + extrap15 | 591 | 170 | 332 | 210 | 49 | 0 | 0 | 0 | 69 | 0.0 |
| M2a zwak strak | 461 | 42 | 302 | 116 | 0 | 43 | 0 | 0 | 199 | 0.0 |
| M2b zwak ruim | 495 | 77 | 302 | 125 | 0 | 68 | 0 | 0 | 165 | 0.0 |
| M2c zwak size<=60 strak | 461 | 42 | 302 | 116 | 0 | 43 | 0 | 0 | 199 | 0.0 |
| M2d zwak strak + M1c vulling | 550 | 129 | 302 | 147 | 58 | 43 | 0 | 0 | 110 | 0.0 |
| M4a template matching | 589 | 168 | 302 | 14 | 0 | 0 | 0 | 273 | 71 | 2.0 |
| M4b OpenCV TrackerMIL | 602 | 182 | 302 | 9 | 0 | 0 | 0 | 291 | 58 | 19.8 |
| M3a crop 640 x0.5 tau .15 | 592 | 171 | 302 | 138 | 116 | 0 | 36 | 0 | 68 | 55.0 |
| M3b crop 960 x0.5 tau .15 | 587 | 166 | 302 | 127 | 116 | 0 | 42 | 0 | 73 | 112.2 |
| M3x crop 320 x3 tau .15 (opgeschaald) | 541 | 120 | 302 | 115 | 124 | 0 | 0 | 0 | 119 | 290.7 |
| M3d crop 640 x0.5 tau .15 + gate-fix | 630 | 209 | 332 | 162 | 116 | 0 | 20 | 0 | 30 | 17.9 |
| M3c crop 640 x0.5 tau .30 | 576 | 155 | 302 | 113 | 139 | 0 | 22 | 0 | 84 | 11.4 |
| M6 M5 + M2a + M3a(640) + M1c vulling | 655 | 234 | 332 | 242 | 25 | 44 | 12 | 0 | 5 | 17.7 |

### E2: nauwkeurigheid op de ground-truthframes (beide bestanden, 28 labels)

ok = bal gegeven binnen 25 px van het label; fout = bal gegeven maar verder; gemist = geen bal; vals = bal gegeven terwijl `ball_visible=false`.

| Methode | zichtbaar | ok | fout | gemist | niet zichtbaar | vals | mediaan px | mediaan m |
|---|---|---|---|---|---|---|---|---|
| M0 basis | 21 | 14 | 1 | 6 | 7 | 4 | 2.7 | 0.18 |
| M1a interp<=30 | 21 | 14 | 0 | 7 | 7 | 4 | 2.6 | 0.18 |
| M1b interp<=60 | 21 | 16 | 0 | 5 | 7 | 5 | 3.0 | 0.21 |
| M1c interp60+extrap15 | 21 | 17 | 1 | 3 | 7 | 5 | 3.0 | 0.21 |
| M1d interp60+extrap30 decay | 21 | 17 | 3 | 1 | 7 | 7 | 3.4 | 0.26 |
| M5a gate-fix | 21 | 15 | 0 | 6 | 7 | 5 | 2.7 | 0.18 |
| M5b gate-fix + interp60 + extrap15 | 21 | 18 | 1 | 2 | 7 | 7 | 3.3 | 0.23 |
| M2a zwak strak | 21 | 14 | 0 | 7 | 7 | 5 | 2.6 | 0.17 |
| M2b zwak ruim | 21 | 15 | 0 | 6 | 7 | 5 | 2.7 | 0.17 |
| M2c zwak size<=60 strak | 21 | 14 | 0 | 7 | 7 | 5 | 2.6 | 0.17 |
| M2d zwak strak + M1c vulling | 21 | 17 | 1 | 3 | 7 | 5 | 3.4 | 0.21 |
| M4a template matching | 21 | 15 | 2 | 4 | 7 | 5 | 2.8 | 0.18 |
| M4b OpenCV TrackerMIL | 21 | 16 | 4 | 1 | 7 | 6 | 3.5 | 0.26 |
| M3a crop 640 x0.5 tau .15 | 21 | 19 | 0 | 2 | 7 | 5 | 2.7 | 0.22 |
| M3b crop 960 x0.5 tau .15 | 21 | 19 | 0 | 2 | 7 | 5 | 2.7 | 0.20 |
| M3x crop 320 x3 tau .15 (opgeschaald) | 21 | 17 | 1 | 3 | 7 | 5 | 3.0 | 0.21 |
| M3d crop 640 x0.5 tau .15 + gate-fix | 21 | 19 | 1 | 1 | 7 | 7 | 3.2 | 0.26 |
| M3c crop 640 x0.5 tau .30 | 21 | 19 | 0 | 2 | 7 | 5 | 2.7 | 0.23 |
| M6 M5 + M2a + M3a(640) + M1c vulling | 21 | 18 | 3 | 0 | 7 | 7 | 3.5 | 0.29 |

Per GT-frame waar een methode afwijkt van de basis:

- M0 basis: feet:240 FALSE (dete), feet:600 -, main:210 -, main:240 FALSE (dete), main:30 FALSE (inte), main:300 FALSE (inte), main:360 -, main:390 -, main:420 -, main:600 -
- M1a interp<=30: feet:240 FALSE (dete), feet:600 -, main:210 -, main:240 FALSE (dete), main:30 FALSE (inte), main:300 FALSE (inte), main:360 -, main:390 -, main:420 -, main:600 -
- M1b interp<=60: feet:240 FALSE (dete), feet:600 12px (inte), main:210 -, main:240 FALSE (dete), main:270 FALSE (inte), main:30 FALSE (inte), main:300 FALSE (inte), main:360 -, main:390 -, main:420 -, main:600 9px (inte)
- M1c interp60+extrap15: feet:240 FALSE (dete), feet:600 12px (inte), main:210 105px (extr), main:240 FALSE (dete), main:270 FALSE (inte), main:30 FALSE (inte), main:300 FALSE (inte), main:360 -, main:390 -, main:420 -, main:600 9px (inte)
- M1d interp60+extrap30 decay: feet:240 FALSE (dete), feet:360 FALSE (extr), feet:600 12px (inte), main:210 111px (extr), main:240 FALSE (dete), main:270 FALSE (inte), main:30 FALSE (inte), main:300 FALSE (inte), main:360 118px (extr), main:390 -, main:420 26px (extr), main:600 9px (inte), main:630 FALSE (extr)
- M5a gate-fix: feet:240 FALSE (dete), feet:600 -, main:210 7px (dete), main:240 FALSE (dete), main:270 FALSE (inte), main:30 FALSE (inte), main:300 FALSE (inte), main:360 -, main:390 -, main:420 -, main:600 -
- M5b gate-fix + interp60 + extrap15: feet:240 FALSE (dete), feet:360 FALSE (extr), feet:600 12px (inte), main:210 7px (dete), main:240 FALSE (dete), main:270 FALSE (inte), main:30 FALSE (inte), main:300 FALSE (inte), main:360 114px (extr), main:390 -, main:420 -, main:600 9px (inte), main:630 FALSE (inte)
- M2a zwak strak: feet:240 FALSE (dete), feet:600 -, main:210 -, main:240 FALSE (dete), main:270 FALSE (inte), main:30 FALSE (weak), main:300 FALSE (inte), main:360 -, main:390 -, main:420 -, main:600 -
- M2b zwak ruim: feet:240 FALSE (dete), feet:600 -, main:210 7px (weak), main:240 FALSE (dete), main:270 FALSE (inte), main:30 FALSE (weak), main:300 FALSE (inte), main:360 -, main:390 -, main:420 -, main:600 -
- M2c zwak size<=60 strak: feet:240 FALSE (dete), feet:600 -, main:210 -, main:240 FALSE (dete), main:270 FALSE (inte), main:30 FALSE (weak), main:300 FALSE (inte), main:360 -, main:390 -, main:420 -, main:600 -
- M2d zwak strak + M1c vulling: feet:240 FALSE (dete), feet:600 12px (inte), main:210 7px (extr), main:240 FALSE (dete), main:270 FALSE (inte), main:30 FALSE (weak), main:300 FALSE (inte), main:360 -, main:390 -, main:420 -, main:600 9px (inte)
- M4a template matching: feet:240 FALSE (dete), feet:600 -, main:210 121px (trac), main:240 FALSE (dete), main:30 FALSE (inte), main:300 FALSE (trac), main:360 -, main:390 136px (trac), main:420 -, main:600 -, main:630 FALSE (trac)
- M4b OpenCV TrackerMIL: feet:240 FALSE (dete), feet:360 FALSE (trac), feet:600 48px (trac), main:210 9px (trac), main:240 FALSE (dete), main:30 FALSE (trac), main:300 FALSE (trac), main:360 139px (trac), main:390 133px (trac), main:420 -, main:600 50px (trac), main:630 FALSE (trac)
- M3a crop 640 x0.5 tau .15: feet:240 FALSE (dete), feet:600 7px (extr), main:210 5px (crop), main:240 FALSE (dete), main:270 FALSE (inte), main:30 FALSE (inte), main:300 FALSE (extr), main:360 -, main:390 1px (crop), main:420 -, main:600 4px (extr)
- M3b crop 960 x0.5 tau .15: feet:240 FALSE (dete), feet:600 7px (extr), main:210 5px (crop), main:240 FALSE (dete), main:270 FALSE (inte), main:30 FALSE (inte), main:300 FALSE (extr), main:360 -, main:390 2px (crop), main:420 -, main:600 4px (extr)
- M3x crop 320 x3 tau .15 (opgeschaald): feet:240 FALSE (dete), feet:600 7px (extr), main:210 111px (extr), main:240 FALSE (dete), main:270 FALSE (extr), main:30 FALSE (inte), main:300 FALSE (inte), main:360 -, main:390 -, main:420 -, main:600 4px (extr)
- M3d crop 640 x0.5 tau .15 + gate-fix: feet:240 FALSE (dete), feet:360 FALSE (extr), feet:600 7px (extr), main:210 7px (dete), main:240 FALSE (dete), main:270 FALSE (inte), main:30 FALSE (inte), main:300 FALSE (extr), main:360 121px (extr), main:390 3px (inte), main:420 -, main:600 4px (extr), main:630 FALSE (inte)
- M3c crop 640 x0.5 tau .30: feet:240 FALSE (dete), feet:600 7px (extr), main:210 5px (crop), main:240 FALSE (dete), main:270 FALSE (extr), main:30 FALSE (inte), main:300 FALSE (inte), main:360 -, main:390 2px (extr), main:420 -, main:600 4px (extr)
- M6 M5 + M2a + M3a(640) + M1c vulling: feet:240 FALSE (dete), feet:360 FALSE (inte), feet:600 12px (inte), main:210 7px (dete), main:240 FALSE (dete), main:270 FALSE (inte), main:30 FALSE (weak), main:300 FALSE (extr), main:360 127px (inte), main:390 3px (inte), main:420 41px (inte), main:600 9px (inte), main:630 FALSE (inte)

## Rekentijd

- Crop-herdetectie (YOLO, imgsz 960 op CPU): 0.40 s per crop (2513 crops deze run; bij een cachehit 0 s).
- Totale looptijd van dit script: 1200 s.

## Voorbeeldplaatjes (map `samples/`)

- M2a zwak strak: `samples/M2a_f212.png`, `samples/M2a_f217.png`, `samples/M2a_f251.png`, `samples/M2a_f260.png`, `samples/M2a_f263.png`, `samples/M2a_f269.png`, `samples/M2a_f273.png`, `samples/M2a_f437.png`
- M4a template matching: `samples/M4a_f204.png`, `samples/M4a_f250.png`, `samples/M4a_f275.png`, `samples/M4a_f358.png`, `samples/M4a_f383.png`, `samples/M4a_f585.png`, `samples/M4a_f635.png`, `samples/M4a_f659.png`
- M3a crop 640 x0.5 tau .15: `samples/M3a_f204.png`, `samples/M3a_f207.png`, `samples/M3a_f211.png`, `samples/M3a_f214.png`, `samples/M3a_f217.png`, `samples/M3a_f261.png`, `samples/M3a_f387.png`, `samples/M3a_f397.png`
- M3b crop 960 x0.5 tau .15: `samples/M3b_f204.png`, `samples/M3b_f207.png`, `samples/M3b_f211.png`, `samples/M3b_f214.png`, `samples/M3b_f218.png`, `samples/M3b_f262.png`, `samples/M3b_f388.png`, `samples/M3b_f392.png`
- M3d crop 640 x0.5 tau .15 + gate-fix: `samples/M3d_f334.png`, `samples/M3d_f335.png`, `samples/M3d_f336.png`, `samples/M3d_f337.png`, `samples/M3d_f388.png`, `samples/M3d_f392.png`, `samples/M3d_f397.png`, `samples/M3d_f629.png`
- M6 M5 + M2a + M3a(640) + M1c vulling: `samples/M6_f250.png`, `samples/M6_f264.png`, `samples/M6_f335.png`, `samples/M6_f345.png`, `samples/M6_f356.png`, `samples/M6_f392.png`, `samples/M6_f628.png`, `samples/M6_f639.png`

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
