# System evaluation: model `psg_vs_inter_CV`

Labelled frames: 6, compared: 6 (6 exactly on a processed frame, max offset 0 frames, 0 skipped: no processed frame within 1 frames).

Person clicks: median height in the box 0.84 (0 = head, 1 = feet).

## Players (anchor feet: match radius 3.0 m, Hungarian)

- position error (m): mean 1.091, median 1.026, p90 1.770 (n=140)
- TP 140, FP 3, FN 0: precision 0.979, recall 1.000, F1 0.989

## Teams / roles (matched players)

- role accuracy (A/B/keeper/referee): 0.993; team accuracy (keeper counts as B, referee as none): 0.993 (n=140)

| GT \ predicted | A | B | K | R | other |
|---|---|---|---|---|---|
| A | 59 | 0 | 0 | 1 | 0 |
| B | 0 | 59 | 0 | 0 | 0 |
| K | 0 | 0 | 6 | 0 | 0 |
| R | 0 | 0 | 0 | 15 | 0 |

## Ball

- frames with a ball in GT: 4; detection rate 1.000
- position error (m): mean 0.481, median 0.165, p90 1.150; within 3 m: 1.000
- frames without a ball in GT: 2; false ball rate 0.500

## Possession

- accuracy 0.667 (n=6)

| GT \ predicted | A | B | - |
|---|---|---|---|
| A | 4 | 0 | 2 |

## Tracking

- ID switches (change of matched track id of a GT person between consecutive labelled frames): 36 (75 GT persons)
- IDF1-lite 0.367 (IDTP 52, IDFP 91, IDFN 88)
- note: labelled frames are far apart, so this is a coarse estimate of the real ID switches.
