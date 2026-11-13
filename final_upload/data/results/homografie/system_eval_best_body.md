# System evaluation: model `best`

Labelled frames: 22, compared: 22 (22 exactly on a processed frame, max offset 0 frames, 0 skipped: no processed frame within 1 frames).

Person clicks: median height in the box 0.40 (0 = head, 1 = feet).

## Players (anchor body: click vs box centre in pixels, match radius 0.6 box heights, Hungarian)

- no position error in metres: body clicks are not on the pitch plane (label the feet for that)
- match distance (box heights): mean 0.130, median 0.127 (n=512)
- TP 512, FP 11, FN 4: precision 0.979, recall 0.992, F1 0.986

## Teams / roles (matched players)

- role accuracy (A/B/keeper/referee): 0.992; team accuracy (keeper counts as B, referee as none): 0.992 (n=512)

| GT \ predicted | A | B | K | R | other |
|---|---|---|---|---|---|
| A | 212 | 0 | 0 | 4 | 0 |
| B | 0 | 217 | 0 | 0 | 0 |
| K | 0 | 0 | 22 | 0 | 0 |
| R | 0 | 0 | 0 | 57 | 0 |

## Ball

- frames with a ball in GT: 17; detection rate 0.882
- position error (m): mean 0.739, median 0.262, p90 1.282; within 3 m: 0.933
- frames without a ball in GT: 5; false ball rate 0.800

## Possession

- accuracy 0.773 (n=22)

| GT \ predicted | A | B | - |
|---|---|---|---|
| A | 17 | 1 | 4 |

## Tracking

- ID switches (change of matched track id of a GT person between consecutive labelled frames): 71 (66 GT persons)
- IDF1-lite 0.581 (IDTP 302, IDFP 221, IDFN 214)
- note: labelled frames are far apart, so this is a coarse estimate of the real ID switches.
