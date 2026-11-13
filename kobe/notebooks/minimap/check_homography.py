"""How well does the homography fit each frame? Suggests where to click extra keyframes.

For every frame the pitch lines are projected into the image (with H of that frame,
interpolated between keyframes) and we measure which fraction of the projected line
points lies on a white pixel in the image (within TOLERANCE px). On the clicked keyframes
this is the best we can get (players, shadows and worn lines hide part of the lines);
between keyframes a clearly lower score means the interpolated H has drifted.

    venv\\Scripts\\python kobe\\notebooks\\minimap\\check_homography.py

Writes output/homography_check.png (score per frame) and output/homography_check.csv,
and prints the frames where an extra keyframe would help, as a ready-to-run command.
"""

import csv

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # write the plot to a file, no window
import matplotlib.pyplot as plt  # noqa: E402

from homography import Homographies  # noqa: E402
from pitch import FRAMES_DIR, LINES, OUTPUT_DIR  # noqa: E402

TOLERANCE = 4           # px: a projected line point within this distance of a white pixel counts as "on the line"
RELATIVE_LIMIT = 0.75   # a frame scoring below 75% of the median keyframe score has drifted
MIN_SPACING = 8         # frames between suggested keyframes

homographies = Homographies()
frames = sorted(FRAMES_DIR.glob("*.jpg"), key=lambda p: int(p.stem.split("_idx_")[1]))
# Dense points along every pitch line (in metres), projected per frame
dense = np.concatenate([np.linspace(a, b, 40) for line in LINES for a, b in zip(line[:-1], line[1:])])


def white_mask(frame):
    """White pixels (painted lines): bright and grey-ish, dilated by TOLERANCE."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 2] > 160) & (hsv[..., 1] < 70)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TOLERANCE + 1, 2 * TOLERANCE + 1))
    return cv2.dilate(white, kernel)


scores = {}
for path in frames:
    idx = int(path.stem.split("_idx_")[1])
    frame = cv2.imread(str(path))
    h, w = frame.shape[:2]
    pts = homographies.pitch_to_image(dense, idx)
    # Only points inside the image and below the top 15% (stands and boards are not pitch)
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0.15 * h) & (pts[:, 1] < h)
    pts = pts[inside].astype(int)
    mask = white_mask(frame)
    scores[idx] = float(mask[pts[:, 1], pts[:, 0]].mean()) if len(pts) > 50 else float("nan")

keyframes = sorted(homographies.keyframes)
reference = np.nanmedian([scores[i] for i in keyframes if i in scores])
limit = RELATIVE_LIMIT * reference

# Suggest the worst frame of every stretch below the limit, at least MIN_SPACING apart
bad = sorted((s, i) for i, s in scores.items() if s < limit and i not in homographies.keyframes)
suggested = []
for _, i in bad:
    if all(abs(i - j) >= MIN_SPACING for j in suggested + keyframes):
        suggested.append(i)
suggested.sort()

with open(OUTPUT_DIR / "homography_check.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "score", "keyframe"])
    for i, s in sorted(scores.items()):
        writer.writerow([i, round(s, 3), int(i in homographies.keyframes)])

fig, ax = plt.subplots(figsize=(12, 3.5))
ids = sorted(scores)
ax.plot(ids, [scores[i] for i in ids], lw=0.8, label="line fit score")
ax.plot(keyframes, [scores[i] for i in keyframes], "o", ms=4, label="clicked keyframe")
ax.axhline(limit, color="red", ls="--", lw=0.8, label=f"limit ({RELATIVE_LIMIT:.0%} of keyframes)")
for i in suggested:
    ax.axvline(i, color="orange", lw=0.8)
ax.set(xlabel="frame", ylabel="fraction of line points on white", xlim=(0, ids[-1]),
       title="Homography check (orange = suggested extra keyframe)")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "homography_check.png", dpi=120)

print(f"keyframes: median score {reference:.2f}; all frames: median {np.nanmedian(list(scores.values())):.2f}, "
      f"worst {min(scores.values()):.2f} (frame {min(scores, key=scores.get)})")
print(f"{len(bad)} frames below the limit {limit:.2f}")
weak = [i for i in keyframes if scores.get(i, 1) < limit]
if weak:  # a low score ON a keyframe: few visible lines, or imprecise clicks (check the error in px)
    print("Keyframes that score low themselves (re-check their clicks):")
    print("  venv\\Scripts\\python kobe\\notebooks\\minimap\\click_keypoints.py --frames " + " ".join(map(str, weak)))
if suggested:
    print("Suggested extra keyframes (click them, existing points are kept):")
    print("  venv\\Scripts\\python kobe\\notebooks\\minimap\\click_keypoints.py --frames " + " ".join(map(str, suggested)))
else:
    print("No extra keyframes needed.")
print(f"Plot: {OUTPUT_DIR / 'homography_check.png'}")
