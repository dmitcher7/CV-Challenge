"""How good is the homography? Two checks, and where an extra keyframe would help.

    python check_homography.py

1. Clicks: per keyframe the reprojection error in METRES (RMSE of the clicked points mapped to the
   pitch with the fitted H) and a spread check (are the points spread out, not on one line?).
   A keyframe with 4 points always has error 0: it cannot be checked, click a 5th point.
2. Drift: for every frame the pitch lines are projected into the image (H interpolated between
   keyframes) and we measure which fraction of the projected line points lies on a white pixel
   (within TOLERANCE px). On keyframes this is the best we can get (players, shadows and worn lines hide
   part of the lines); between keyframes a clearly lower score means the interpolated H has drifted.

Writes output/homography_check.csv (drift score per frame) and output/homography_reprojection.csv (per keyframe).
"""

import argparse
import csv

import cv2
import numpy as np

from core.homography import ERROR_WARN_M, Homographies, load_keypoints, reprojection_report, spread_check
from core.paths import OUTPUT_DIR, VIDEO
from core.pitch import LINES
from core.video import iter_frames, video_info

TOLERANCE = 4           # px: a projected line point within this distance of a white pixel counts as "on the line"
RELATIVE_LIMIT = 0.75   # a frame scoring below 75% of the median keyframe score has drifted
MIN_SPACING = 8         # frames between suggested keyframes


def white_mask(frame):
    """White pixels (painted lines): bright and grey-ish, dilated by TOLERANCE."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 2] > 160) & (hsv[..., 1] < 70)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TOLERANCE + 1, 2 * TOLERANCE + 1))
    return cv2.dilate(white, kernel)


def check_clicks():
    """Print and save the reprojection error (metres) and the spread check per keyframe."""
    keypoints = load_keypoints()
    rows = reprojection_report(keypoints)
    print("Keyframe clicks (error in metres on the pitch; 4-point keyframes cannot be checked)")
    print(f"{'frame':>6} {'points':>6} {'rmse_m':>8} {'max_m':>7}  {'worst point':<22} {'hull%':>6}  notes")
    out = []
    for r in rows:
        spread = spread_check(keypoints[r["frame"]])
        notes = list(spread["warnings"])
        if not r["checkable"]:
            notes.insert(0, "4 points: error is always 0, not checkable")
        elif r["rmse_m"] > ERROR_WARN_M:
            notes.insert(0, f"RMSE > {ERROR_WARN_M:.1f} m: re-check the clicks")
        rmse = f"{r['rmse_m']:.2f}" if r["checkable"] else "n/a"
        print(f"{r['frame']:>6} {r['n_points']:>6} {rmse:>8} {r['max_m'] if r['checkable'] else float('nan'):>7.2f}  "
              f"{r['worst'] if r['checkable'] else '-':<22} {100 * spread['hull_frac']:>5.1f}%  {'; '.join(notes)}")
        out.append({**r, "hull_frac": round(spread["hull_frac"], 4), "n_collinear": spread["n_collinear"],
                    "good_quad": int(spread["good_quad"]), "notes": "; ".join(notes)})
    errors = [r["rmse_m"] for r in rows if r["checkable"]]
    if errors:
        print(f"\nRMSE over {len(errors)} checkable keyframes (of {len(rows)}): min {min(errors):.2f} m, "
              f"median {np.median(errors):.2f} m, max {max(errors):.2f} m")
    else:
        print("\nNo keyframe has 5 or more points: nothing can be checked yet.")
    with open(OUTPUT_DIR / "homography_reprojection.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "n_points", "checkable", "rmse_m", "max_m", "worst",
                                               "hull_frac", "n_collinear", "good_quad", "notes"])
        writer.writeheader()
        writer.writerows({**r, "rmse_m": round(r["rmse_m"], 3), "max_m": round(r["max_m"], 3)} for r in out)


def check_drift(video, step):
    """White-pixel score per frame (every step-th frame); prints where extra keyframes would help."""
    hs = Homographies()
    ids = list(range(0, video_info(video)["frames"], step))
    # Dense points along every pitch line (in metres), projected per frame
    dense = np.concatenate([np.linspace(a, b, 40) for line in LINES for a, b in zip(line[:-1], line[1:])])
    scores = {}
    for idx, frame in iter_frames(video, ids):
        h, w = frame.shape[:2]
        pts = hs.pitch_to_image(dense, idx)
        # Only points inside the image and below the top 15% (stands and boards are not pitch)
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0.15 * h) & (pts[:, 1] < h)
        pts = pts[inside].astype(int)
        scores[idx] = float(white_mask(frame)[pts[:, 1], pts[:, 0]].mean()) if len(pts) > 50 else float("nan")

    keyframes = sorted(hs.keyframes)
    reference = np.nanmedian([scores[i] for i in keyframes if i in scores])
    limit = RELATIVE_LIMIT * reference
    # Suggest the worst frame of every stretch below the limit, at least MIN_SPACING apart
    bad = sorted((s, i) for i, s in scores.items() if s < limit and i not in hs.keyframes)
    suggested = []
    for _, i in bad:
        if all(abs(i - j) >= MIN_SPACING for j in suggested + keyframes):
            suggested.append(i)
    suggested.sort()

    with open(OUTPUT_DIR / "homography_check.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "score", "keyframe"])
        for i, s in sorted(scores.items()):
            writer.writerow([i, round(s, 3), int(i in hs.keyframes)])

    print(f"\nDrift check on {len(scores)} frames: keyframes median score {reference:.2f}; all frames median "
          f"{np.nanmedian(list(scores.values())):.2f}, worst {np.nanmin(list(scores.values())):.2f}")
    print(f"{len(bad)} frames below the limit {limit:.2f}")
    weak = [i for i in keyframes if scores.get(i, 1) < limit]
    if weak:  # a low score ON a keyframe: few visible lines, or imprecise clicks
        print("Keyframes that score low themselves (re-check their clicks):")
        print("  python click_keypoints.py --frames " + " ".join(map(str, weak)))
    if suggested:
        print("Suggested extra keyframes (click them, existing points are kept):")
        print("  python click_keypoints.py --frames " + " ".join(map(str, suggested)))
    else:
        print("No extra keyframes needed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=str(VIDEO))
    parser.add_argument("--step", type=int, default=1, help="drift check on every n-th frame")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    check_clicks()
    check_drift(args.video, args.step)
    print(f"\nWrote {OUTPUT_DIR / 'homography_check.csv'} and {OUTPUT_DIR / 'homography_reprojection.csv'}")


if __name__ == "__main__":
    main()
