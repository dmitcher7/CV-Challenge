"""Export frames that best.pt has never seen, ready to upload to Roboflow for box labelling (a fair test set).

    python kobe\\match_analyse\\gpu\\export_unseen_frames.py                    (frames 63-199 and 246-524, every 5th)
    python kobe\\match_analyse\\gpu\\export_unseen_frames.py --every 10 --ranges 63-199,334-437

The 100 training frames of best.pt are 0-62, 200-245 and 525-658; everything in between is unseen.
Writes gpu/unseen_frames/frame_0065.jpg, ... (the 'frame_<n>' name lets the leakage check recognise the frame).
How to turn the labelled export into --test-data: see gpu/README.md, section "Eerlijke testset".
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpu.common import GPU_DIR, VIDEO  # noqa: E402
from core.video import iter_frames  # noqa: E402

DEFAULT_RANGES = "63-199,246-524"


def parse_ranges(text):
    """'63-199,246-524' -> [(63, 199), (246, 524)]."""
    out = []
    for part in text.split(","):
        a, b = part.strip().split("-")
        a, b = int(a), int(b)
        if b < a:
            raise ValueError(f"bereik {part}: einde voor begin")
        out.append((a, b))
    return out


def frames_to_export(ranges, every):
    ids = []
    for a, b in ranges:
        ids += range(a, b + 1, every)
    return sorted(set(ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=str(VIDEO))
    parser.add_argument("--ranges", default=DEFAULT_RANGES, help="framebereiken, bv. 63-199,246-524")
    parser.add_argument("--every", type=int, default=5, help="elk n-de frame (opeenvolgende frames lijken sterk op elkaar)")
    parser.add_argument("--out", default=str(GPU_DIR / "unseen_frames"))
    args = parser.parse_args()
    if not Path(args.video).exists():
        sys.exit(f"video niet gevonden: {args.video}")
    ids = frames_to_export(parse_ranges(args.ranges), args.every)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for idx, frame in iter_frames(args.video, ids):
        cv2.imwrite(str(out / f"frame_{idx:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    print(f"{n} frames -> {out}")


if __name__ == "__main__":
    main()
