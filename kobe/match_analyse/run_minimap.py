"""Video with the pitch lines and detections next to the 2D minimap, plus csv files with all positions.

    venv\\Scripts\\python kobe\\match_analyse\\run_minimap.py --start 0 --end 120 --stride 2

Writes (in output/, <model> = name of the weights file):
    minimap_<model>.mp4   frame + minimap side by side (skip with --no-video)
    frames_<model>.csv    one row per frame: ball position, ball state, possession
    tracks_<model>.csv    one row per tracked person per frame: track id, group, team, position in metres
The exact columns are documented at the top of core/pipeline.py.
YOLO is cached in cache/: running again with other thresholds (--conf, --ball-conf) is fast.
--ball-refine looks for the ball again in crops of the frames without a ball (core/ball_refine.py):
~0.3 s per crop on CPU the first time, then cached in cache/ as well.
"""

import argparse
import time
from collections import Counter
from pathlib import Path

from core.paths import DEFAULT_WEIGHTS, OUTPUT_DIR, VIDEO
from core.pipeline import DEFAULT_BALL_CONF, DEFAULT_IMGSZ
from core.pipeline import possession_share, run, save_frames_csv, save_tracks_csv
from core.render import label, write_video


def progress_printer(what):
    def show(done, total):
        if done == total or done % 10 == 0:
            print(f"{what} {done}/{total}", flush=True)
    return show


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=str(VIDEO))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help="1280 finds the small ball more often; 960 = training size, faster on CPU")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None, help="last frame (inclusive), default: end of the video")
    parser.add_argument("--stride", type=int, default=1, help="process every n-th frame (2 = twice as fast)")
    parser.add_argument("--conf", type=float, default=0.3, help="confidence to start a person track")
    parser.add_argument("--ball-conf", type=float, default=DEFAULT_BALL_CONF, help="confidence threshold for ball candidates")
    parser.add_argument("--no-video", action="store_true", help="only write the csv files")
    parser.add_argument("--ball-refine", action="store_true",
                        help="re-detect the ball in crops of frames without a ball (slow once, then cached)")
    args = parser.parse_args()
    make_minimap(args.video, args.weights, args.imgsz, args.start, args.end, args.stride, args.conf,
                 args.ball_conf, args.no_video, args.ball_refine)


def make_minimap(video=VIDEO, weights=DEFAULT_WEIGHTS, imgsz=DEFAULT_IMGSZ, start=0, end=None, stride=1, conf=0.3,
                 ball_conf=DEFAULT_BALL_CONF, no_video=False, ball_refine=False):
    """The whole minimap step (also used by run_all.py). Prints progress and returns the model name."""
    name = Path(weights).stem
    OUTPUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    result = run(str(video), str(weights), imgsz, start=start, end=end, stride=stride,
                 conf=conf, ball_conf=ball_conf, progress=progress_printer("detected"),
                 ball_refine=ball_refine, refine_progress=progress_printer("ball crops"))
    t1 = time.time()

    frames_csv = save_frames_csv(result, OUTPUT_DIR / f"frames_{name}.csv")
    tracks_csv = save_tracks_csv(result, OUTPUT_DIR / f"tracks_{name}.csv")
    print(f"analysis: {t1 - t0:.0f} s -> {frames_csv}\n                     {tracks_csv}")
    if not no_video:
        out = write_video(result, OUTPUT_DIR / f"minimap_{name}.mp4", progress=progress_printer("rendered"))
        print(f"video: {time.time() - t1:.0f} s -> {out}")

    n = len(result.ids)
    groups = Counter(p["group"] for ps in result.people.values() for p in ps)
    states = Counter(result.ball[i]["state"] if i in result.ball else "none" for i in result.ids)
    tracks = {p["track_id"] for ps in result.people.values() for p in ps}
    print(f"{n} frames, {len(tracks)} tracks; person-frames per group: "
          + ", ".join(f"{label(g)} {c}" for g, c in sorted(groups.items())))
    print("ball: " + ", ".join(f"{s} {c}/{n}" for s, c in sorted(states.items())))
    share = possession_share(result)
    print("possession: " + (", ".join(f"{label(t)} {p:.0f}%" for t, p in share.items()) or "none"))
    return name


if __name__ == "__main__":
    main()
