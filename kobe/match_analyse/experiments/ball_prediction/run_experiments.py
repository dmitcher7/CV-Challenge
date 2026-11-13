"""Experiment: can the ball be found in more frames by PREDICTING it from earlier frames?

Run (from anywhere):  venv\\Scripts\\python.exe kobe/match_analyse/experiments/ball_prediction/run_experiments.py
Options:  --skip-crops   skip the (slow, CPU) YOLO crop re-detection M3
          --fresh        ignore the crop cache (crop_cache.json) and run YOLO again

Nothing in core/ is changed; this file only imports from it.  Output goes to this folder:
results.md, results.json, samples/*.png, crop_cache.json.

Methods (all work in the "stabilised" space = pitch metres of the pixel, so camera pan/zoom is compensated by
the per-frame homography; a ball in the air is projected wrongly but consistently, which is fine for prediction):
  M0  baseline: core.tracking.track_ball (Kalman in metres, gap <= 15 interpolated)
  M1  longer interpolation + extrapolation (constant velocity, optional decay) from the anchors
  M2  weak detections (conf 0.05-0.1 / bigger) accepted only inside a gate around the prediction
  M3  YOLO on a crop (upscaled) around the predicted pixel position, sequentially chained
  M4  visual tracker through the gap: template matching and OpenCV TrackerMIL on camera-stabilised frames
"""

import argparse
import bisect
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                     # match_analyse
sys.path.insert(0, str(ROOT))

from core.detection import detect, load_model                       # noqa: E402
from core.homography import Homographies, project                   # noqa: E402
from core.paths import DEFAULT_WEIGHTS, VIDEO                       # noqa: E402
from core.pipeline import MAX_BALL_GAP, MAX_BALL_SIZE, SPOTS, SPOT_RADIUS, on_pitch  # noqa: E402
from core.teams import is_ball                                      # noqa: E402
from core.tracking import BALL_START_CONF, BALL_WEAK, track_ball    # noqa: E402
from core.video import video_info                                   # noqa: E402

VMAX = 1.2          # m/frame: faster than this (36 m/s) is not a football
OK_PX = 25          # a found ball within this many pixels of the truth counts as correct
SAMPLES = HERE / "samples"
CROP_CACHE = HERE / "crop_cache.json"


# --------------------------------------------------------------------------- data
def load_all():
    info = video_info(VIDEO)
    ids = list(range(info["frames"]))
    frames = detect(VIDEO, DEFAULT_WEIGHTS, 960, 0.7, 0, info["frames"] - 1, 1)   # from the cache
    hs = Homographies()
    cands = {}
    for f in frames:
        idx = f["frame"]
        for d in f["detections"]:
            if not is_ball(d):
                continue
            x1, y1, x2, y2 = d["box"]
            xy = hs.image_to_pitch([((x1 + x2) / 2, (y1 + y2) / 2)], idx)[0]
            if on_pitch(xy) and min(np.hypot(*np.subtract(xy, s)) for s in SPOTS) > SPOT_RADIUS:
                cands.setdefault(idx, []).append((xy, d["conf"], tuple(d["box"])))
    return info, ids, hs, cands


def size_of(box):
    return max(box[2] - box[0], box[3] - box[1])


def centre(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def filt(cands, min_conf, max_size):
    return {i: [c for c in cs if c[1] >= min_conf and size_of(c[2]) <= max_size] for i, cs in cands.items()}


def baseline_track(cands, ids):
    """core's ball tracker exactly as the pipeline calls it (ball_conf 0.1, size <= 40)."""
    return track_ball(filt(cands, 0.1, MAX_BALL_SIZE), ids, MAX_BALL_GAP, BALL_START_CONF, BALL_WEAK, stride=1)


class Ctx:
    def __init__(self):
        self.info, self.ids, self.hs, self.cands = load_all()
        self.reader = FrameReader(VIDEO)
        self.model = None

    def px_of(self, m, t):
        return self.hs.pitch_to_image([m], t)[0]

    def m_of(self, px, t):
        return self.hs.image_to_pitch([px], t)[0]

    def get_model(self):
        if self.model is None:
            self.model = load_model(DEFAULT_WEIGHTS)
        return self.model


class FrameReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.pos = 0
        self.cache = {}

    def get(self, t):
        if t in self.cache:
            return self.cache[t]
        if t != self.pos:
            if 0 < t - self.pos <= 40:
                for _ in range(t - self.pos):
                    self.cap.grab()
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, fr = self.cap.read()
        self.pos = t + 1
        if len(self.cache) > 8:
            self.cache.pop(next(iter(self.cache)))
        self.cache[t] = fr if ok else None
        return self.cache[t]


# --------------------------------------------------------------------------- anchors + prediction
class Anchors:
    """Known ball positions (metres) per frame, with prediction for the frames in between / around."""

    def __init__(self, d=None):
        self.m = {k: np.asarray(v, float) for k, v in (d or {}).items()}
        self.f = sorted(self.m)

    def add(self, t, m):
        if t not in self.m:
            bisect.insort(self.f, t)
        self.m[t] = np.asarray(m, float)

    def _vel(self, p, direction):
        """velocity (m/frame) at anchor p estimated from the anchors up to 8 frames before (direction=+1) / after (-1)."""
        i = self.f.index(p)
        others = [q for q in (self.f[:i] if direction > 0 else self.f[i + 1:]) if abs(p - q) <= 8]
        if not others:
            return np.zeros(2)
        q = others[0] if direction > 0 else others[-1]     # the furthest one inside 8 frames
        v = (self.m[p] - self.m[q]) / (p - q)
        n = np.linalg.norm(v)
        return v if n <= VMAX else v / n * VMAX

    def predict(self, t, G, N, decay=1.0, forward_only=False):
        """-> (metres, kind, frames to nearest anchor) or None. interp when both neighbours <= G apart,
        else extrapolation up to N frames from the nearest neighbour (velocity decays by `decay` per frame)."""
        if t in self.m:
            return self.m[t], "anchor", 0
        i = bisect.bisect_left(self.f, t)
        p = self.f[i - 1] if i > 0 else None
        n = self.f[i] if i < len(self.f) else None
        if not forward_only and p is not None and n is not None and n - p <= G:
            if np.linalg.norm(self.m[n] - self.m[p]) <= VMAX * (n - p):
                a = (t - p) / (n - p)
                return (1 - a) * self.m[p] + a * self.m[n], "interp", min(t - p, n - t)
        options = []
        if p is not None and t - p <= N:
            h = t - p
            disp = self._vel(p, +1) * (h if decay >= 1 else decay * (1 - decay ** h) / (1 - decay))
            options.append((h, self.m[p] + disp))
        if not forward_only and n is not None and n - t <= N:
            h = n - t
            disp = self._vel(n, -1) * (h if decay >= 1 else decay * (1 - decay ** h) / (1 - decay))
            options.append((h, self.m[n] - disp))
        if options:
            h, m = min(options, key=lambda o: o[0])
            return m, "extrap", h
        return None


def entry(ctx, t, m, state, conf=None, px=None):
    return {"m": np.asarray(m, float), "px": np.asarray(px if px is not None else ctx.px_of(m, t), float),
            "state": state, "conf": conf}


def base_out(ctx, ids, cands):
    """M0 output + the detected anchors it contains."""
    track = baseline_track(cands, ids)
    out, anchors = {}, {}
    for t, (xy, box, interp) in track.items():
        if interp:
            out[t] = entry(ctx, t, xy, "interp")
        else:
            out[t] = entry(ctx, t, xy, "detected", px=centre(box))
            out[t]["box"] = box
            anchors[t] = xy
    return out, Anchors(anchors)


def fill(ctx, out, anchors, ids, G, N, decay, forward_only=False, replace_interp=True):
    """Fill the frames that are not anchors by interpolation / extrapolation. Returns a new dict."""
    res = {t: e for t, e in out.items() if e["state"] not in ("interp", "extrap")} if replace_interp else dict(out)
    for t in ids:
        if t in res:
            continue
        p = anchors.predict(t, G, N, decay, forward_only)
        if p is not None and p[1] != "anchor":
            res[t] = entry(ctx, t, p[0], p[1])
    return res


# --------------------------------------------------------------------------- methods
def m0(ctx, cands, ids, **_):
    return base_out(ctx, ids, cands)[0]


def m1(ctx, cands, ids, G=30, N=0, decay=1.0, forward_only=False, **_):
    out, anchors = base_out(ctx, ids, cands)
    return fill(ctx, out, anchors, ids, G, N, decay, forward_only)


def weak_pass(ctx, out, anchors, cands, ids, gate0, gate1, min_conf, max_size, only=None):
    """Accept a weak / larger candidate when it lies inside the gate around the prediction (chained in time)."""
    pool = filt(cands, min_conf, max_size)
    for t in ids:
        if t in anchors.m or (only is not None and t not in only):
            continue
        cs = pool.get(t, [])
        if not cs:
            continue
        p = anchors.predict(t, 30, 15, 0.9)
        if p is None:
            continue
        gate = gate0 + gate1 * min(p[2], 10)
        near = [(np.linalg.norm(c[0] - p[0]), c) for c in cs]
        near = [(d, c) for d, c in near if d <= gate]
        if not near:
            continue
        _, c = min(near, key=lambda x: x[0])
        anchors.add(t, c[0])
        out[t] = entry(ctx, t, c[0], "weak", conf=c[1], px=centre(c[2]))
        out[t]["box"] = c[2]


def m2(ctx, cands, ids, gate0=1.0, gate1=0.3, min_conf=0.05, max_size=40, G=15, N=0, decay=1.0, **_):
    out, anchors = base_out(ctx, ids, cands)
    weak_pass(ctx, out, anchors, cands, ids, gate0, gate1, min_conf, max_size)
    return fill(ctx, out, anchors, ids, G, N, decay)


# ---- M3: crop re-detection
class CropCache:
    def __init__(self, fresh):
        self.d = {} if fresh or not CROP_CACHE.exists() else json.loads(CROP_CACHE.read_text())
        self.new = 0
        self.seconds = 0.0
        self.calls = 0

    def get(self, key):
        return self.d.get(key)

    def put(self, key, val):
        self.d[key] = val
        self.new += 1

    def save(self):
        CROP_CACHE.write_text(json.dumps(self.d))


def crop_detect(ctx, cache, t, centre_px, S, z, conf=0.05):
    """YOLO on an SxS crop (upscaled by z) around centre_px. Returns [(px centre, conf, size)] in full-frame px."""
    W, H = ctx.info["width"], ctx.info["height"]
    x0 = int(round(min(max(centre_px[0] - S / 2, 0), W - S)))
    y0 = int(round(min(max(centre_px[1] - S / 2, 0), H - S)))
    key = f"{t}|{x0}|{y0}|{S}|{z}"
    hit = cache.get(key)
    if hit is None:
        frame = ctx.reader.get(t)
        crop = frame[y0:y0 + S, x0:x0 + S]
        t0 = time.time()
        res = ctx.get_model().predict(crop, imgsz=int(S * z), conf=conf, iou=0.7, verbose=False)[0]
        dt = time.time() - t0
        cache.seconds += dt
        cache.calls += 1
        hit = []
        names = res.names
        for cls, cf, b in zip(res.boxes.cls.int().tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist()):
            if names[cls] in ("ball", "sports ball"):
                hit.append([x0 + (b[0] + b[2]) / 2 / z * (S * z / crop.shape[1]) if False else x0 + (b[0] + b[2]) / 2,
                            y0 + (b[1] + b[3]) / 2, cf, max(b[2] - b[0], b[3] - b[1])])
        # YOLO returns boxes in the coordinates of the crop (original size), so no rescaling is needed
        cache.put(key, hit)
    return [(np.array([h[0], h[1]]), h[2], h[3]) for h in hit], (x0, y0)


# ---- M5: gate fix (own copy of core's tracker with a faster-growing gate and a look-ahead restart)
def track_ball_fixed(candidates, frame_ids, growth=1.5, max_missed=30, restart_conf=0.4, lookahead=3, max_speed=1.5):
    """Like core.tracking.track_ball, but (a) the gate grows `growth` m per missed frame (a kicked ball is
    ~1.4 m/frame), the track survives max_missed frames, and (b) a confident candidate (>= restart_conf) outside
    the gate restarts the track when a later candidate (within `lookahead` frames) continues it at a plausible speed."""
    from core.tracking import BALL_GATE, BALL_SWITCH_CONF, BallKalman
    measured, kf, missed, segment = {}, None, 0, 0

    def continues(c, idx):
        for dt in range(1, lookahead + 1):
            for c2 in candidates.get(idx + dt, []):
                if np.linalg.norm(c2[0] - c[0]) <= max_speed * dt and c2[1] >= 0.1:
                    return True
        return False

    for idx in frame_ids:
        cands = candidates.get(idx, [])
        if kf is not None:
            pred = kf.predict()
            gate = BALL_GATE + growth * missed
            inside = [(np.hypot(*np.subtract(c[0], pred)), c) for c in cands]
            inside = [(d, c) for d, c in inside if d <= gate]
            best = max(cands, key=lambda c: c[1]) if cands else None
            if inside:
                _, match = min(inside, key=lambda t: t[0])
                if match[1] < BALL_WEAK and best[1] >= BALL_SWITCH_CONF and best is not match:
                    kf, segment, match = BallKalman(best[0]), segment + 1, best
                else:
                    kf.update(match[0])
                measured[idx] = (match[0], match[2], segment)
                missed = 0
                continue
            strong = [c for c in cands if c[1] >= restart_conf and continues(c, idx)]
            if strong:
                c = max(strong, key=lambda c: c[1])
                kf, missed, segment = BallKalman(c[0]), 0, segment + 1
                measured[idx] = (c[0], c[2], segment)
                continue
            missed += 1
            if missed > max_missed:
                kf = None
        if kf is None:
            starts = [c for c in cands if c[1] >= BALL_START_CONF]
            if starts:
                c = max(starts, key=lambda c: c[1])
                kf, missed, segment = BallKalman(c[0]), 0, segment + 1
                measured[idx] = (c[0], c[2], segment)
    return measured


def m5(ctx, cands, ids, G=15, N=0, decay=1.0, only=None, **kw):
    """Gate-fixed tracker; anchors = its measurements, gaps filled like M1(G, N)."""
    measured = track_ball_fixed(filt(cands, 0.1, MAX_BALL_SIZE), ids, **kw)
    out, anchors = {}, {}
    for t, (xy, box, _) in measured.items():
        out[t] = entry(ctx, t, xy, "detected", px=centre(box))
        out[t]["box"] = box
        anchors[t] = xy
    return fill(ctx, out, Anchors(anchors), ids, G, N, decay)


def m3(ctx, cands, ids, cache=None, S=320, z=3, tau=0.15, only=None, G=30, N=15, decay=0.9, N_crop=60,
       log=None, fixed=False, **_):
    if fixed:
        measured = track_ball_fixed(filt(cands, 0.1, MAX_BALL_SIZE), ids)
        out, an = {}, {}
        for t, (xy, box, _s) in measured.items():
            out[t] = entry(ctx, t, xy, "detected", px=centre(box))
            out[t]["box"] = box
            an[t] = xy
        anchors = Anchors(an)
    else:
        out, anchors = base_out(ctx, ids, cands)
    for t in ids:
        if t in anchors.m or (only is not None and t not in only):
            continue
        p = anchors.predict(t, G, N_crop, 0.9)
        if p is None:
            continue
        cpx = ctx.px_of(p[0], t)
        if not (-S / 2 < cpx[0] < ctx.info["width"] + S / 2 and -S / 2 < cpx[1] < ctx.info["height"] + S / 2):
            continue
        found, (x0, y0) = crop_detect(ctx, cache, t, cpx, S, z)
        good = []
        for px, cf, size in found:
            if cf < tau or size > MAX_BALL_SIZE:
                continue
            m = ctx.m_of(px, t)
            if on_pitch(m) and min(np.hypot(*np.subtract(m, s)) for s in SPOTS) > SPOT_RADIUS:
                good.append((cf, px, m))
        if log is not None:
            log.append({"t": t, "crop": (x0, y0, S), "pred_px": cpx.tolist(), "n": len(good)})
        if good:
            cf, px, m = max(good, key=lambda g: g[0])
            anchors.add(t, m)
            out[t] = entry(ctx, t, m, "crop", conf=cf, px=px)
    return fill(ctx, out, anchors, ids, G, N, decay)


# ---- M4: visual trackers on camera-stabilised frames
def _gaps(out, ids, only):
    """[(last detected frame, [gap frames])] for the runs of non-detected frames."""
    res, last, run = [], None, []
    for t in ids:
        det = t in out and out[t]["state"] == "detected"
        if det:
            if last is not None and run:
                res.append((last, run))
            last, run = t, []
        elif last is not None:
            run.append(t)
    if last is not None and run:
        res.append((last, run))
    if only is not None:
        res = [(a, [t for t in r if t in only]) for a, r in res]
        res = [(a, r) for a, r in res if r]
    return res


def m4(ctx, cands, ids, kind="tm", thr=0.6, N=60, only=None, G=30, **_):
    base, anchors = base_out(ctx, ids, cands)
    out = dict(base)
    for d, gap in _gaps(base, ids, only):
        box = base[d]["box"]
        Ha = ctx.hs.for_frame(d)
        Hai = np.linalg.inv(Ha)
        frame_a = ctx.reader.get(d)
        cx, cy = centre(box)
        w = max(size_of(box) + 6, 16)
        x0, y0 = int(cx - w / 2), int(cy - w / 2)
        tpl = frame_a[y0:y0 + int(w), x0:x0 + int(w)].copy()
        if tpl.shape[0] < 8 or tpl.shape[1] < 8:
            continue
        local = Anchors({d: anchors.m[d]})
        # velocity from the detections before d
        for q in (d - 1, d - 2, d - 3, d - 4):
            if q in anchors.m:
                local.add(q, anchors.m[q])
        tracker = None
        if kind == "mil":
            tracker = cv2.TrackerMIL_create()
            tracker.init(frame_a, (int(x0), int(y0), int(w), int(w)))
        misses = 0
        for t in gap:
            if t - d > N or misses > 3:
                break
            fr = ctx.reader.get(t)
            if fr is None:
                break
            M = Hai @ ctx.hs.for_frame(t)           # pixels of frame t -> pixels of frame d
            if kind == "tm":
                p = local.predict(t, 0, N, 0.9, True)
                pa = project([p[0]], Ha.copy() if False else Hai)[0]
                R = int(60 + 25 * misses)
                ox, oy = int(pa[0] - R), int(pa[1] - R)
                T = np.array([[1, 0, -ox], [0, 1, -oy], [0, 0, 1]], float)
                roi = cv2.warpPerspective(fr, T @ M, (2 * R, 2 * R), flags=cv2.INTER_LINEAR)
                resp = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(resp)
                if score < thr:
                    misses += 1
                    continue
                pa_found = np.array([ox + loc[0] + tpl.shape[1] / 2, oy + loc[1] + tpl.shape[0] / 2])
                conf = float(score)
            else:
                warped = cv2.warpPerspective(fr, M, (ctx.info["width"], ctx.info["height"]))
                ok, bb = tracker.update(warped)
                if not ok:
                    break
                pa_found = np.array([bb[0] + bb[2] / 2, bb[1] + bb[3] / 2])
                conf = None
            m = project([pa_found], Ha)[0]
            prev = local.f[-1]
            if not on_pitch(m) or np.linalg.norm(m - local.m[prev]) > VMAX * (t - prev) + 1.0:
                misses += 1
                continue
            misses = 0
            local.add(t, m)
            px_t = project([m], np.linalg.inv(ctx.hs.for_frame(t)))[0]
            out[t] = entry(ctx, t, m, "tracker", conf=conf, px=px_t)
    return fill(ctx, out, anchors, ids, G, 0, 1.0)


# --------------------------------------------------------------------------- evaluation
def truth_of(ctx, ids, cands):
    """Frames where the baseline detects the ball with conf >= 0.3: these are the 'known' balls."""
    out, _ = base_out(ctx, ids, cands)
    truth = {}
    for t, e in out.items():
        if e["state"] == "detected":
            c = [c for c in cands.get(t, []) if tuple(c[2]) == tuple(e["box"])]
            if c and c[0][1] >= 0.3 and size_of(c[0][2]) <= MAX_BALL_SIZE:
                truth[t] = (e["px"], e["m"])
    return truth


def hide(cands, window, thr=0.1):
    """Remove ball candidates with conf >= thr in the window (what the pipeline would see in a real gap)."""
    lo, hi = window
    return {t: ([c for c in cs if c[1] < thr] if lo <= t < hi else cs) for t, cs in cands.items()}


def pick_windows(truth, k, n_max, rng_seed=0):
    ts = sorted(truth)
    tset = set(ts)
    ok = []
    for s in ts:
        s0 = s + 1
        if s0 in tset or s0 + k > 659:
            pass
        have = sum(1 for t in range(s0, s0 + k) if t in tset)
        if have >= max(3, k // 2) and s0 + k <= 659:
            ok.append(s0)
    if len(ok) > n_max:
        idx = np.linspace(0, len(ok) - 1, n_max).round().astype(int)
        ok = [ok[i] for i in sorted(set(idx))]
    return ok


def run_sim(ctx, truth, method, k, starts, **kw):
    errs, hits, total, secs = [], 0, 0, 0.0
    for s in starts:
        window = (s, s + k)
        c = hide(ctx.cands, window)
        only = set(range(s, s + k))
        t0 = time.time()
        out = method(ctx, c, ctx.ids, only=only, **kw)
        secs += time.time() - t0
        for t in range(s, s + k):
            if t not in truth:
                continue
            total += 1
            if t in out:
                hits += 1
                errs.append(float(np.linalg.norm(out[t]["px"] - truth[t][0])))
    return summarize(errs, hits, total), secs


def summarize(errs, found, total):
    e = np.array(errs) if errs else np.array([np.nan])
    return {"n": total, "recall": found / total if total else 0.0,
            "med": float(np.nanmedian(e)), "p90": float(np.nanpercentile(e, 90)) if errs else float("nan"),
            "w10": float(np.mean(e <= 10)) if errs else 0.0, "w25": float(np.mean(e <= OK_PX)) if errs else 0.0,
            "good": float(np.sum(e <= OK_PX)) / total if total and errs else 0.0}


def load_gt():
    rows = []
    for name in ("ground_truth.json", "ground_truth_feet.json"):
        d = json.loads((ROOT / "data" / name).read_text())
        for f, v in d["frames"].items():
            rows.append({"file": name.replace(".json", ""), "frame": int(f), "visible": bool(v["ball_visible"]),
                         "px": np.array(v["ball"]["px"]) if v["ball_visible"] else None,
                         "m": np.array(v["ball"]["m"]) if v["ball_visible"] else None})
    return rows


def eval_gt(out, gt, base_state):
    r = {"vis": 0, "ok": 0, "wrong": 0, "miss": 0, "neg": 0, "false": 0, "errs_px": [], "errs_m": [], "detail": {}}
    for g in gt:
        e = out.get(g["frame"])
        key = f"{g['file'][13:] or 'main'}:{g['frame']}"
        if g["visible"]:
            r["vis"] += 1
            if e is None:
                r["miss"] += 1
                r["detail"][key] = "-"
            else:
                d = float(np.linalg.norm(e["px"] - g["px"]))
                r["errs_px"].append(d)
                r["errs_m"].append(float(np.linalg.norm(e["m"] - g["m"])))
                r["ok" if d <= OK_PX else "wrong"] += 1
                r["detail"][key] = f"{d:.0f}px ({e['state'][:4]})"
        else:
            r["neg"] += 1
            if e is not None:
                r["false"] += 1
                r["detail"][key] = f"FALSE ({e['state'][:4]})"
    return r


def states(out, ids):
    c = {}
    for t in ids:
        s = out[t]["state"] if t in out else "none"
        c[s] = c.get(s, 0) + 1
    return c


# --------------------------------------------------------------------------- samples
def save_samples(ctx, out, name, frames, size=200, z=3):
    SAMPLES.mkdir(exist_ok=True)
    paths = []
    for t in frames:
        fr = ctx.reader.get(t)
        px = out[t]["px"]
        x0 = int(min(max(px[0] - size / 2, 0), ctx.info["width"] - size))
        y0 = int(min(max(px[1] - size / 2, 0), ctx.info["height"] - size))
        crop = cv2.resize(fr[y0:y0 + size, x0:x0 + size], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
        c = ((px[0] - x0) * z, (px[1] - y0) * z)
        cv2.circle(crop, (int(c[0]), int(c[1])), 30, (0, 0, 255), 2)
        cv2.putText(crop, f"{name} f{t} {out[t]['state']} {out[t]['conf'] or ''}"[:48], (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        p = SAMPLES / f"{name}_f{t}.png"
        cv2.imwrite(str(p), crop)
        paths.append(p.name)
    return paths


# --------------------------------------------------------------------------- main
def fmt(v, p=2):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-crops", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--quick", action="store_true", help="fewer simulation windows (debug)")
    args = ap.parse_args()
    t_start = time.time()
    ctx = Ctx()
    ids = ctx.ids
    cache = CropCache(args.fresh)
    results = {"e1": {}, "e2": {}, "time": {}}

    truth = truth_of(ctx, ids, ctx.cands)
    print(f"truth frames (baseline detected, conf>=0.3): {len(truth)}")

    # ---------------- E1
    ks = [5, 10, 20, 40]
    nmax = 8 if args.quick else 40
    windows = {k: pick_windows(truth, k, nmax) for k in ks}
    print("windows per k:", {k: len(v) for k, v in windows.items()})
    e1_methods = {
        "M0 basis": (m0, {}),
        "M1a interp<=30": (m1, dict(G=30, N=0)),
        "M1b interp<=60": (m1, dict(G=60, N=0)),
        "M1c interp60+extrap15 (v const)": (m1, dict(G=60, N=15)),
        "M1d interp60+extrap30 (decay .9)": (m1, dict(G=60, N=30, decay=0.9)),
        "M1e alleen extrap vooruit 40 (v const)": (m1, dict(G=0, N=40, forward_only=True)),
        "M1f alleen extrap vooruit 40 (decay .9)": (m1, dict(G=0, N=40, decay=0.9, forward_only=True)),
        "M5a gate-fix (groei 1.5, 30 gemist, restart)": (m5, dict(G=15)),
        "M5b gate-fix + interp60 + extrap15": (m5, dict(G=60, N=15)),
        "M2a zwak 0.05 strak gate (1+0.3d m)": (m2, dict(gate0=1.0, gate1=0.3)),
        "M2b zwak 0.05 ruim gate (2+0.5d m)": (m2, dict(gate0=2.0, gate1=0.5)),
        "M2c zwak 0.05, size<=60, strak": (m2, dict(gate0=1.0, gate1=0.3, max_size=60)),
        "M4a template matching": (m4, dict(kind="tm")),
        "M4b OpenCV TrackerMIL": (m4, dict(kind="mil")),
    }
    for name, (fn, kw) in e1_methods.items():
        results["e1"][name] = {}
        for k in ks:
            nm = nmax if "M4" not in name else (6 if args.quick else 25)
            starts = windows[k] if len(windows[k]) <= nm else [windows[k][i] for i in
                                                               sorted(set(np.linspace(0, len(windows[k]) - 1, nm).round().astype(int)))]
            s, secs = run_sim(ctx, truth, fn, k, starts, **kw)
            s["windows"] = len(starts)
            s["secs"] = secs
            results["e1"][name][k] = s
        print("E1", name, {k: (round(v["recall"], 2), round(v["med"], 1)) for k, v in results["e1"][name].items()})

    # ---------------- E1 M3 (sample; optimistic, see results.md)
    if not args.skip_crops:
        for label, S, z in (("M3a crop 640 x0.5 (native schaal)", 640, 0.5), ("M3b crop 960 x0.5", 960, 0.5), ("M3x crop 320 x3 (opgeschaald)", 320, 3)):
            if label.startswith(("M3b", "M3x")) and args.quick:
                continue
            results["e1"][label] = {}
            for k in (10, 20, 40):
                nw = 3 if args.quick else (6 if k < 40 else 5)
                starts = [windows[k][i] for i in sorted(set(np.linspace(0, len(windows[k]) - 1, nw).round().astype(int)))]
                errs, hits, total, contain, nframes = [], 0, 0, 0, 0
                t0 = time.time()
                for s in starts:
                    c = hide(ctx.cands, (s, s + k))
                    log = []
                    out = m3(ctx, c, ids, cache=cache, S=S, z=z, only=set(range(s, s + k)), log=log)
                    for t in range(s, s + k):
                        if t in truth:
                            total += 1
                            if t in out and out[t]["state"] == "crop":
                                hits += 1
                                errs.append(float(np.linalg.norm(out[t]["px"] - truth[t][0])))
                    for lg in log:
                        if lg["t"] in truth:
                            x0, y0, sz = lg["crop"]
                            tp = truth[lg["t"]][0]
                            nframes += 1
                            contain += x0 <= tp[0] <= x0 + sz and y0 <= tp[1] <= y0 + sz
                r = summarize(errs, hits, total)
                r.update({"windows": len(starts), "contain": contain / nframes if nframes else 0.0,
                          "secs": time.time() - t0})
                results["e1"][label][k] = r
                print("E1", label, k, {a: round(b, 2) for a, b in r.items()})
            cache.save()

    # ---------------- E2 (real gaps)
    gt = load_gt()
    base, banchors = base_out(ctx, ids, ctx.cands)
    base_none = {t for t in ids if t not in base}
    print("baseline states:", states(base, ids))
    e2_methods = {
        "M0 basis": lambda: base,
        "M1a interp<=30": lambda: m1(ctx, ctx.cands, ids, G=30, N=0),
        "M1b interp<=60": lambda: m1(ctx, ctx.cands, ids, G=60, N=0),
        "M1c interp60+extrap15": lambda: m1(ctx, ctx.cands, ids, G=60, N=15),
        "M1d interp60+extrap30 decay": lambda: m1(ctx, ctx.cands, ids, G=60, N=30, decay=0.9),
        "M5a gate-fix": lambda: m5(ctx, ctx.cands, ids, G=15),
        "M5b gate-fix + interp60 + extrap15": lambda: m5(ctx, ctx.cands, ids, G=60, N=15),
        "M2a zwak strak": lambda: m2(ctx, ctx.cands, ids, gate0=1.0, gate1=0.3),
        "M2b zwak ruim": lambda: m2(ctx, ctx.cands, ids, gate0=2.0, gate1=0.5),
        "M2c zwak size<=60 strak": lambda: m2(ctx, ctx.cands, ids, gate0=1.0, gate1=0.3, max_size=60),
        "M2d zwak strak + M1c vulling": lambda: m2(ctx, ctx.cands, ids, gate0=1.0, gate1=0.3, G=60, N=15),
        "M4a template matching": lambda: m4(ctx, ctx.cands, ids, kind="tm"),
        "M4b OpenCV TrackerMIL": lambda: m4(ctx, ctx.cands, ids, kind="mil"),
    }
    if not args.skip_crops:
        for label, S, z in (("M3a crop 640 x0.5 tau .15", 640, 0.5), ("M3b crop 960 x0.5 tau .15", 960, 0.5), ("M3x crop 320 x3 tau .15 (opgeschaald)", 320, 3)):
            e2_methods[label] = (lambda S=S, z=z: m3(ctx, ctx.cands, ids, cache=cache, S=S, z=z, tau=0.15))
        e2_methods["M3d crop 640 x0.5 tau .15 + gate-fix"] = lambda: m3(ctx, ctx.cands, ids, cache=cache, S=640, z=0.5, tau=0.15, fixed=True)
        e2_methods["M3c crop 640 x0.5 tau .30"] = lambda: m3(ctx, ctx.cands, ids, cache=cache, S=640, z=0.5, tau=0.30)
    outs = {}
    for name, fn in e2_methods.items():
        t0 = time.time()
        out = fn()
        secs = time.time() - t0
        outs[name] = out
        ev = eval_gt(out, gt, None)
        st = states(out, ids)
        gained = sum(1 for t in base_none if t in out)
        results["e2"][name] = {"states": st, "gained_none": gained, "total_ball": sum(1 for t in ids if t in out),
                               "secs": secs, "gt": {k: v for k, v in ev.items() if k not in ("errs_px", "errs_m")},
                               "gt_med_px": float(np.median(ev["errs_px"])) if ev["errs_px"] else None,
                               "gt_med_m": float(np.median(ev["errs_m"])) if ev["errs_m"] else None}
        print("E2", name, st, "gained", gained, "gt", {k: ev[k] for k in ("vis", "ok", "wrong", "miss", "neg", "false")})
        if name.startswith("M3"):
            cache.save()
    cache.save()

    # combination: weak (M2a) + crop + long fill
    if not args.skip_crops:
        name = "M6 M5 + M2a + M3a(640) + M1c vulling"
        t0 = time.time()
        measured = track_ball_fixed(filt(ctx.cands, 0.1, MAX_BALL_SIZE), ids)
        out, an = {}, {}
        for t, (xy, box, _s) in measured.items():
            out[t] = entry(ctx, t, xy, "detected", px=centre(box))
            out[t]["box"] = box
            an[t] = xy
        anchors = Anchors(an)
        weak_pass(ctx, out, anchors, ctx.cands, ids, 1.0, 0.3, 0.05, 40)
        # crop pass on what is still open, chained on the anchors including the weak ones
        for t in ids:
            if t in anchors.m:
                continue
            p = anchors.predict(t, 30, 60, 0.9)
            if p is None:
                continue
            cpx = ctx.px_of(p[0], t)
            found, _ = crop_detect(ctx, cache, t, cpx, 640, 0.5)
            good = []
            for px, cf, size in found:
                if cf < 0.15 or size > MAX_BALL_SIZE:
                    continue
                m = ctx.m_of(px, t)
                if on_pitch(m) and min(np.hypot(*np.subtract(m, s)) for s in SPOTS) > SPOT_RADIUS:
                    good.append((cf, px, m))
            if good:
                cf, px, m = max(good, key=lambda g: g[0])
                anchors.add(t, m)
                out[t] = entry(ctx, t, m, "crop", conf=cf, px=px)
        out = fill(ctx, out, anchors, ids, 60, 15, 1.0)
        outs[name] = out
        ev = eval_gt(out, gt, None)
        results["e2"][name] = {"states": states(out, ids), "gained_none": sum(1 for t in base_none if t in out),
                               "total_ball": sum(1 for t in ids if t in out), "secs": time.time() - t0,
                               "gt": {k: v for k, v in ev.items() if k not in ("errs_px", "errs_m")},
                               "gt_med_px": float(np.median(ev["errs_px"])) if ev["errs_px"] else None,
                               "gt_med_m": float(np.median(ev["errs_m"])) if ev["errs_m"] else None}
        print("E2", name, results["e2"][name]["states"])
        cache.save()

    # crop timing
    results["time"]["crop_calls"] = cache.calls
    results["time"]["crop_seconds_per_call"] = cache.seconds / cache.calls if cache.calls else None
    results["time"]["total_seconds"] = time.time() - t_start

    # samples (M3a, M3b, M4a): up to 8 found frames, spread over the real gaps
    sample_paths = {}
    for name in [n for n in outs if n.startswith(("M3a", "M3b", "M3d", "M4a", "M5", "M6", "M2a"))]:
        fr = sorted(t for t, e in outs[name].items() if e["state"] in ("crop", "tracker", "weak") and t in base_none)
        if fr:
            pick = [fr[i] for i in sorted(set(np.linspace(0, len(fr) - 1, min(8, len(fr))).round().astype(int)))]
            sample_paths[name] = save_samples(ctx, outs[name], name.split()[0], pick)
    results["samples"] = sample_paths
    (HERE / "results.json").write_text(json.dumps(results, indent=1, default=float))
    # per-frame outputs of every method for later inspection
    (HERE / "outputs.json").write_text(json.dumps(
        {n: {str(t): [round(float(e["px"][0]), 1), round(float(e["px"][1]), 1), e["state"]] for t, e in o.items()}
         for n, o in outs.items()}))
    write_md(results, windows, len(truth))
    print("done in", round(time.time() - t_start), "s")


def write_md(r, windows, ntruth):
    L = ["# Resultaten: bal voorspellen uit eerdere frames", "",
         "Automatisch gegenereerd door `run_experiments.py`. Conclusie onderaan komt uit `conclusie.md`.", ""]
    L += ["## E1: gesimuleerde gaten", "",
          f"Waarheid = frames waar de pipeline de bal met conf >= 0,3 detecteert ({ntruth} frames). In een venster van k frames "
          "worden alle balkandidaten met conf >= 0,1 verborgen (zwakkere blijven zichtbaar, zoals in een echt gat). "
          "Recall = aandeel waarheidsframes in het venster waarvoor de methode een bal geeft; fout = pixelafstand tot de verborgen "
          f"detectie; goed = recall met fout <= {OK_PX} px. `w10/w25` = aandeel van de gevonden frames binnen 10/25 px.", ""]
    for k in (5, 10, 20, 40):
        L += [f"### k = {k} frames ({windows[k] and len(windows[k])} geschikte vensters)", "",
              "| Methode | vensters | frames | recall | goed | mediaan px | p90 px | w10 | w25 |", "|---|---|---|---|---|---|---|---|---|"]
        for name, per in r["e1"].items():
            if k in per:
                s = per[k]
                L.append(f"| {name} | {s['windows']} | {s['n']} | {s['recall']:.0%} | {s['good']:.0%} | {fmt(s['med'], 1)} | "
                         f"{fmt(s['p90'], 1)} | {s['w10']:.0%} | {s['w25']:.0%} |")
        L.append("")
    L += ["Opmerking M3 (crop): in de simulatie ziet het model in de crop het echte frame, waarin de bal wel gevonden "
          "was. M3-recall is dus optimistisch; `contain` (aandeel frames waar de echte bal binnen de voorspelde crop lag) "
          "meet de voorspelling zelf: " + "; ".join(
              f"{n} k={k}: {s.get('contain', 0):.0%}" for n, per in r["e1"].items() if n.startswith("M3") for k, s in per.items()), ""]
    L += ["## E2: echte gaten (660 frames)", "",
          "| Methode | bal totaal | bij 'none' erbij | detected | interp | extrap | weak | crop | tracker | none | tijd (s) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in r["e2"].items():
        st = s["states"]
        L.append(f"| {name} | {s['total_ball']} | {s['gained_none']} | {st.get('detected', 0)} | {st.get('interp', 0)} | "
                 f"{st.get('extrap', 0)} | {st.get('weak', 0)} | {st.get('crop', 0)} | {st.get('tracker', 0)} | "
                 f"{st.get('none', 0)} | {s['secs']:.1f} |")
    L += ["", "### E2: nauwkeurigheid op de ground-truthframes (beide bestanden, 28 labels)", "",
          f"ok = bal gegeven binnen {OK_PX} px van het label; fout = bal gegeven maar verder; gemist = geen bal; "
          "vals = bal gegeven terwijl `ball_visible=false`.", "",
          "| Methode | zichtbaar | ok | fout | gemist | niet zichtbaar | vals | mediaan px | mediaan m |", "|---|---|---|---|---|---|---|---|---|"]
    for name, s in r["e2"].items():
        g = s["gt"]
        L.append(f"| {name} | {g['vis']} | {g['ok']} | {g['wrong']} | {g['miss']} | {g['neg']} | {g['false']} | "
                 f"{fmt(s['gt_med_px'], 1)} | {fmt(s['gt_med_m'], 2)} |")
    L += ["", "Per GT-frame waar een methode afwijkt van de basis:", ""]
    keys = list(next(iter(r["e2"].values()))["gt"]["detail"])
    for name, s in r["e2"].items():
        d = s["gt"]["detail"]
        L.append(f"- {name}: " + ", ".join(f"{k} {d.get(k, 'n.v.t.')}" for k in sorted(d) if k.split(":")[1] in
                                         ("210", "360", "390", "420", "600", "240", "270", "300", "630", "30")))
    t = r["time"]
    L += ["", "## Rekentijd", "",
          f"- Crop-herdetectie (YOLO, imgsz 960 op CPU): {fmt(t.get('crop_seconds_per_call'), 2)} s per crop "
          f"({t.get('crop_calls')} crops deze run; bij een cachehit 0 s).",
          f"- Totale looptijd van dit script: {t['total_seconds']:.0f} s.", ""]
    L += ["## Voorbeeldplaatjes (map `samples/`)", ""]
    for name, ps in r.get("samples", {}).items():
        L.append(f"- {name}: " + ", ".join(f"`samples/{p}`" for p in ps))
    L.append("")
    concl = HERE / "conclusie.md"
    if concl.exists():
        L += [concl.read_text(encoding="utf-8")]
    (HERE / "results.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
