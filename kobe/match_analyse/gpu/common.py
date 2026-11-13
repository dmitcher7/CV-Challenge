"""Shared helpers of the GPU jobs: paths, the run context, dataset yamls, label remapping, leakage check.

The pure functions (remap_label_lines, frame_number, find_overlap, best_threshold) have no ultralytics
dependency and are tested in tests/test_gpu_jobs.py.
"""

import hashlib
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

GPU_DIR = Path(__file__).resolve().parent
ROOT = GPU_DIR.parent                                   # the match_analyse folder
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import DATA_DIR, MODELS_DIR, VIDEO  # noqa: E402

DATASETS_DIR = ROOT / "datasets"
RESULTS_DIR = GPU_DIR / "results"
BEST_WEIGHTS = MODELS_DIR / "best.pt"
COCO_WEIGHTS = MODELS_DIR / "yolo26l_coco.pt"
FT_WEIGHTS = MODELS_DIR / "best_ball_ft.pt"
DEFAULT_TEST_DATA = DATASETS_DIR / "notebook_split" / "data.yaml"
DEFAULT_TRAIN_DATA = DATASETS_DIR / "ball_finetune" / "data.yaml"
BEST_TRAIN_DATA = DATASETS_DIR / "roboflow_train" / "data.yaml"   # what best.pt was trained on (all 100 frames)
GT_BODY = DATA_DIR / "ground_truth.json"
GT_FEET = DATA_DIR / "ground_truth_feet.json"
EXPECTED_ULTRALYTICS = "8.4.182"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PERSON_NAMES = {"player_A", "player_B", "player_B_goal_keeper", "referee", "person"}
BALL_NAMES = {"ball", "sports ball"}
MERGED_NAMES = {0: "ball", 1: "person"}                 # the fair "COCO view": ball + all persons together
MERGED_ID = {"ball": 0, "person": 1}


class JobError(RuntimeError):
    """An expected failure with a clear (Dutch) message for the teammate; no traceback needed."""


# ---------------------------------------------------------------- run context

@dataclass
class Ctx:
    """Settings shared by all jobs of one run."""
    out: Path
    device: str = "cpu"
    quick: bool = False
    test_data: Path = DEFAULT_TEST_DATA
    test_split: str = "test"
    val_imgsz: int = 960
    video: Path = VIDEO
    best: Path = BEST_WEIGHTS
    coco: Path = COCO_WEIGHTS
    ft: Path = FT_WEIGHTS
    info: dict = field(default_factory=dict)          # results the jobs pass to each other / to the summary

    @property
    def imgsz(self):
        return 320 if self.quick else self.val_imgsz

    def sub(self, name):
        d = self.out / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_json(self, name, obj):
        path = self.out / name
        path.write_text(json.dumps(obj, indent=1, default=_json_default), encoding="utf-8")
        return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def default_device():
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def new_results_dir(base=RESULTS_DIR, quick=False):
    stamp = time.strftime("%Y-%m-%d_%H%M%S") + ("_quick" if quick else "")
    out = Path(base) / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def common_args(parser):
    """Arguments every standalone job understands."""
    parser.add_argument("--device", default=None, help="0 = eerste GPU, cpu = processor (standaard: 0 als CUDA er is)")
    parser.add_argument("--quick", action="store_true", help="rooktest: klein en snel, ook zonder GPU")
    parser.add_argument("--test-data", default=str(DEFAULT_TEST_DATA), help="data.yaml van de testset")
    parser.add_argument("--test-split", default="test", help="split in die data.yaml (test / val / train)")
    parser.add_argument("--out", default=None, help="resultatenmap (standaard gpu/results/<tijdstip>)")
    return parser


def ctx_from_args(args):
    out = Path(args.out) if args.out else new_results_dir(quick=args.quick)
    out.mkdir(parents=True, exist_ok=True)
    return Ctx(out=out, device=args.device or default_device(), quick=args.quick,
               test_data=Path(args.test_data), test_split=args.test_split)


# ---------------------------------------------------------------- dataset yamls

def load_yaml(path):
    path = Path(path)
    if not path.exists():
        raise JobError(f"data.yaml niet gevonden: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, list):
        data["names"] = dict(enumerate(names))
    elif isinstance(names, dict):
        data["names"] = {int(k): v for k, v in names.items()}
    else:
        raise JobError(f"{path}: geen 'names' (klassennamen) in de data.yaml")
    return data


def split_dirs(yaml_path, split):
    """Image folders of one split of a data.yaml (relative paths are resolved like ultralytics, from the yaml)."""
    yaml_path = Path(yaml_path).resolve()
    data = load_yaml(yaml_path)
    base = Path(data["path"]) if data.get("path") else yaml_path.parent
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()
    entry = data.get(split)
    if not entry:
        raise JobError(f"{yaml_path}: split '{split}' staat niet in de data.yaml")
    dirs = []
    for e in entry if isinstance(entry, list) else [entry]:
        p = (base / e).resolve()
        if not p.exists() and str(e).startswith("../"):   # Roboflow export style
            p = (base / str(e)[3:]).resolve()
        if not p.exists():
            raise JobError(f"{yaml_path}: map van split '{split}' bestaat niet: {p}")
        dirs.append(p)
    return dirs


def images_of(dirs):
    files = []
    for d in dirs:
        d = Path(d)
        files += sorted(f for f in d.rglob("*") if f.suffix.lower() in IMG_EXT) if d.is_dir() else [d]
    return files


def label_of(img):
    """YOLO label file of an image (.../images/x.jpg -> .../labels/x.txt)."""
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def resolved_yaml(src, dst, splits=None):
    """Write a copy of a data.yaml with absolute paths (works from every working directory). Returns dst."""
    data = load_yaml(src)
    out = {"names": data["names"], "nc": len(data["names"])}
    for split in splits or ("train", "val", "test"):
        if data.get(split):
            dirs = split_dirs(src, split)
            out[split] = [str(d) for d in dirs] if len(dirs) > 1 else str(dirs[0])
    for split in ("train", "val"):        # ultralytics requires both keys, even for a pure test evaluation
        out.setdefault(split, next(v for k, v in out.items() if k not in ("names", "nc")))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return Path(dst)


# ---------------------------------------------------------------- label remapping (fair COCO comparison)

def merged_map(names):
    """{class id in a dataset/model: merged id (0 ball, 1 person)}; classes that are neither are left out."""
    out = {}
    for i, n in names.items():
        if n in BALL_NAMES:
            out[int(i)] = MERGED_ID["ball"]
        elif n in PERSON_NAMES:
            out[int(i)] = MERGED_ID["person"]
    return out


def remap_label_lines(lines, mapping):
    """YOLO label lines with the class id replaced via mapping; lines of unmapped classes are dropped."""
    out = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        cls = int(float(parts[0]))
        if cls in mapping:
            out.append(" ".join([str(mapping[cls])] + parts[1:]))
    return out


def build_merged_dataset(yaml_path, split, dst):
    """Copy of one split with labels remapped to {0: ball, 1: person}. Returns (yaml path, counts)."""
    data = load_yaml(yaml_path)
    mapping = merged_map(data["names"])
    if MERGED_ID["ball"] not in mapping.values():
        raise JobError(f"{yaml_path}: geen klasse 'ball' gevonden in names {data['names']}")
    dst = Path(dst)
    img_dir, lab_dir = dst / "images", dst / "labels"
    for d in (img_dir, lab_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    counts = {"images": 0, "ball": 0, "person": 0}
    for img in images_of(split_dirs(yaml_path, split)):
        shutil.copy2(img, img_dir / img.name)
        lab = label_of(img)
        lines = lab.read_text(encoding="utf-8").splitlines() if lab.exists() else []
        new = remap_label_lines(lines, mapping)
        (lab_dir / (img.stem + ".txt")).write_text("\n".join(new) + ("\n" if new else ""), encoding="utf-8")
        counts["images"] += 1
        counts["ball"] += sum(line.split()[0] == "0" for line in new)
        counts["person"] += sum(line.split()[0] == "1" for line in new)
    yml = dst / "data.yaml"
    yml.write_text(yaml.safe_dump({"names": MERGED_NAMES, "nc": 2, "train": str(img_dir), "val": str(img_dir),
                                   "test": str(img_dir)},
                                  sort_keys=False), encoding="utf-8")
    return yml, counts


# ---------------------------------------------------------------- leakage

FRAME_RE = re.compile(r"frame_(\d+)", re.IGNORECASE)


def frame_number(name):
    """Video frame number in a file name ('frame_0600_idx_00600_jpg.rf.x.jpg' -> 600), else None."""
    m = FRAME_RE.search(Path(name).name)
    return int(m.group(1)) if m else None


def file_hash(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def find_overlap(train_files, test_files, hasher=file_hash):
    """Test images that also occur in the training images: same frame number OR identical file content.

    Returns {"n_test", "n_train", "n_overlap", "by_frame", "by_hash", "frames"} (frames = sorted overlapping
    frame numbers, by_frame/by_hash = number of test images found that way).
    """
    train_frames = {frame_number(f) for f in train_files} - {None}
    train_hashes = {hasher(f) for f in train_files}
    by_frame, by_hash, overlap, frames = 0, 0, 0, set()
    for f in test_files:
        fr = frame_number(f)
        same_frame = fr is not None and fr in train_frames
        same_hash = hasher(f) in train_hashes
        by_frame += same_frame
        by_hash += same_hash
        if same_frame or same_hash:
            overlap += 1
            if fr is not None:
                frames.add(fr)
    return {"n_test": len(test_files), "n_train": len(train_files), "n_overlap": overlap,
            "by_frame": by_frame, "by_hash": by_hash, "frames": sorted(frames)}


def dataset_overlap(train_yaml, test_yaml, test_split, train_splits=("train", "val")):
    """find_overlap between the given splits of two data.yamls (missing train splits are skipped)."""
    train = []
    data = load_yaml(train_yaml)
    for s in train_splits:
        if data.get(s):
            train += images_of(split_dirs(train_yaml, s))
    test = images_of(split_dirs(test_yaml, test_split))
    return find_overlap(sorted(set(train)), test)


# ---------------------------------------------------------------- thresholds

def smooth(y, fraction=0.05):
    """Box filter over `fraction` of the points, edges padded (same idea as ultralytics.utils.metrics.smooth)."""
    y = np.asarray(y, float)
    nf = round(len(y) * fraction * 2) // 2 + 1
    if nf <= 1 or len(y) < nf:
        return y
    pad = np.ones(nf // 2)
    return np.convolve(np.concatenate([pad * y[0], y, pad * y[-1]]), np.ones(nf) / nf, mode="valid")


def best_threshold(px, f1, p=None, r=None, smoothing=0.05):
    """Confidence with the highest (smoothed) F1 on a curve: {"conf", "f1", "precision", "recall"} at that point.

    The raw F1 curve often peaks right before a cliff (a few detections more or less); smoothing over 5 % of the
    confidence axis picks a point on the plateau instead. f1/precision/recall are the raw values there.
    """
    px, f1 = np.asarray(px, float), np.asarray(f1, float)
    if not len(f1) or not np.isfinite(f1).any():
        return {"conf": None, "f1": None, "precision": None, "recall": None}
    i = int(np.nanargmax(smooth(np.nan_to_num(f1), smoothing) if smoothing else f1))
    if np.nanmax(f1) <= 0:      # the class is never found: no meaningful threshold
        return {"conf": None, "f1": 0.0, "precision": None, "recall": None}
    return {"conf": float(px[i]), "f1": float(f1[i]),
            "precision": None if p is None else float(np.asarray(p)[i]),
            "recall": None if r is None else float(np.asarray(r)[i])}


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "ja" if v else "nee"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}f}"
    return str(v)


def md_table(rows, cols):
    """Markdown table of a list of dicts."""
    if not rows:
        return "_(geen gegevens)_\n"
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def write_csv(path, rows, cols=None):
    import csv
    cols = cols or (list(rows[0]) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    return Path(path)


# ---------------------------------------------------------------- detection caches that travel to another PC

CACHE_EXPORT = "cache_export"          # folder in the results with the detection caches (portable file names)


# ---------------------------------------------------------------- resumable runs

STATE_FILE = "state.json"


def load_state(out):
    path = Path(out) / STATE_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_state(out, state):
    path = Path(out) / STATE_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def find_resumable(base, settings):
    """Newest unfinished run under base with exactly these settings, else None."""
    base = Path(base)
    if not base.exists():
        return None
    for d in sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
        st = load_state(d)
        if st is None:
            continue
        if not st.get("finished") and st.get("settings") == settings:
            return d
    return None


def step_done(state, name, force=False):
    """True when the step finished ok in an earlier attempt of this run (and --force is not given)."""
    return not force and bool(state) and state.get("steps", {}).get(name, {}).get("status") == "ok"
