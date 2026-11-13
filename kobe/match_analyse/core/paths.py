"""All paths of match_analyse, computed from this file: the folder can be copied anywhere."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent       # the match_analyse folder
MODELS_DIR = ROOT / "models"                        # weights (.pt), local
DATA_DIR = ROOT / "data"                            # match.mp4 + keypoints.json
CACHE_DIR = ROOT / "cache"                          # cached detections, not committed
OUTPUT_DIR = ROOT / "output"                        # videos / csv files, not committed

DEFAULT_WEIGHTS = MODELS_DIR / "best.pt"
VIDEO = DATA_DIR / "match.mp4"
KEYPOINTS_JSON = DATA_DIR / "keypoints.json"        # clicked pitch keypoints, per keyframe
