"""All paths of the final upload, computed from this file: the final_upload folder can be copied anywhere.

Layout:
    final_upload/
        models/psg_vs_inter_CV.pt     the final detection model
        data/raw/match.mp4            the clip
        data/keypoints.json           clicked pitch keypoints, per keyframe
        streamlit/                    this code (app, core, analytics, evaluation) + cache/ and output/
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent       # the streamlit folder (code, cache, output)
UPLOAD_DIR = ROOT.parent                            # the final_upload folder
MODELS_DIR = UPLOAD_DIR / "models"                  # weights (.pt)
DATA_DIR = UPLOAD_DIR / "data"                      # keypoints.json, ground truth, datasets, results
CACHE_DIR = ROOT / "cache"                          # cached detections
OUTPUT_DIR = ROOT / "output"                        # videos / csv files

DEFAULT_WEIGHTS = MODELS_DIR / "psg_vs_inter_CV.pt"
VIDEO = DATA_DIR / "raw" / "match.mp4"
KEYPOINTS_JSON = DATA_DIR / "keypoints.json"        # clicked pitch keypoints, per keyframe
