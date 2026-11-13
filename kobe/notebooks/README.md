# Notebooks: van gelabelde frames tot minimapvideo

Volgorde:

| # | Bestand | Wat | Kernel |
|---|---|---|---|
| 1 | [01_split_images.ipynb](01_split_images.ipynb) | Roboflow-export → `dataset/` (train/val/test op tijd), met de labelkeuzes | `venv` |
| 2 | [02_model.ipynb](02_model.ipynb) | YOLO26l trainen op 1920 px (max. 100 epochs, early stopping na 15), evalueren, video, OpenVINO-export | `venv-xpu` |
| 3 | [minimap/01_keypoints.md](minimap/01_keypoints.md) | veldpunten aanklikken voor de homografie (`click_keypoints.py`) en controleren (`check_homography.py`) | `venv` |
| 4 | [minimap/02_minimap_video.ipynb](minimap/02_minimap_video.ipynb) | detectie → tracking → bal → balbezit → video met minimap | `venv` |

## Omgevingen

- **`venv`**: alles behalve het trainen (CPU-PyTorch, OpenVINO voor de Intel Arc GPU bij inference).
- **`venv-xpu`**: PyTorch met Intel Arc-ondersteuning (XPU), alleen voor `02_model.ipynb`. Aangemaakt met:
  ```powershell
  C:\Python313\python -m venv venv-xpu
  venv-xpu\Scripts\python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/xpu
  venv-xpu\Scripts\python -m pip install ultralytics ipykernel openvino lap
  ```
  Twee voorwaarden, allebei getest op deze laptop (Core Ultra 7 155H, Intel Arc iGPU):
  - **een recente Intel-grafische driver** (getest: 32.0.101.9150). Met 31.0.101.5592 (juni 2024) crasht PyTorch
    zodra het de GPU aanspreekt (segmentation fault). OpenVINO werkt wel met die oude driver;
  - **PyTorch-XPU 2.8.0**, niet de nieuwste: met 2.15 crasht de Intel Level Zero-runtime zodra ultralytics traint of
    voorspelt (`UR_RESULT_ERROR_OUT_OF_RESOURCES`), ook met het kleinste model.

  Gemeten: yolo26l op 1920 px, batch 2 → ±10 min per epoch (±4 min trainen, ±6 min validatie), piekgeheugen
  17,3 van 17,8 GB. 100 epochs ≈ 17 u; met early stopping (patience 15) waarschijnlijk ±9–10 u.

In VS Code: open het notebook → rechtsboven *Select Kernel* → *Python Environments* → kies `venv` of `venv-xpu`.

## Gegenereerd (niet gecommit)

`dataset/`, `runs/`, `minimap/output/` en de gedownloade `*.pt`-gewichten worden opnieuw gemaakt door de notebooks.
`minimap/keypoints.json` en `experiments.csv` worden wel gecommit.
