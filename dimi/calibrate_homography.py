"""
Interactive 4-Point Homography Calibration Tool for Dimi.
Click 4 known field points on the video frame to compute matrix H.

Usage:
    python dimi/calibrate_homography.py
Controls:
    - Left Click: Select point (1 -> 2 -> 3 -> 4)
    - Key 'r': Reset all points
    - Key 's': Save calibration and compute matrix H
    - Key 'q' or 'ESC': Quit without saving
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path

DIMI_DIR = Path(__file__).resolve().parent
REPO_ROOT = DIMI_DIR.parent
VIDEO_PATH = REPO_ROOT / "data" / "raw_video" / "CV-video.mp4"
PITCH_IMG_PATH = REPO_ROOT / "data" / "pitch_template" / "tactical_pitch.png"
PITCH_JSON_PATH = REPO_ROOT / "data" / "pitch_template" / "tactical_pitch_keypoints.json"
CONFIG_OUTPUT = DIMI_DIR / "homography_config.json"

# Standaard veld-bestemmingen op de 2D kaart (1050 x 680 px FIFA template)
# Breedte = 1050, Hoogte = 680
# 16m lijn links = x=178, y_top=149, y_bot=530; Middenlijn = x=525, y_top=20, y_bot=660
# Penaltystip links = [125, 340]; Middenstip = [525, 340]
DEFAULT_TARGET_POINTS = [
    {"name": "16m lijn linksboven", "coord_2d": [178, 149]},
    {"name": "16m lijn linksonder", "coord_2d": [178, 530]},
    {"name": "Middellijn boven",     "coord_2d": [525, 20]},
    {"name": "Middellijn onder",     "coord_2d": [525, 660]},
]

clicked_points = []
preview_img = None
original_img = None

def mouse_callback(event, x, y, flags, param):
    global clicked_points, preview_img, original_img
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([float(x), float(y)])
            redraw()

def redraw():
    global preview_img, original_img, clicked_points
    preview_img = original_img.copy()

    # Teken hulplijnen en punten
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    
    for idx, pt in enumerate(clicked_points):
        pt_int = (int(pt[0]), int(pt[1]))
        color = colors[idx]
        target_name = DEFAULT_TARGET_POINTS[idx]["name"]

        # Teken cirkel en dradenkruis
        cv2.circle(preview_img, pt_int, 7, color, -1)
        cv2.circle(preview_img, pt_int, 14, color, 2)
        cv2.line(preview_img, (pt_int[0] - 20, pt_int[1]), (pt_int[0] + 20, pt_int[1]), color, 1)
        cv2.line(preview_img, (pt_int[0], pt_int[1] - 20), (pt_int[0], pt_int[1] + 20), color, 1)

        # Labeltekst
        text = f"P{idx+1}: {target_name} ({pt_int[0]}, {pt_int[1]})"
        cv2.putText(preview_img, text, (pt_int[0] + 15, pt_int[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(preview_img, text, (pt_int[0] + 15, pt_int[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Als 4 punten gekozen zijn, verbind ze met een polygoon
    if len(clicked_points) == 4:
        poly_pts = np.array(clicked_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(preview_img, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=2)
        status_text = "[KLAAR] 4 punten aangeduid! Druk op 's' om op te slaan, of 'r' om te resetten."
        cv2.putText(preview_img, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        next_target = DEFAULT_TARGET_POINTS[len(clicked_points)]["name"]
        status_text = f"Klik op punt P{len(clicked_points)+1}: [{next_target}] (of druk 'r' voor reset)"
        cv2.putText(preview_img, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Homografie Calibratie - Dimi", preview_img)


def main():
    global original_img, preview_img, clicked_points

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"[FOUT] Kon video niet openen: {VIDEO_PATH}")
        return

    original_img = frame
    preview_img = frame.copy()

    # Als er al een bestaande configuratie is, laad die in als start
    if CONFIG_OUTPUT.exists():
        try:
            with open(CONFIG_OUTPUT, "r") as f:
                saved = json.load(f)
                clicked_points = saved.get("camera_points", [])
                print(f"[INFO] Bestaande {len(clicked_points)} punten geladen uit {CONFIG_OUTPUT}")
        except Exception:
            pass

    cv2.namedWindow("Homografie Calibratie - Dimi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Homografie Calibratie - Dimi", 1280, 720)
    cv2.setMouseCallback("Homografie Calibratie - Dimi", mouse_callback)

    redraw()

    print("=" * 65)
    print("🎯 HOMOGRAFIE CALIBRATIE TOOL")
    print("=" * 65)
    print("Instructies:")
    print("1. Klik achtereenvolgens op de 4 bekende veldpunten:")
    for i, t in enumerate(DEFAULT_TARGET_POINTS):
        print(f"   P{i+1}: {t['name']}")
    print("2. Druk op de toets 's' om de Homografie Matrix H op te slaan.")
    print("3. Druk op 'r' om opnieuw te beginnen.")
    print("4. Druk op 'q' of ESC om af te sluiten.")
    print("=" * 65)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            print("[INFO] Afgesloten zonder opslaan.")
            break
        elif key == ord('r'):
            clicked_points = []
            print("[RESET] Punten gewist.")
            redraw()
        elif key == ord('s'):
            if len(clicked_points) < 4:
                print(f"[WAARSCHUWING] Slechts {len(clicked_points)}/4 punten geselecteerd. Klik nog {4 - len(clicked_points)} punten!")
                continue

            pts_src = np.array(clicked_points[:4], dtype=np.float32)
            pts_dst = np.array([t["coord_2d"] for t in DEFAULT_TARGET_POINTS], dtype=np.float32)

            # Bereken Homografie matrix H
            H, status = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC)

            config = {
                "description": "Homografie transformatie van 1080p camerabeeld naar 1050x680 2D FIFA minimap",
                "camera_points": clicked_points[:4],
                "minimap_points": [t["coord_2d"] for t in DEFAULT_TARGET_POINTS],
                "point_labels": [t["name"] for t in DEFAULT_TARGET_POINTS],
                "homography_matrix": H.tolist() if H is not None else None,
            }

            with open(CONFIG_OUTPUT, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)

            print("=" * 65)
            print("[SUCCESS] Homografie Matrix H succesvol berekend en opgeslagen!")
            print(f"[CONFIG] Opgeslagen in: {CONFIG_OUTPUT}")
            print(f"[MATRIX H]:\n{H}")
            print("=" * 65)
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
