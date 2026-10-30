"""
Interactieve Homografie Calibratie Tool (Point-Pair GUI).
Auteur: Dmitrij Tchervonisjenko (Team: Git Push and Pray)

Gebruik:
    python dimi/calibrate_homography.py

Instructies:
    1. Klik op een bekend punt op het CAMERABEELD (links) -> bijv. middellijn, stip, doelpaal, 16m-hoek.
    2. Klik op het CORRESPONDERENDE punt op het 2D VELD (rechts).
    3. Herhaal dit voor zoveel punten als je wilt (minimaal 4, aanbevolen: 6 tot 12 punten).
    4. Zodra je >= 4 punten hebt, toont de tool LIVE de geprojecteerde veldlijnen over het gras!
    5. Druk op 's' om op te slaan in dimi/homography_config.json.

Toetsen:
    - Linker Muisklik : Punt aanduiden (eerst links op camera, dan rechts op veld)
    - 'z'              : Laatste punt ongedaan maken (Undo)
    - 'r'              : Alle punten wissen (Reset)
    - 'f'              : 30 frames vooruit in video (voor een ander frame)
    - 'b'              : 30 frames achteruit in video
    - 's'              : Opslaan naar homography_config.json
    - 'q' of ESC       : Afsluiten
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
CONFIG_OUTPUT = DIMI_DIR / "homography_config.json"

# Doelresoluties van de bronbestanden
CAM_ORIG_W = 1920
CAM_ORIG_H = 1080
PITCH_ORIG_W = 1050
PITCH_ORIG_H = 680

# Weergavehoogte in venster
DISP_H = 600
CAM_DISP_W = int(CAM_ORIG_W * (DISP_H / CAM_ORIG_H))      # ~1066
PITCH_DISP_W = int(PITCH_ORIG_W * (DISP_H / PITCH_ORIG_H)) # ~926
TOTAL_W = CAM_DISP_W + PITCH_DISP_W

# Globale toestand
cap = None
current_frame_idx = 0
cam_frame_orig = None
pitch_orig = None

# Geklikte paren in originele resoluties:
# cam_points: [(x_cam, y_cam), ...] (in 1920x1080)
# pitch_points: [(x_pitch, y_pitch), ...] (in 1050x680)
cam_points = []
pitch_points = []
pending_cam_pt = None  # Wacht op bijbehorend klik op het 2D veld

window_name = "Homografie Calibratie Tool (Team: Git Push and Pray)"


def load_frame(idx):
    global cap, cam_frame_orig, current_frame_idx
    if cap is None:
        cap = cv2.VideoCapture(str(VIDEO_PATH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        cam_frame_orig = frame
        current_frame_idx = idx
    return ret


def mouse_callback(event, x, y, flags, param):
    global pending_cam_pt, cam_points, pitch_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if x < CAM_DISP_W:
            # Klik op Camera (links)
            orig_x = float(x) * (CAM_ORIG_W / CAM_DISP_W)
            orig_y = float(y) * (CAM_ORIG_H / DISP_H)
            pending_cam_pt = (orig_x, orig_y)
            print(f"[KLIK CAMERA] Punt {len(cam_points)+1}: Cam=({orig_x:.1f}, {orig_y:.1f}) -> Klik nu op het 2D veld (rechts)!")
        else:
            # Klik op 2D Veld (rechts)
            if pending_cam_pt is not None:
                pitch_x = float(x - CAM_DISP_W) * (PITCH_ORIG_W / PITCH_DISP_W)
                pitch_y = float(y) * (PITCH_ORIG_H / DISP_H)
                cam_points.append(pending_cam_pt)
                pitch_points.append((pitch_x, pitch_y))
                pending_cam_pt = None
                pair_idx = len(cam_points)
                print(f"[KLIK VELD] Punt {pair_idx}: Pitch=({pitch_x:.1f}, {pitch_y:.1f}) | Paar {pair_idx} compleet!")
                if len(cam_points) >= 4:
                    eval_current_homography()
            else:
                print("[WAARSCHUWING] Klik eerst op een punt in het camerabeeld (links) voordat je op het veld klikt!")


def eval_current_homography():
    if len(cam_points) < 4:
        return None
    pts_cam = np.array(cam_points, dtype=np.float32)
    pts_pitch = np.array(pitch_points, dtype=np.float32)
    H, _ = cv2.findHomography(pts_cam, pts_pitch, 0)
    if H is not None:
        pts_h = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
        proj = (H @ pts_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        errors = np.linalg.norm(proj - pts_pitch, axis=1)
        mean_err = np.mean(errors)
        print(f"[HOMOGRAFIE] {len(cam_points)} paren actief | Gemiddelde projectiefout: {mean_err:.2f} pixels")
        return H
    return None


def render_display():
    global cam_frame_orig, pitch_orig, pending_cam_pt, cam_points, pitch_points

    cam_vis = cam_frame_orig.copy()
    pitch_vis = pitch_orig.copy()

    # Bereken live homografie en projecteer veldlijnen als >= 4 punten
    H = None
    if len(cam_points) >= 4:
        pts_cam = np.array(cam_points, dtype=np.float32)
        pts_pitch = np.array(pitch_points, dtype=np.float32)
        H, _ = cv2.findHomography(pts_cam, pts_pitch, 0)
        if H is not None:
            try:
                H_inv = np.linalg.inv(H)
                # Teken belangrijke veldlijnen over de camera in cyaan/groen
                lines = [
                    ([525, 20], [525, 658], (0, 255, 0)),        # Middellijn
                    ([20, 20], [1030, 20], (255, 255, 0)),       # Bovenste zijlijn
                    ([20, 658], [1030, 658], (255, 255, 0)),     # Onderste zijlijn
                    ([1030, 20], [1030, 658], (0, 165, 255)),    # Rechter doellijn
                    ([872, 150], [1030, 150], (0, 255, 255)),    # 16m boven
                    ([872, 529], [1030, 529], (0, 255, 255)),    # 16m onder
                    ([872, 150], [872, 529], (0, 255, 255)),     # 16m front
                    ([978, 254], [1030, 254], (0, 200, 255)),    # 5m boven
                    ([978, 426], [1030, 426], (0, 200, 255)),    # 5m onder
                    ([978, 254], [978, 426], (0, 200, 255)),     # 5m front
                ]
                for p1, p2, col in lines:
                    xs = np.linspace(p1[0], p2[0], 25)
                    ys = np.linspace(p1[1], p2[1], 25)
                    pts = np.stack([xs, ys, np.ones_like(xs)], axis=0)
                    proj_cam = H_inv @ pts
                    proj_cam /= proj_cam[2:3, :]
                    pts_draw = np.int32(proj_cam[:2, :].T)
                    # Filter redelijke coördinaten
                    valid = (pts_draw[:, 0] >= -200) & (pts_draw[:, 0] <= 2120) & (pts_draw[:, 1] >= -200) & (pts_draw[:, 1] <= 1280)
                    if np.sum(valid) > 2:
                        cv2.polylines(cam_vis, [pts_draw[valid]], isClosed=False, color=col, thickness=2, lineType=cv2.LINE_AA)
            except Exception:
                pass

    # Teken geklikte paren
    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (0, 165, 255), (128, 0, 255),
        (255, 128, 0), (0, 255, 128), (255, 0, 128), (128, 255, 0),
        (0, 128, 255), (128, 0, 128), (0, 128, 128), (255, 255, 255)
    ]

    for idx, (c_pt, p_pt) in enumerate(zip(cam_points, pitch_points)):
        col = colors[idx % len(colors)]
        cx, cy = int(round(c_pt[0])), int(round(c_pt[1]))
        px, py = int(round(p_pt[0])), int(round(p_pt[1]))

        # Op camera
        cv2.circle(cam_vis, (cx, cy), 7, col, -1, cv2.LINE_AA)
        cv2.circle(cam_vis, (cx, cy), 13, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(cam_vis, f"P{idx+1}", (cx + 10, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(cam_vis, f"P{idx+1}", (cx + 10, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)

        # Op veld
        cv2.circle(pitch_vis, (px, py), 7, col, -1, cv2.LINE_AA)
        cv2.circle(pitch_vis, (px, py), 13, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(pitch_vis, f"P{idx+1}", (px + 10, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(pitch_vis, f"P{idx+1}", (px + 10, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)

    # Als er een punt op camera staat te wachten op het veld-punt:
    if pending_cam_pt is not None:
        cx, cy = int(round(pending_cam_pt[0])), int(round(pending_cam_pt[1]))
        cv2.circle(cam_vis, (cx, cy), 9, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(cam_vis, (cx, cy), 16, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(cam_vis, f"P{len(cam_points)+1}?", (cx + 12, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

    # Schaal naar venstergrootte
    cam_disp = cv2.resize(cam_vis, (CAM_DISP_W, DISP_H))
    pitch_disp = cv2.resize(pitch_vis, (PITCH_DISP_W, DISP_H))

    combined = np.hstack((cam_disp, pitch_disp))

    # Bovenste instructiebalk
    bar = np.zeros((65, TOTAL_W, 3), dtype=np.uint8)
    status_text = f"Punten aangeduid: {len(cam_points)} paren | Frame: {current_frame_idx}"
    if pending_cam_pt is not None:
        status_text += f" | >> KLIK NU OP HET 2D VELD (RECHTS) VOOR PUNT P{len(cam_points)+1} <<"
        bar_color = (0, 100, 200)
    elif len(cam_points) < 4:
        status_text += f" | Klik op een punt in het camerabeeld (links) [nog {4 - len(cam_points)} nodig]"
        bar_color = (30, 30, 30)
    else:
        status_text += " | >> GEREED! Druk op 's' om op te slaan! (of klik meer punten)"
        bar_color = (20, 120, 20)

    cv2.rectangle(bar, (0, 0), (TOTAL_W, 65), bar_color, -1)
    cv2.putText(bar, status_text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    help_keys = "Toetsen: 's'=Opslaan | 'z'=Undo | 'r'=Reset | 'f'=Volgend frame | 'b'=Vorig frame | 'q'=Afsluiten"
    cv2.putText(bar, help_keys, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    final_gui = np.vstack((bar, combined))
    return final_gui


def save_homography():
    if len(cam_points) < 4:
        print(f"[FOUT] Kan niet opslaan: Minimaal 4 paren vereist (huidig aantal: {len(cam_points)})!")
        return False

    pts_cam = np.array(cam_points, dtype=np.float32)
    pts_pitch = np.array(pitch_points, dtype=np.float32)

    H, status = cv2.findHomography(pts_cam, pts_pitch, 0)
    if H is None:
        print("[FOUT] Homografie kon niet berekend worden. Controleer of de punten niet collineair zijn.")
        return False

    pts_h = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
    proj = (H @ pts_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    errors = np.linalg.norm(proj - pts_pitch, axis=1)

    labels = [f"P{i+1}: Cam ({c[0]:.1f}, {c[1]:.1f}) -> Pitch ({p[0]:.1f}, {p[1]:.1f})" for i, (c, p) in enumerate(zip(cam_points, pitch_points))]

    config_data = {
        "description": f"Zelf Geijkte Homografie ({len(cam_points)} punten) door Dmitrij",
        "num_points": len(cam_points),
        "mean_reprojection_error_px": float(np.mean(errors)),
        "camera_points": cam_points,
        "minimap_points": pitch_points,
        "point_labels": labels,
        "homography_matrix": H.tolist()
    }

    with open(CONFIG_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)

    print("=" * 65)
    print(f"🎉 SUCCESVOL OPGESLAGEN!")
    print(f"Locatie: {CONFIG_OUTPUT}")
    print(f"Aantal referentieparen: {len(cam_points)}")
    print(f"Gemiddelde projectiefout: {np.mean(errors):.2f} pixels")
    print(f"Matrix H:\n{H}")
    print("=" * 65)
    print("Je kunt nu direct de videopipeline draaien:")
    print("    python dimi/minimap_pipeline.py")
    print("=" * 65)
    return True


def main():
    global pitch_orig, cam_points, pitch_points

    if not VIDEO_PATH.exists():
        print(f"[FOUT] Video niet gevonden in: {VIDEO_PATH}")
        return

    pitch_orig = cv2.imread(str(PITCH_IMG_PATH))
    if pitch_orig is None:
        print(f"[FOUT] Veld sjabloon niet gevonden in: {PITCH_IMG_PATH}")
        return

    load_frame(0)

    # Als er al een configuratie bestaat, vraag of de gebruiker die wil inladen als startpunt
    if CONFIG_OUTPUT.exists():
        try:
            with open(CONFIG_OUTPUT, "r", encoding="utf-8") as f:
                saved = json.load(f)
                saved_cam = saved.get("camera_points", [])
                saved_pitch = saved.get("minimap_points", [])
                if len(saved_cam) >= 4 and len(saved_cam) == len(saved_pitch):
                    cam_points = [tuple(p) for p in saved_cam]
                    pitch_points = [tuple(p) for p in saved_pitch]
                    print(f"[INFO] Bestaande {len(cam_points)} punten geladen als startpunt.")
                    print("       Druk op 'r' om blanco opnieuw te beginnen, of 'z' om punten te wissen.")
        except Exception:
            pass

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 750)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("=" * 65)
    print("⚽ INTERACTIEVE HOMOGRAFIE CALIBRATIE TOOL")
    print("=" * 65)
    print("Instructies:")
    print("1. Klik op een punt in het CAMERABEELD (links).")
    print("2. Klik op hetzelfde punt op het 2D VELD (rechts).")
    print("3. Herhaal voor minimaal 4 punten (bijv. middellijn, middenstip, doellijn, penaltystip).")
    print("4. Druk op 's' om de nieuwe homografie op te slaan!")
    print("=" * 65)

    while True:
        gui = render_display()
        cv2.imshow(window_name, gui)
        key = cv2.waitKey(25) & 0xFF

        if key == 27 or key == ord('q'):
            print("[INFO] Afgesloten.")
            break
        elif key == ord('z'):
            # Undo
            if pending_cam_pt is not None:
                pending_cam_pt = None
                print("[UNDO] Onvoltooid camera-punt geannuleerd.")
            elif len(cam_points) > 0:
                removed_c = cam_points.pop()
                removed_p = pitch_points.pop()
                print(f"[UNDO] Laatste paar {len(cam_points)+1} verwijderd.")
        elif key == ord('r'):
            cam_points = []
            pitch_points = []
            pending_cam_pt = None
            print("[RESET] Alle punten gewist. Je kunt nu blanco klikken.")
        elif key == ord('f'):
            # Volgend frame (+30)
            load_frame(current_frame_idx + 30)
            print(f"[FRAME] Vooruit gesprongen naar frame {current_frame_idx}")
        elif key == ord('b'):
            # Vorig frame (-30)
            load_frame(max(0, current_frame_idx - 30))
            print(f"[FRAME] Terug gesprongen naar frame {current_frame_idx}")
        elif key == ord('s'):
            if save_homography():
                # Toon succesmelding in console
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
