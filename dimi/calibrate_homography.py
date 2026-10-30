"""
Interactieve Begeleide Homografie Calibratie Tool (Guided Step-by-Step GUI).
Auteur: Dmitrij Tchervonisjenko (Team: Git Push and Pray)

BELANGRIJKE VERBETERINGEN:
1. GELEID MODEL: Je hoeft NIET meer te gokken of te klikken op het 2D veld!
   Het 2D veld toont automatisch een pulserend doelwit voor het gevraagde veldpunt.
   Je klikt ALLEEN MAAR op het camerabeeld (links)!
2. GEEN VERTIKALE OFFSET: De muiscoördinaten zijn 100% pixel-perfect gekalibreerd.
   Waar je klikt, daar verschijnt de cirkel exact onder je cursor.
3. LIVE OVERLAY: Zodra je >= 4 punten hebt, toont het camerabeeld de echte
   geprojecteerde veldlijnen in realtime.
4. TOETS 'a': Laad direct de wiskundig berekende voorinstelling!

Toetsen:
    - Linker Muisklik : Punt aanduiden op camerabeeld (links)
    - SPATIE of 'n'   : Huidig doelwit overslaan (Skip)
    - 'z'              : Laatste punt ongedaan maken (Undo)
    - 'r'              : Reset alle punten
    - 'a'              : Laad automatische referentie-voorinstelling (Auto)
    - 'f' / 'b'        : 30 frames vooruit / achteruit in de video
    - 's'              : Opslaan naar dimi/homography_config.json
    - 'q' of ESC       : Afsluiten
"""

import os
import json
import math
import cv2
import numpy as np
from pathlib import Path

DIMI_DIR = Path(__file__).resolve().parent
REPO_ROOT = DIMI_DIR.parent
VIDEO_PATH = REPO_ROOT / "data" / "raw_video" / "CV-video.mp4"
PITCH_IMG_PATH = REPO_ROOT / "data" / "pitch_template" / "tactical_pitch.png"
CONFIG_OUTPUT = DIMI_DIR / "homography_config.json"

# Bronresoluties
CAM_ORIG_W = 1920
CAM_ORIG_H = 1080
PITCH_ORIG_W = 1050
PITCH_ORIG_H = 680

# Weergave in venster
DISP_H = 540
CAM_DISP_W = 960   # 1920 * 0.5
PITCH_DISP_W = int(PITCH_ORIG_W * (DISP_H / PITCH_ORIG_H)) # 834
TOTAL_W = CAM_DISP_W + PITCH_DISP_W
BAR_H = 60

# Officiële doellijst voor stap-voor-stap kalibratie (Ground Truth FIFA 1050x680)
GUIDED_TARGETS = [
    {
        "id": "center_spot",
        "name": "1. Middenstip",
        "desc": "Klik op de witte stip precies in het midden van het veld",
        "coord_pitch": [525.0, 340.0],
    },
    {
        "id": "mid_top",
        "name": "2. Middellijn Boven",
        "desc": "Kruising van de middellijn met de BOVENSTE zijlijn",
        "coord_pitch": [525.0, 20.0],
    },
    {
        "id": "mid_bottom",
        "name": "3. Middellijn Onder",
        "desc": "Kruising van de middellijn met de ONDERSTE zijlijn",
        "coord_pitch": [525.0, 660.0],
    },
    {
        "id": "pen_spot",
        "name": "4. Rechter Penaltystip (11m)",
        "desc": "De witte 11-meterstip voor het rechterdoel",
        "coord_pitch": [925.0, 340.0],
    },
    {
        "id": "box16_top",
        "name": "5. Rechter 16m Hoek Boven",
        "desc": "Bovenste hoek van het grote strafschopgebied (16m-lijn)",
        "coord_pitch": [872.0, 150.0],
    },
    {
        "id": "box16_bottom",
        "name": "6. Rechter 16m Hoek Onder",
        "desc": "Onderste hoek van het grote strafschopgebied (16m-lijn)",
        "coord_pitch": [872.0, 529.0],
    },
    {
        "id": "goal_post_top",
        "name": "7. Rechter Doelpaal Boven",
        "desc": "Waar de bovenste doelpaal het gras raakt op de doellijn",
        "coord_pitch": [1030.0, 303.4],
    },
    {
        "id": "goal_post_bottom",
        "name": "8. Rechter Doelpaal Onder",
        "desc": "Waar de onderste doelpaal het gras raakt op de doellijn",
        "coord_pitch": [1030.0, 376.6],
    },
    {
        "id": "circle_top",
        "name": "9. Middencirkel Boven",
        "desc": "Snijpunt van de middellijn met de bovenrand van de middencirkel",
        "coord_pitch": [525.0, 248.5],
    },
    {
        "id": "circle_bottom",
        "name": "10. Middencirkel Onder",
        "desc": "Snijpunt van de middellijn met de onderrand van de middencirkel",
        "coord_pitch": [525.0, 431.5],
    },
]

# Voorinstelling berekend op basis van de broadcast camera geometrie
PRESET_CALIBRATION = {
    "center_spot": [513.0, 550.0],
    "mid_top": [487.1, 205.7],
    "mid_bottom": [548.3, 997.7],
    "pen_spot": [1565.0, 402.0],
    "box16_top": [1378.0, 240.0],
    "box16_bottom": [1519.0, 630.0],
    "goal_post_top": [1740.0, 330.0],
    "goal_post_bottom": [1840.0, 381.0],
    "circle_top": [505.0, 450.0],
    "circle_bottom": [522.0, 670.0],
}

# Globale toestand
cap = None
current_frame_idx = 0
cam_frame_orig = None
pitch_orig = None

# Geklikte punten mapping: { target_id: [cam_x, cam_y] }
calibrated_points = {}
current_target_idx = 0
mouse_x, mouse_y = -1, -1
animation_tick = 0

window_name = "Begeleide Homografie Calibrator - Team Git Push and Pray"


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
    global mouse_x, mouse_y, current_target_idx, calibrated_points

    mouse_x, mouse_y = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        # Controleer of klik binnen het camerabeeld valt (onder de menubalk en links)
        if y >= BAR_H and y < (BAR_H + DISP_H) and x < CAM_DISP_W:
            img_y = y - BAR_H
            orig_cam_x = float(x) * (CAM_ORIG_W / CAM_DISP_W)
            orig_cam_y = float(img_y) * (CAM_ORIG_H / DISP_H)

            if current_target_idx < len(GUIDED_TARGETS):
                tgt = GUIDED_TARGETS[current_target_idx]
                calibrated_points[tgt["id"]] = [orig_cam_x, orig_cam_y]
                print(f"[AANGEDUID] {tgt['name']} vastgelegd op Cam=({orig_cam_x:.1f}, {orig_cam_y:.1f})")

                # Ga naar volgend doelwit
                current_target_idx += 1
                if current_target_idx >= len(GUIDED_TARGETS):
                    print("Alle doelwitten doorlopen! Druk op 's' om op te slaan.")


def compute_current_homography():
    pts_cam = []
    pts_pitch = []
    for tgt in GUIDED_TARGETS:
        tid = tgt["id"]
        if tid in calibrated_points:
            pts_cam.append(calibrated_points[tid])
            pts_pitch.append(tgt["coord_pitch"])

    if len(pts_cam) < 4:
        return None, 0.0

    pts_cam = np.array(pts_cam, dtype=np.float32)
    pts_pitch = np.array(pts_pitch, dtype=np.float32)

    H, _ = cv2.findHomography(pts_cam, pts_pitch, 0)
    if H is None:
        return None, 0.0

    # Bereken reprojection error
    pts_h = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
    proj = (H @ pts_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    errors = np.linalg.norm(proj - pts_pitch, axis=1)
    mean_err = float(np.mean(errors))
    return H, mean_err


def render_display():
    global animation_tick
    animation_tick += 1

    cam_vis = cam_frame_orig.copy()
    pitch_vis = pitch_orig.copy()

    # Bereken live homografie als er >= 4 punten zijn
    H, mean_err = compute_current_homography()
    if H is not None:
        try:
            H_inv = np.linalg.inv(H)
            lines_2d = [
                ([525, 20], [525, 660], (0, 255, 0)),      # Middellijn
                ([525, 20], [1030, 20], (255, 255, 0)),    # Bovenste zijlijn
                ([525, 660], [1030, 660], (255, 255, 0)),  # Onderste zijlijn
                ([1030, 20], [1030, 660], (0, 165, 255)),  # Rechter doellijn
                ([872, 150], [1030, 150], (0, 255, 255)),  # 16m boven
                ([872, 529], [1030, 529], (0, 255, 255)),  # 16m onder
                ([872, 150], [872, 529], (0, 255, 255)),   # 16m front
                ([978, 254], [1030, 254], (0, 200, 255)),  # 5m boven
                ([978, 426], [1030, 426], (0, 200, 255)),  # 5m onder
                ([978, 254], [978, 426], (0, 200, 255)),   # 5m front
            ]
            for p1, p2, col in lines_2d:
                xs = np.linspace(p1[0], p2[0], 35)
                ys = np.linspace(p1[1], p2[1], 35)
                pts = np.stack([xs, ys, np.ones_like(xs)], axis=0)
                proj = H_inv @ pts
                proj /= proj[2:3, :]
                pts_d = np.int32(proj[:2, :].T)
                valid = (pts_d[:, 0] >= -100) & (pts_d[:, 0] < 2020) & (pts_d[:, 1] >= -100) & (pts_d[:, 1] < 1180)
                if np.sum(valid) > 2:
                    cv2.polylines(cam_vis, [pts_d[valid]], isClosed=False, color=col, thickness=2, lineType=cv2.LINE_AA)

            # Middencirkel projectie
            angles = np.linspace(0, 2 * np.pi, 60)
            xs = 525.0 + 91.5 * np.cos(angles)
            ys = 340.0 + 91.5 * np.sin(angles)
            pts = np.stack([xs, ys, np.ones_like(xs)], axis=0)
            proj = H_inv @ pts
            proj /= proj[2:3, :]
            pts_d = np.int32(proj[:2, :].T)
            valid = (pts_d[:, 0] >= -100) & (pts_d[:, 0] < 2020) & (pts_d[:, 1] >= -100) & (pts_d[:, 1] < 1180)
            if np.sum(valid) > 4:
                cv2.polylines(cam_vis, [pts_d[valid]], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        except Exception:
            pass

    # Teken alle reeds gekalibreerde punten
    for idx, tgt in enumerate(GUIDED_TARGETS):
        tid = tgt["id"]
        px, py = int(tgt["coord_pitch"][0]), int(tgt["coord_pitch"][1])

        if tid in calibrated_points:
            cx, cy = int(calibrated_points[tid][0]), int(calibrated_points[tid][1])
            # Groen vinkje / cirkel op camera
            cv2.circle(cam_vis, (cx, cy), 7, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(cam_vis, (cx, cy), 13, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(cam_vis, f"{idx+1}", (cx + 10, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(cam_vis, f"{idx+1}", (cx + 10, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

            # Groen op 2D veld
            cv2.circle(pitch_vis, (px, py), 7, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(pitch_vis, (px, py), 13, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(pitch_vis, f"{idx+1}", (px + 10, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(pitch_vis, f"{idx+1}", (px + 10, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # Neutraal grijs klein stipje op 2D veld
            cv2.circle(pitch_vis, (px, py), 4, (120, 120, 120), -1, cv2.LINE_AA)

    # Animeer het ACTIEVE doelwit op het 2D veld (pulserende gele bullseye)
    active_target = None
    if current_target_idx < len(GUIDED_TARGETS):
        active_target = GUIDED_TARGETS[current_target_idx]
        tpx, tpy = int(active_target["coord_pitch"][0]), int(active_target["coord_pitch"][1])

        pulse_r = int(14 + 6 * math.sin(animation_tick * 0.25))
        # Gele pulserende cirkel
        cv2.circle(pitch_vis, (tpx, tpy), pulse_r, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(pitch_vis, (tpx, tpy), 5, (0, 255, 255), -1, cv2.LINE_AA)
        # Dradenkruis
        cv2.line(pitch_vis, (tpx - 25, tpy), (tpx + 25, tpy), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(pitch_vis, (tpx, tpy - 25), (tpx, tpy + 25), (0, 255, 255), 1, cv2.LINE_AA)
        # Label
        lbl = f"HIER: {active_target['name']}"
        cv2.putText(pitch_vis, lbl, (tpx - 50, tpy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(pitch_vis, lbl, (tpx - 50, tpy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    # Schaal beide schermen exact naar vensterafmetingen
    cam_disp = cv2.resize(cam_vis, (CAM_DISP_W, DISP_H))
    pitch_disp = cv2.resize(pitch_vis, (PITCH_DISP_W, DISP_H))

    # VOEG EEN VERGROOTGLAS TOE AAN DE CURSOR OP HET CAMERABEELD
    if mouse_y >= BAR_H and mouse_y < (BAR_H + DISP_H) and mouse_x < CAM_DISP_W:
        # Cursor bevindt zich op het camerabeeld
        img_y = mouse_y - BAR_H
        cx = int(float(mouse_x) * (CAM_ORIG_W / CAM_DISP_W))
        cy = int(float(img_y) * (CAM_ORIG_H / DISP_H))

        # Vergrootglas in de rechterbovenhoek van het camerabeeld (200x200 px)
        loupe_w, loupe_h = 180, 180
        zoom_radius = 45
        min_x, max_x = max(0, cx - zoom_radius), min(CAM_ORIG_W, cx + zoom_radius)
        min_y, max_y = max(0, cy - zoom_radius), min(CAM_ORIG_H, cy + zoom_radius)

        crop = cam_frame_orig[min_y:max_y, min_x:max_x]
        if crop.size > 0:
            loupe = cv2.resize(crop, (loupe_w, loupe_h), interpolation=cv2.INTER_LINEAR)
            # Dradenkruis in vergrootglas
            mid_lx, mid_ly = loupe_w // 2, loupe_h // 2
            cv2.line(loupe, (mid_lx - 15, mid_ly), (mid_lx + 15, mid_ly), (0, 0, 255), 2)
            cv2.line(loupe, (mid_lx, mid_ly - 15), (mid_lx, mid_ly + 15), (0, 0, 255), 2)
            cv2.rectangle(loupe, (0, 0), (loupe_w - 1, loupe_h - 1), (0, 255, 255), 2)
            cv2.putText(loupe, f"Zoom 3x ({cx}, {cy})", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # Plaats loupe in rechterbovenhoek van cam_disp
            cam_disp[10:10 + loupe_h, CAM_DISP_W - loupe_w - 10:CAM_DISP_W - 10] = loupe

    combined = np.hstack((cam_disp, pitch_disp))

    # Menubalk bovenaan (exact BAR_H = 60 px)
    bar = np.zeros((BAR_H, TOTAL_W, 3), dtype=np.uint8)

    num_done = len(calibrated_points)
    if active_target is not None:
        title = f"👉 STAP {current_target_idx + 1}/{len(GUIDED_TARGETS)}: Klik op '{active_target['name']}' in de video (LINKS)"
        sub = f"Uitleg: {active_target['desc']}  |  [Spatie]=Overslaan  [z]=Undo  [a]=Auto-Preset  [s]=Opslaan"
        bar_bg = (20, 40, 120)
    else:
        title = f"🎉 KLAAR! Alle {num_done} punten vastgelegd!  |  Druk op 's' om op te slaan!"
        sub = f"Foutmarge: {mean_err:.2f} px  |  [s]=Opslaan  [z]=Undo  [r]=Reset  [a]=Auto-Preset"
        bar_bg = (20, 120, 20)

    cv2.rectangle(bar, (0, 0), (TOTAL_W, BAR_H), bar_bg, -1)
    cv2.putText(bar, title, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bar, sub, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 220, 255), 1, cv2.LINE_AA)

    final_gui = np.vstack((bar, combined))
    return final_gui


def save_homography():
    H, mean_err = compute_current_homography()
    if H is None:
        print("[FOUT] Minimaal 4 punten vereist om een homografie op te slaan!")
        return False

    cam_pts = []
    pitch_pts = []
    labels = []
    for tgt in GUIDED_TARGETS:
        tid = tgt["id"]
        if tid in calibrated_points:
            c = calibrated_points[tid]
            p = tgt["coord_pitch"]
            cam_pts.append(c)
            pitch_pts.append(p)
            labels.append(f"{tgt['name']} -> Cam: ({c[0]:.1f}, {c[1]:.1f}) | Pitch: ({p[0]:.1f}, {p[1]:.1f})")

    config_data = {
        "description": f"Geijkte Veldlijnen Homografie ({len(cam_pts)} punten) door Dmitrij",
        "num_points": len(cam_pts),
        "mean_reprojection_error_px": float(mean_err),
        "camera_points": cam_pts,
        "minimap_points": pitch_pts,
        "point_labels": labels,
        "homography_matrix": H.tolist()
    }

    with open(CONFIG_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)

    print("=" * 65)
    print("🎉 SUCCESVOL OPGESLAGEN IN HOMOGRAPHY_CONFIG.JSON!")
    print(f"Aantal punten: {len(cam_pts)}")
    print(f"Gemiddelde foutmarge: {mean_err:.2f} pixels")
    print(f"Matrix H:\n{H}")
    print("=" * 65)
    print("Voer nu uit:")
    print("    .\\venv\\Scripts\\python.exe dimi/minimap_pipeline.py")
    print("=" * 65)
    return True


def load_preset():
    global calibrated_points, current_target_idx
    calibrated_points = {}
    for tid, cam_pt in PRESET_CALIBRATION.items():
        calibrated_points[tid] = list(cam_pt)
    current_target_idx = len(GUIDED_TARGETS)
    print("=" * 65)
    print("[PRESET] Berekende referentie-voorinstelling succesvol geladen!")
    print("Je ziet nu alle groene veldlijnen live over het cameragras.")
    print("Druk op 's' om deze direct op te slaan, of 'z' om aan te passen.")
    print("=" * 65)


def main():
    global pitch_orig, current_target_idx, calibrated_points

    if not VIDEO_PATH.exists():
        print(f"[FOUT] Video niet gevonden: {VIDEO_PATH}")
        return

    pitch_orig = cv2.imread(str(PITCH_IMG_PATH))
    if pitch_orig is None:
        print(f"[FOUT] Veldtemplate niet gevonden: {PITCH_IMG_PATH}")
        return

    load_frame(0)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("=" * 65)
    print("⚽ BEGELEIDE HOMOGRAFIE CALIBRATIE TOOL")
    print("=" * 65)
    print("Instructies:")
    print("1. Kijk naar het 2D veld (rechts): de gele pulserende stip toont het doelwit.")
    print("2. Klik ALLEEN op het camerabeeld (links) op datzelfde punt.")
    print("3. Druk op [SPATIE] om een doelwit over te slaan als het niet zichtbaar is.")
    print("4. Druk op 'a' om direct de automatische voorinstelling te laden.")
    print("5. Druk op 's' om op te slaan!")
    print("=" * 65)

    while True:
        gui = render_display()
        cv2.imshow(window_name, gui)
        key = cv2.waitKey(30) & 0xFF

        if key == 27 or key == ord('q'):
            print("[INFO] Afgesloten.")
            break
        elif key == ord(' '):
            # Skip current target
            if current_target_idx < len(GUIDED_TARGETS):
                tgt = GUIDED_TARGETS[current_target_idx]
                print(f"[SKIP] '{tgt['name']}' overgeslagen.")
                current_target_idx += 1
        elif key == ord('z'):
            # Undo
            if current_target_idx > 0:
                current_target_idx -= 1
                tgt = GUIDED_TARGETS[current_target_idx]
                if tgt["id"] in calibrated_points:
                    del calibrated_points[tgt["id"]]
                print(f"[UNDO] Terug naar '{tgt['name']}'.")
        elif key == ord('r'):
            # Reset
            calibrated_points = {}
            current_target_idx = 0
            print("[RESET] Alle punten gewist. We beginnen weer bij Stap 1.")
        elif key == ord('a'):
            # Load preset
            load_preset()
        elif key == ord('f'):
            # Next frame
            load_frame(current_frame_idx + 30)
            print(f"[FRAME] Naar frame {current_frame_idx}")
        elif key == ord('b'):
            # Prev frame
            load_frame(max(0, current_frame_idx - 30))
            print(f"[FRAME] Naar frame {current_frame_idx}")
        elif key == ord('s'):
            if save_homography():
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
