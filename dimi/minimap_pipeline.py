"""
Advanced Sports Vision & Tactical Mapping Pipeline (Super Code Edition).
Author: Dmitrij Tchervonisjenko (Team: Git Push and Pray)

Features:
  - High-Resolution Multi-Class Detection (1600p on RTX 4070 Laptop GPU)
  - Class-Specific Dual Confidence Thresholds (Ball: 0.12, Players: 0.35)
  - Official Team Colors (PSG Navy Blue vs. Inter Bright Yellow, Fluo Referee)
  - Kinematic Ball Tracker with Spatial Distance Gating (prevents false jumps to penalty spot)
  - Physics-based Outlier Rejection & Smooth Neon Trajectory Comet Trail
  - Live Tactical Radar with Planar Homography Projection (Camera -> 2D FIFA Pitch)
  - Real-time Match Telemetry: Live Ball Speed (km/h) & Tactical Ball Possession
  - Spatial Occupancy Heatmap (Gaussian KDE)
"""

import os
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

from common import VIDEO_PATH, BEST_WEIGHTS, RUNS_DIR, pick_device

DIMI_DIR = Path(__file__).resolve().parent
REPO_ROOT = DIMI_DIR.parent
PITCH_IMG_PATH = REPO_ROOT / "data" / "pitch_template" / "tactical_pitch.png"
CONFIG_PATH = DIMI_DIR / "homography_config.json"

# ==============================================================================
# OFFICIËLE TEAMKLEUREN & STIJL (BGR Formaat voor OpenCV)
# ==============================================================================
CLASS_CONFIG = {
    0: {
        "name": "Ball",
        "color_fill": (255, 255, 255),    # Zuiver wit
        "color_border": (0, 0, 255),      # Fel rood
        "conf": 0.12,                     # Hoge bal-recall threshold
        "radius": 8
    },
    1: {
        "name": "PSG (Team A)",
        "color_fill": (180, 50, 20),      # Royal Navy Blue
        "color_border": (255, 255, 255),  # Witte rand
        "conf": 0.35,
        "radius": 11
    },
    2: {
        "name": "Inter (Team B)",
        "color_fill": (0, 220, 255),      # Fel zonnegeel
        "color_border": (20, 20, 20),     # Zwarte rand
        "conf": 0.35,
        "radius": 11
    },
    3: {
        "name": "Keeper B",
        "color_fill": (0, 130, 255),      # Vlammend oranje
        "color_border": (255, 255, 255),  # Witte rand
        "conf": 0.30,
        "radius": 12
    },
    4: {
        "name": "Scheidsrechter",
        "color_fill": (30, 255, 140),     # Fluo gifgroen
        "color_border": (10, 10, 10),     # Zwarte rand
        "conf": 0.35,
        "radius": 10
    }
}


def load_homography_matrix():
    """Laadt de geijkte Homografie matrix uit homography_config.json."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                H = np.array(data["homography_matrix"], dtype=np.float32)
                print(f"[HOMOGRAFIE] Geijkte matrix H geladen ({len(data['camera_points'])} referentiepunten)")
                return H
        except Exception as e:
            print("[WARN] Fout bij inlezen homografie config:", e)

    raise FileNotFoundError("homography_config.json ontbreekt. Run eerst calibrate_homography.py!")


def transform_points(points, H, pitch_w=1050, pitch_h=680):
    """Projecteert cameracoordinaten (x, y) naar het 2D veld (X, Y)."""
    if len(points) == 0:
        return np.array([], dtype=np.float32)
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    # Clip netjes binnen veldgrenzen
    mapped[:, 0] = np.clip(mapped[:, 0], 5, pitch_w - 5)
    mapped[:, 1] = np.clip(mapped[:, 1], 5, pitch_h - 5)
    return mapped


class KinematicBallTracker:
    """
    Intelligente Ball Tracker met Kinematische Gating en Ruimtelijke Afstandsbegrenzing.
    Elimineert valse sprongen naar statische grasmarkeringen (penalty spot) en white cleats.
    """
    def __init__(self, max_trail_len=25, max_jump_cam=110.0, max_jump_2d=75.0):
        self.trail = []                 # Historie van (x, y) op 2D pitch
        self.last_pos_cam = None        # Laatst bevestigde camera positie
        self.last_pos_2d = None         # Laatst bevestigde 2D veld positie
        self.frames_lost = 0            # Aantal opeenvolgende frames zonder detectie
        self.max_trail_len = max_trail_len
        self.max_jump_cam = max_jump_cam
        self.max_jump_2d = max_jump_2d
        self.total_tracked_frames = 0
        self.speed_history = []

    def update(self, ball_candidates, H, pitch_w, pitch_h):
        """
        ball_candidates: list van dicts: [{'box': box, 'conf': score, 'center': (cx, cy)}]
        """
        chosen_cam = None

        if not ball_candidates:
            self.frames_lost += 1
            if self.frames_lost > 4 and self.trail:
                self.trail.pop(0)
            return self.get_display_pos()

        # Als we een recente voorgeschiedenis hebben: ruimtelijke gating toepassen
        if self.last_pos_cam is not None and self.frames_lost <= 6:
            valid = []
            for cand in ball_candidates:
                cx, cy = cand['center']
                dist_cam = np.hypot(cx - self.last_pos_cam[0], cy - self.last_pos_cam[1])
                # Toegestane sprong schaalt licht met gemiste frames (bal in vlucht)
                allowed_gate = self.max_jump_cam + (self.frames_lost * 50.0)
                if dist_cam <= allowed_gate:
                    # Score combinatie van nabijheid en confidence
                    score = cand['conf'] - (dist_cam / 500.0)
                    valid.append((score, cand))

            if valid:
                valid.sort(key=lambda x: x[0], reverse=True)
                chosen_cam = valid[0][1]['center']
            else:
                # Buiten gate: enkel accepteren bij hoge zekere confidence (> 0.40)
                high_conf = [c for c in ball_candidates if c['conf'] >= 0.40]
                if high_conf:
                    high_conf.sort(key=lambda x: x['conf'], reverse=True)
                    chosen_cam = high_conf[0]['center']
                else:
                    self.frames_lost += 1
                    return self.get_display_pos()
        else:
            # Geen voorgeschiedenis: kies kandidaat met hoogste confidence
            ball_candidates.sort(key=lambda x: x['conf'], reverse=True)
            chosen_cam = ball_candidates[0]['center']

        if chosen_cam is not None:
            # Projecteer naar 2D veld
            mapped = transform_points([chosen_cam], H, pitch_w, pitch_h)
            if len(mapped) > 0:
                p2d = (int(mapped[0][0]), int(mapped[0][1]))

                # 2D sprong controle
                if self.last_pos_2d is not None and self.frames_lost <= 2:
                    jump_2d = np.hypot(p2d[0] - self.last_pos_2d[0], p2d[1] - self.last_pos_2d[1])
                    if jump_2d > self.max_jump_2d:
                        # Onwaarschijnlijke teleportsprong, negeer
                        self.frames_lost += 1
                        return self.get_display_pos()

                # Bevestigde update
                self.last_pos_cam = chosen_cam
                self.last_pos_2d = p2d
                self.frames_lost = 0
                self.total_tracked_frames += 1
                self.trail.append(p2d)
                if len(self.trail) > self.max_trail_len:
                    self.trail.pop(0)

        return self.get_display_pos()

    def get_display_pos(self):
        # Behoud positie tot 3 frames bij occlusie
        if self.frames_lost <= 3:
            return self.last_pos_2d
        return None


def draw_tactical_radar(pitch_base, detected_players, H, ball_pos_2d, ball_trail):
    """Tekent de spelers en de bal met professionele radar-styling zonder foutieve lijnen."""
    radar = pitch_base.copy()
    pitch_h, pitch_w = radar.shape[:2]

    # 1. Teken balspoor (vloeiende neon komeet trail)
    if len(ball_trail) > 1:
        for i in range(1, len(ball_trail)):
            p_prev = ball_trail[i - 1]
            p_curr = ball_trail[i]
            # Max 65 px afstand tussen opeenvolgende frames voor aaneengesloten trail
            dist = np.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            if dist <= 65:
                progress = i / len(ball_trail)
                thickness = int(1 + progress * 3)
                color = (0, int(160 + 95 * progress), 255)
                cv2.line(radar, p_prev, p_curr, color, thickness, cv2.LINE_AA)

    # 2. Teken spelers per categorie
    for cid in [1, 2, 3, 4]:
        pts = detected_players.get(cid, [])
        if not pts:
            continue
        mapped = transform_points(pts, H, pitch_w, pitch_h)
        cfg = CLASS_CONFIG[cid]

        for pt in mapped:
            px, py = int(pt[0]), int(pt[1])
            # Schaduw onder speler voor 3D diepte-effect
            cv2.circle(radar, (px + 2, py + 2), cfg["radius"], (15, 15, 15), -1, cv2.LINE_AA)
            # Speler dot in officiële teamkleur
            cv2.circle(radar, (px, py), cfg["radius"], cfg["color_fill"], -1, cv2.LINE_AA)
            # Rand
            cv2.circle(radar, (px, py), cfg["radius"], cfg["color_border"], 2, cv2.LINE_AA)

    # 3. Teken de bal op de 2D positie
    if ball_pos_2d is not None:
        bx, by = int(ball_pos_2d[0]), int(ball_pos_2d[1])
        # Buitenste rode glow
        cv2.circle(radar, (bx, by), 12, (0, 0, 255), 2, cv2.LINE_AA)
        # Zuiver witte bal
        cv2.circle(radar, (bx, by), 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(radar, (bx, by), 8, (0, 0, 255), 1, cv2.LINE_AA)

    return radar


def draw_styled_camera_box(img, box, cid, score):
    """Tekent professionele bounding boxes op het camerabeeld."""
    cfg = CLASS_CONFIG.get(cid, CLASS_CONFIG[1])
    x1, y1, x2, y2 = [int(v) for v in box]

    color = cfg["color_fill"]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    label = f"{cfg['name']} {score:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
    
    text_color = (0, 0, 0) if cid in [2, 4] else (255, 255, 255)
    cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA)


def compute_match_telemetry(ball_pos_2d, last_ball_pos_2d, mapped_a, mapped_b, fps=30.0):
    """
    Berekent live balsnelheid (km/h) en tactisch balbezit (dichtstbijzijnde speler).
    Schaal: 1050 px = 105 meter -> 0.1 meter per pixel.
    """
    speed_kmh = 0.0
    if ball_pos_2d is not None and last_ball_pos_2d is not None:
        dist_px = np.hypot(ball_pos_2d[0] - last_ball_pos_2d[0], ball_pos_2d[1] - last_ball_pos_2d[1])
        raw_speed = (dist_px * 0.1) * fps * 3.6
        if raw_speed < 140.0:
            speed_kmh = raw_speed

    possession = "BETWIST"
    if ball_pos_2d is not None:
        min_a = min([np.hypot(ball_pos_2d[0] - p[0], ball_pos_2d[1] - p[1]) for p in mapped_a], default=9999.0)
        min_b = min([np.hypot(ball_pos_2d[0] - p[0], ball_pos_2d[1] - p[1]) for p in mapped_b], default=9999.0)

        if min_a < min_b and min_a < 150:
            possession = "PSG (Team A)"
        elif min_b < min_a and min_b < 150:
            possession = "INTER (Team B)"

    return speed_kmh, possession


def generate_heatmap(all_points, pitch_template, output_path):
    """Genereert een high-res Kernel Density Estimation (KDE) Heatmap."""
    h, w = pitch_template.shape[:2]
    density = np.zeros((h, w), dtype=np.float32)

    for pt in all_points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(density, (x, y), 35, 1.0, -1)

    density = cv2.GaussianBlur(density, (65, 65), 0)
    if density.max() > 0:
        density = density / density.max()

    heatmap_color = cv2.applyColorMap((density * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(pitch_template, 0.55, heatmap_color, 0.45, 0)
    cv2.imwrite(str(output_path), overlay)
    print(f"[EXTENSIE] Heatmap succesvol opgeslagen in: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Super-Code Tactical Mapping Pipeline")
    parser.add_argument("--weights", type=str, default=str(BEST_WEIGHTS))
    parser.add_argument("--video", type=str, default=str(VIDEO_PATH))
    parser.add_argument("--imgsz", type=int, default=1600, help="High-resolution inference")
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()

    device = pick_device()
    H = load_homography_matrix()

    pitch_img = cv2.imread(str(PITCH_IMG_PATH))
    if pitch_img is None:
        print("[ERROR] tactical_pitch.png niet gevonden!")
        return

    output_dir = RUNS_DIR / "tactical_radar_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = output_dir / "synced_tactical_radar.mp4"

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    target_h = 720
    cam_w = 1280
    minimap_w = int(pitch_img.shape[1] * (target_h / pitch_img.shape[0]))
    combined_w = cam_w + minimap_w

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_video_path), fourcc, fps, (combined_w, target_h))

    print("=" * 65)
    print(f"[SUPER-PIPELINE] Starten op {device} met imgsz={args.imgsz}")
    print(f"[OUTPUT] {out_video_path}")
    print("=" * 65)

    tracker = KinematicBallTracker(max_trail_len=25, max_jump_cam=110.0, max_jump_2d=75.0)
    prev_ball_pos_2d = None
    all_player_points = []
    frame_idx = 0

    while cap.isOpened() and (args.max_frames == 0 or frame_idx < args.max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=args.imgsz, conf=0.10, device=device, verbose=False)[0]

        detected_players = {1: [], 2: [], 3: [], 4: []}
        ball_candidates = []
        cam_display = frame.copy()

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            scores = results.boxes.conf.cpu().numpy()

            for box, cid, score in zip(boxes, classes, scores):
                if cid not in CLASS_CONFIG:
                    continue

                min_conf = CLASS_CONFIG[cid]["conf"]
                if score < min_conf:
                    continue

                x1, y1, x2, y2 = box
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                if cid == 0:
                    ball_candidates.append({
                        "box": box,
                        "conf": float(score),
                        "center": center
                    })
                else:
                    draw_styled_camera_box(cam_display, box, cid, score)
                    # Planar assumption: voeten van spelers
                    feet = [(x1 + x2) / 2.0, y2]
                    detected_players[cid].append(feet)

        # Update bal tracker met ruimtelijke gating
        ball_pos_2d = tracker.update(ball_candidates, H, pitch_img.shape[1], pitch_img.shape[0])

        # Teken bal op camerabeeld als die getrackt wordt
        if tracker.last_pos_cam is not None and tracker.frames_lost <= 2:
            bcx, bcy = int(tracker.last_pos_cam[0]), int(tracker.last_pos_cam[1])
            cv2.circle(cam_display, (bcx, bcy), 11, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(cam_display, (bcx, bcy), 7, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(cam_display, "BALL", (bcx - 18, bcy - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # Mappings voor balbezit en heatmap
        mapped_a = transform_points(detected_players[1], H, pitch_img.shape[1], pitch_img.shape[0])
        mapped_b = transform_points(detected_players[2], H, pitch_img.shape[1], pitch_img.shape[0])

        # Telemetrie berekenen
        speed_kmh, possession = compute_match_telemetry(ball_pos_2d, prev_ball_pos_2d, mapped_a, mapped_b, fps)
        prev_ball_pos_2d = ball_pos_2d

        # Verzamel posities voor Heatmap
        for pts_2d in [mapped_a, mapped_b]:
            for p in pts_2d:
                all_player_points.append(p)

        # -------------------------------------------------------------
        # Renderen & Compositing
        # -------------------------------------------------------------
        minimap_frame = draw_tactical_radar(pitch_img, detected_players, H, ball_pos_2d, tracker.trail)

        cam_resized = cv2.resize(cam_display, (cam_w, target_h))
        minimap_resized = cv2.resize(minimap_frame, (minimap_w, target_h))
        combined = np.hstack((cam_resized, minimap_resized))

        # Semi-transparante Tactical HUD Header
        overlay = combined.copy()
        cv2.rectangle(overlay, (0, 0), (combined_w, 55), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, combined, 0.15, 0, combined)

        # Header tekst met teamkleuren & live telemetrie
        cv2.putText(combined, "PSG (Team A)", (25, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 120, 40), 2, cv2.LINE_AA)
        cv2.putText(combined, f": {len(detected_players[1])}", (190, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(combined, "INTER (Team B)", (280, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, f": {len(detected_players[2])}", (465, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

        ball_status = "BAL IN SPEL" if tracker.frames_lost <= 2 else "BAL ZOEKEND"
        ball_status_color = (0, 255, 0) if "IN SPEL" in ball_status else (0, 165, 255)
        cv2.putText(combined, f"STATUS: {ball_status}", (550, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.68, ball_status_color, 2, cv2.LINE_AA)

        cv2.putText(combined, f"BEZIT: {possession}", (820, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        if speed_kmh > 1.0:
            cv2.putText(combined, f"SNELHEID: {speed_kmh:.1f} km/h", (1070, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(combined, "2D FIFA TACTICAL RADAR", (cam_w + 25, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        out.write(combined)
        frame_idx += 1
        if frame_idx % 60 == 0 or frame_idx == total_frames:
            print(f"[SUPER-PIPELINE] Frame {frame_idx}/{total_frames} (Bal getrackt in {tracker.total_tracked_frames} frames)")

    cap.release()
    out.release()

    # Heatmap genereren
    heatmap_out = output_dir / "spatial_occupancy_heatmap.png"
    generate_heatmap(all_player_points, pitch_img, heatmap_out)

    print("=" * 65)
    print(f"[SUCCESS] Super video gereed: {out_video_path}")
    print(f"[SUMMARY] Totaal frames: {frame_idx} | Bal getrackt: {tracker.total_tracked_frames} frames ({tracker.total_tracked_frames/max(1, frame_idx)*100:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
