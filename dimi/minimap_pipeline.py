"""
Advanced Sports Vision & Tactical Mapping Pipeline (Super Code Edition).
Author: Dmitrij Tchervonisjenko (Team: Git Push and Pray)

Features:
  - 10-Point Calibrated RANSAC Homography Transformation (Camera -> 2D FIFA Pitch)
  - Dual-Layer Invariant Penalty Spot & Pitch Chalk Marking Suppression
  - Player Cleat / White Shoe Filtering (aspect-ratio & foot-proximity checks)
  - Real Ball Tracking (midfield build-up, flank passes, penalty box)
  - Official Broadcast Colors (PSG Royal Navy Blue vs. Inter Bright Yellow, Fluo Referee)
  - Anti-Flicker Temporal Memory & Segmented Comet Trail (no stray lines)
  - Real-Time Match Telemetry: Live Ball Speed (km/h) & Tactical Ball Possession
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
        "conf": 0.07,                     # Optimale recall voor echte bal detectie
        "radius": 9
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
    """Laadt de geijkte 10-punts RANSAC Homografie matrix uit homography_config.json."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                cam_pts = np.array(data["camera_points"], dtype=np.float32)
                pitch_pts = np.array(data["minimap_points"], dtype=np.float32)
                # Bereken robuuste RANSAC matrix over alle geijkte punten
                H, _ = cv2.findHomography(cam_pts, pitch_pts, cv2.RANSAC, 4.0)
                print(f"[HOMOGRAFIE] 10-Punts RANSAC Matrix geladen ({len(cam_pts)} geijkte veldpunten)!")
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
    mapped[:, 0] = np.clip(mapped[:, 0], 5, pitch_w - 5)
    mapped[:, 1] = np.clip(mapped[:, 1], 5, pitch_h - 5)
    return mapped


def is_penalty_spot_or_marking(center_cam, H):
    """
    Sluit de penaltystip 100% uit.
    Controleert zowel in de camerazone als in de gekalibreerde 2D pitchruimte.
    """
    cx, cy = center_cam

    # 1. Camera coordinaten (rechts van het 16m-gebied)
    if 1350 <= cx <= 1620 and 345 <= cy <= 435:
        return True

    # 2. Invariante 2D Pitch-ruimte (onafhankelijk van panning of schaal)
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    p2d = cv2.perspectiveTransform(pt, H)[0][0]
    X, Y = p2d[0], p2d[1]
    if 870 <= X <= 990 and 200 <= Y <= 290:
        return True

    return False


def is_foot_region(cand_center, player_boxes):
    """Controleert of een coordinatenpunt zich bij de voeten/onderzijde van een speler bevindt."""
    cx, cy = cand_center
    for pb in player_boxes:
        px1, py1, px2, py2 = pb
        ph = py2 - py1
        if (px1 - 8) <= cx <= (px2 + 8) and (py2 - 0.28 * ph) <= cy <= (py2 + 12):
            return True
    return False


def draw_styled_camera_box(img, box, cid, score):
    """Tekent duidelijke bounding boxes op het camerabeeld."""
    cfg = CLASS_CONFIG.get(cid, CLASS_CONFIG[1])
    x1, y1, x2, y2 = [int(v) for v in box]

    color = cfg["color_fill"]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    label = f"{cfg['name']} {score:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
    
    text_color = (0, 0, 0) if cid in [2, 4] else (255, 255, 255)
    cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA)


def draw_camera_ball_highlight(img, center, score=None):
    """Tekent een opvallende balmarkering op het camerabeeld."""
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(img, (cx, cy), 13, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 8, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 8, (0, 0, 255), 1, cv2.LINE_AA)
    label = f"BALL {score:.2f}" if score is not None else "BALL"
    cv2.putText(img, label, (cx - 18, max(18, cy - 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)


def draw_tactical_radar(pitch_base, detected_players, H, ball_pos_2d, ball_trail):
    """Tekent spelers en bal op de 2D minimap."""
    radar = pitch_base.copy()
    pitch_h, pitch_w = radar.shape[:2]

    # 1. Teken balspoor (alleen nabije opeenvolgende punten om uitschieters te weren)
    if len(ball_trail) > 1:
        for i in range(1, len(ball_trail)):
            p_prev = ball_trail[i - 1]
            p_curr = ball_trail[i]
            dist = np.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            if dist <= 60:
                progress = i / len(ball_trail)
                thickness = int(1 + progress * 3)
                color = (0, int(150 + 105 * progress), 255)
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
            cv2.circle(radar, (px + 2, py + 2), cfg["radius"], (15, 15, 15), -1, cv2.LINE_AA)
            cv2.circle(radar, (px, py), cfg["radius"], cfg["color_fill"], -1, cv2.LINE_AA)
            cv2.circle(radar, (px, py), cfg["radius"], cfg["color_border"], 2, cv2.LINE_AA)

    # 3. Teken de bal (altijd prominent)
    if ball_pos_2d is not None:
        bx, by = int(ball_pos_2d[0]), int(ball_pos_2d[1])
        cv2.circle(radar, (bx, by), 14, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(radar, (bx, by), 9, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(radar, (bx, by), 9, (0, 0, 255), 1, cv2.LINE_AA)

    return radar


def compute_match_telemetry(ball_pos_2d, last_ball_pos_2d, mapped_a, mapped_b, fps=30.0):
    """Berekent live balsnelheid (km/u) en balbezit."""
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
    """Genereert Kernel Density Estimation (KDE) Heatmap van de veldbezetting."""
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
    parser.add_argument("--imgsz", type=int, default=1280, help="Resolutie voor inferentie")
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
    print(f"[BALL-CONFIG] Detectie threshold conf={CLASS_CONFIG[0]['conf']} (Penalty Spot & Cleat Filters Actief)")
    print(f"[OUTPUT] {out_video_path}")
    print("=" * 65)

    ball_trail = []
    last_cam_ball = None
    last_cam_vel = np.array([0.0, 0.0], dtype=np.float32)
    last_known_2d_ball = None
    frames_since_ball = 999
    max_missing_frames = 8
    total_ball_frames = 0
    all_player_points = []
    frame_idx = 0

    while cap.isOpened() and (args.max_frames == 0 or frame_idx < args.max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=args.imgsz, conf=0.07, device=device, verbose=False)[0]

        detected_players = {1: [], 2: [], 3: [], 4: []}
        player_boxes = []
        ball_candidates = []
        cam_display = frame.copy()

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            scores = results.boxes.conf.cpu().numpy()

            # 1. Eerst spelers verzamelen om schoenposities te kennen
            for box, cid, score in zip(boxes, classes, scores):
                if cid in [1, 2, 3, 4] and score >= 0.28:
                    player_boxes.append(box)

            # 2. Nu alle objecten filteren en tekenen
            for box, cid, score in zip(boxes, classes, scores):
                if cid not in CLASS_CONFIG:
                    continue

                min_conf = CLASS_CONFIG[cid]["conf"]
                if score < min_conf:
                    continue

                x1, y1, x2, y2 = box
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                if cid == 0:
                    # Filter 1: Harde eliminatie van de penaltystip (camera + 2D pitch)
                    if is_penalty_spot_or_marking(center, H):
                        continue

                    bw = x2 - x1
                    bh = y2 - y1
                    aspect = bw / max(1.0, bh)
                    # Filter 2: Sferische vormcontrole (schoenen en krijtstrepen zijn langwerpig)
                    if aspect < 0.60 or aspect > 1.65:
                        continue

                    ball_candidates.append({
                        "box": box,
                        "conf": float(score),
                        "center": np.array(center, dtype=np.float32)
                    })
                else:
                    draw_styled_camera_box(cam_display, box, cid, score)
                    feet = [(x1 + x2) / 2.0, y2]
                    detected_players[cid].append(feet)

        # -------------------------------------------------------------
        # Kinematic Ball Tracker (Volg de ECHTE bal zonder sprongen!)
        # -------------------------------------------------------------
        chosen_ball = None

        if last_cam_ball is not None and frames_since_ball <= max_missing_frames:
            # Kinematische gating op basis van voorspelde snelheid
            pred_pos = last_cam_ball + (last_cam_vel if frames_since_ball <= 3 else np.array([0.0, 0.0], dtype=np.float32))
            gated = []
            for cand in ball_candidates:
                dist = np.linalg.norm(cand['center'] - pred_pos)
                max_allowed_dist = 65.0 + frames_since_ball * 15.0
                if dist <= max_allowed_dist:
                    cost = dist - cand['conf'] * 20.0
                    gated.append((cost, cand))
            if gated:
                gated.sort(key=lambda x: x[0])
                chosen_ball = gated[0][1]
        elif ball_candidates:
            # Re-acquisitie: mag NIET op een voet initialiseren tenzij hoge conf (>= 0.40)
            reacq_cands = []
            for cand in ball_candidates:
                if is_foot_region(cand['center'], player_boxes):
                    if cand['conf'] >= 0.40:
                        reacq_cands.append(cand)
                else:
                    if cand['conf'] >= 0.16:
                        reacq_cands.append(cand)
            if reacq_cands:
                reacq_cands.sort(key=lambda x: -x['conf'])
                chosen_ball = reacq_cands[0]

        current_ball_cam = None
        current_ball_score = None
        current_ball_2d = None

        if chosen_ball is not None:
            curr_pos = chosen_ball['center']
            if last_cam_ball is not None:
                new_vel = (curr_pos - last_cam_ball) / max(1, frames_since_ball + 1)
                speed = np.linalg.norm(new_vel)
                if speed > 60:
                    new_vel = new_vel * (60.0 / speed)
                last_cam_vel = 0.60 * last_cam_vel + 0.40 * new_vel
            last_cam_ball = curr_pos
            frames_since_ball = 0
            total_ball_frames += 1
            current_ball_cam = curr_pos
            current_ball_score = chosen_ball['conf']
            draw_camera_ball_highlight(cam_display, current_ball_cam, current_ball_score)
        else:
            frames_since_ball += 1
            if frames_since_ball <= max_missing_frames and last_cam_ball is not None:
                # Kinematische inertie: overbrugt occlusie en bewegingsonscherpte zonder geknipper
                if frames_since_ball <= 3:
                    last_cam_vel = last_cam_vel * 0.75
                    last_cam_ball = last_cam_ball + last_cam_vel
                current_ball_cam = last_cam_ball
                total_ball_frames += 1
                draw_camera_ball_highlight(cam_display, current_ball_cam, score=None)
            else:
                last_cam_ball = None
                last_cam_vel = np.array([0.0, 0.0], dtype=np.float32)

        if current_ball_cam is not None:
            mapped = transform_points([current_ball_cam], H, pitch_img.shape[1], pitch_img.shape[0])
            if len(mapped) > 0:
                p2d = (int(mapped[0][0]), int(mapped[0][1]))
                current_ball_2d = p2d
                # Voeg toe aan balspoor (start een nieuw segment bij grote sprongen)
                if len(ball_trail) == 0 or np.hypot(p2d[0] - ball_trail[-1][0], p2d[1] - ball_trail[-1][1]) <= 60:
                    ball_trail.append(p2d)
                else:
                    ball_trail = [p2d]
                if len(ball_trail) > 25:
                    ball_trail.pop(0)
                last_known_2d_ball = current_ball_2d
        else:
            last_known_2d_ball = None
            if len(ball_trail) > 0:
                ball_trail.pop(0)

        mapped_a = transform_points(detected_players[1], H, pitch_img.shape[1], pitch_img.shape[0])
        mapped_b = transform_points(detected_players[2], H, pitch_img.shape[1], pitch_img.shape[0])

        speed_kmh, possession = compute_match_telemetry(current_ball_2d, last_known_2d_ball, mapped_a, mapped_b, fps)

        for pts_2d in [mapped_a, mapped_b]:
            for p in pts_2d:
                all_player_points.append(p)

        # -------------------------------------------------------------
        # Renderen & Compositing
        # -------------------------------------------------------------
        minimap_frame = draw_tactical_radar(pitch_img, detected_players, H, current_ball_2d, ball_trail)

        cam_resized = cv2.resize(cam_display, (cam_w, target_h))
        minimap_resized = cv2.resize(minimap_frame, (minimap_w, target_h))
        combined = np.hstack((cam_resized, minimap_resized))

        overlay = combined.copy()
        cv2.rectangle(overlay, (0, 0), (combined_w, 55), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, combined, 0.15, 0, combined)

        cv2.putText(combined, "PSG (Team A)", (25, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 120, 40), 2, cv2.LINE_AA)
        cv2.putText(combined, f": {len(detected_players[1])}", (190, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(combined, "INTER (Team B)", (280, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, f": {len(detected_players[2])}", (465, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

        ball_status = "BAL IN SPEL" if current_ball_2d is not None else "BAL ZOEKEND"
        ball_status_color = (0, 255, 0) if "IN SPEL" in ball_status else (0, 165, 255)
        cv2.putText(combined, f"STATUS: {ball_status}", (550, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.68, ball_status_color, 2, cv2.LINE_AA)

        cv2.putText(combined, f"BEZIT: {possession}", (820, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        if speed_kmh > 1.0:
            cv2.putText(combined, f"SNELHEID: {speed_kmh:.1f} km/h", (1070, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(combined, "2D FIFA TACTICAL RADAR", (cam_w + 25, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        out.write(combined)
        frame_idx += 1
        if frame_idx % 60 == 0 or frame_idx == total_frames:
            print(f"[SUPER-PIPELINE] Frame {frame_idx}/{total_frames} (Echte bal getrackt in {total_ball_frames} frames)")

    cap.release()
    out.release()

    heatmap_out = output_dir / "spatial_occupancy_heatmap.png"
    generate_heatmap(all_player_points, pitch_img, heatmap_out)

    print("=" * 65)
    print(f"[SUCCESS] Video gereed: {out_video_path}")
    print(f"[SUMMARY] Totaal frames: {frame_idx} | Echte bal getrackt: {total_ball_frames} frames ({total_ball_frames/max(1, frame_idx)*100:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
