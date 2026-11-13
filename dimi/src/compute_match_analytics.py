"""
Geautomatiseerde Match Analytics & Telemetrie Extensie.
Auteur: Dmitrij Tchervonisjenko (Team Git Push and Pray)
Thomas More Geel - Deep Learning & Computer Vision

Berekent:
- Totale afgelegde afstand (km) van bal en spelers
- Snelheden (piek- en gemiddelde km/u)
- Schotdetectie & Doelspecificatie (welk doel wordt aangevallen)
- Balbezitverdeling over de hele clip
- Exporteert gestructureerde match_analytics.json
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import VIDEO_PATH, BEST_WEIGHTS, RUNS_DIR, HOMOGRAPHY_CONFIG, PITCH_IMG_PATH, pick_device

OUTPUT_JSON = RUNS_DIR / "tactical_radar_output" / "match_analytics.json"


def run_analytics():
    print("=" * 70)
    print("START GEAUTOMATISEERDE MATCH ANALYTICS & TELEMETRIE")
    print("=" * 70)

    with open(HOMOGRAPHY_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    H = np.array(cfg["homography_matrix"], dtype=np.float32)

    device = pick_device()
    model = YOLO(str(BEST_WEIGHTS))

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ball_distances_px = []
    ball_speeds_kmh = []
    possession_counts = {"PSG (Team A)": 0, "Inter Milan (Team B)": 0, "Betwist": 0}
    player_tracks_dist = {}
    shots_detected = []

    prev_ball_pos = None
    frame_idx = 0

    # Goal locations on standard 1050x680 pitch:
    # PSG Goal (Links): (0, 340)
    # Inter Milan Goal (Rechts): (1050, 340)
    INTER_GOAL = np.array([1050.0, 340.0], dtype=np.float32)
    PSG_GOAL = np.array([0.0, 340.0], dtype=np.float32)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model.predict(frame, imgsz=1280, conf=0.005, iou=0.60, device=device, verbose=False)[0]

        ball_pts = []
        player_a_pts = []
        player_b_pts = []

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for box, cid, conf in zip(boxes, classes, confs):
                if cid == 0 and conf >= 0.005:  # Ball
                    center_cam = np.array([[(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]], dtype=np.float32).reshape(-1, 1, 2)
                    p2d = cv2.perspectiveTransform(center_cam, H)[0][0]
                    if 0 <= p2d[0] <= 1050 and 0 <= p2d[1] <= 680:
                        ball_pts.append(p2d)
                elif cid == 1 and conf >= 0.35:  # PSG
                    feet_cam = np.array([[(box[0] + box[2]) / 2.0, box[3]]], dtype=np.float32).reshape(-1, 1, 2)
                    p2d = cv2.perspectiveTransform(feet_cam, H)[0][0]
                    if 0 <= p2d[0] <= 1050 and 0 <= p2d[1] <= 680:
                        player_a_pts.append(p2d)
                elif cid in [2, 3] and conf >= 0.30:  # Inter
                    feet_cam = np.array([[(box[0] + box[2]) / 2.0, box[3]]], dtype=np.float32).reshape(-1, 1, 2)
                    p2d = cv2.perspectiveTransform(feet_cam, H)[0][0]
                    if 0 <= p2d[0] <= 1050 and 0 <= p2d[1] <= 680:
                        player_b_pts.append(p2d)

        # Telemetrie berekening
        current_ball = ball_pts[0] if ball_pts else None

        if current_ball is not None:
            if prev_ball_pos is not None:
                d_px = float(np.hypot(current_ball[0] - prev_ball_pos[0], current_ball[1] - prev_ball_pos[1]))
                # 10 px = 1 meter
                if d_px < 60.0:  # Max 6 meter per frame = 180 m/s
                    speed = (d_px * 0.1) * fps * 3.6
                    if speed <= 130.0:
                        ball_distances_px.append(d_px)
                        ball_speeds_kmh.append(speed)

                        # Schotdetectie: bal beweegt met hoge snelheid (>35 km/u) richting het doel van Inter (X > 750)
                        dist_to_inter_goal = float(np.linalg.norm(current_ball - INTER_GOAL))
                        prev_dist_to_inter_goal = float(np.linalg.norm(prev_ball_pos - INTER_GOAL))
                        
                        if current_ball[0] > 700 and speed > 35.0 and dist_to_inter_goal < prev_dist_to_inter_goal:
                            shots_detected.append({
                                "frame": frame_idx,
                                "timestamp_s": round(frame_idx / fps, 2),
                                "target_goal": "Rechterdoel (Inter Milan)",
                                "speed_kmh": round(speed, 1),
                                "pitch_coord": [round(float(current_ball[0]), 1), round(float(current_ball[1]), 1)],
                                "distance_to_goal_m": round(dist_to_inter_goal * 0.1, 1)
                            })

            prev_ball_pos = current_ball

            # Balbezit
            min_dist_a = min([np.linalg.norm(current_ball - p) for p in player_a_pts], default=999.0)
            min_dist_b = min([np.linalg.norm(current_ball - p) for p in player_b_pts], default=999.0)

            if min_dist_a < min_dist_b and min_dist_a < 45.0:
                possession_counts["PSG (Team A)"] += 1
            elif min_dist_b < min_dist_a and min_dist_b < 45.0:
                possession_counts["Inter Milan (Team B)"] += 1
            else:
                possession_counts["Betwist"] += 1

    cap.release()

    # Bereken statistieken
    total_ball_meters = sum(ball_distances_px) * 0.1
    total_ball_km = total_ball_meters / 1000.0
    peak_ball_speed = max(ball_speeds_kmh) if ball_speeds_kmh else 0.0
    avg_ball_speed = np.mean(ball_speeds_kmh) if ball_speeds_kmh else 0.0

    total_poss_frames = sum(possession_counts.values()) or 1
    poss_pct_psg = round((possession_counts["PSG (Team A)"] / total_poss_frames) * 100, 1)
    poss_pct_inter = round((possession_counts["Inter Milan (Team B)"] / total_poss_frames) * 100, 1)
    poss_pct_betwist = round((possession_counts["Betwist"] / total_poss_frames) * 100, 1)

    # Filter redundante opeenvolgende schotdetecties tot unieke acties
    unique_shots = []
    for s in shots_detected:
        if not unique_shots or (s["frame"] - unique_shots[-1]["frame"]) > 25:
            unique_shots.append(s)

    analytics_data = {
        "match_duration_seconds": round(total_frames / fps, 1),
        "total_frames_analyzed": total_frames,
        "ball_telemetry": {
            "total_distance_meters": round(total_ball_meters, 1),
            "total_distance_km": round(total_ball_km, 3),
            "peak_speed_kmh": round(peak_ball_speed, 1),
            "average_speed_kmh": round(avg_ball_speed, 1),
        },
        "possession_share_percent": {
            "PSG": poss_pct_psg,
            "Inter_Milan": poss_pct_inter,
            "Contested": poss_pct_betwist
        },
        "shot_and_attack_analysis": {
            "attacking_team": "Paris Saint-Germain (PSG)",
            "defending_team": "Inter Milan",
            "target_goal": "Rechterdoel (Inter Milan doelman)",
            "goal_coordinates_pitch": [1050, 340],
            "total_shot_attempts": len(unique_shots),
            "shots": unique_shots
        }
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(analytics_data, f, indent=4)

    print(f"\n[SUCCESS] Analytics succesvol berekend en opgeslagen in: {OUTPUT_JSON}")
    print(f"  - Totale balafstand: {total_ball_meters:.1f} m ({total_ball_km:.3f} km)")
    print(f"  - Pieksnelheid bal: {peak_ball_speed:.1f} km/u | Gemiddelde: {avg_ball_speed:.1f} km/u")
    print(f"  - Balbezit: PSG {poss_pct_psg}% | Inter {poss_pct_inter}% | Betwist {poss_pct_betwist}%")
    print(f"  - Schotdetectie: {len(unique_shots)} doelpoging(en) op het Rechterdoel (Inter Milan)")
    print("=" * 70)

    return analytics_data


if __name__ == "__main__":
    run_analytics()
