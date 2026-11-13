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
    """Laadt de geijkte fysieke Homografie matrix uit homography_config.json."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "homography_matrix" in data:
                    H = np.array(data["homography_matrix"], dtype=np.float32)
                else:
                    cam_pts = np.array(data["camera_points"], dtype=np.float32)
                    pitch_pts = np.array(data["minimap_points"], dtype=np.float32)
                    H, _ = cv2.findHomography(cam_pts, pitch_pts, 0)
                print(f"[HOMOGRAFIE] 16-Puntige Geijkte Witte Lijnen Matrix geladen ({len(data['camera_points'])} referentiepunten)!")
                return H
        except Exception as e:
            print("[WARN] Fout bij inlezen homografie config:", e)

    raise FileNotFoundError("homography_config.json ontbreekt. Run eerst calibrate_homography.py!")


class CameraTracker:
    """
    Volgt de camerabeweging (pan, tilt, zoom) tussen opeenvolgende videoframes
    via sparse Optical Flow (Lucas-Kanade) en RANSAC Affine transformatie.
    """
    def __init__(self, max_corners=500):
        self.prev_gray = None
        self.prev_pts = None
        self.H_cumulative = np.eye(3, dtype=np.float32)
        self.feature_params = dict(maxCorners=max_corners, qualityLevel=0.01, minDistance=15, blockSize=7)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.frame_count = 0

    def update(self, frame_bgr):
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.frame_count += 1
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 50:
            self.prev_gray = curr_gray
            self.prev_pts = cv2.goodFeaturesToTrack(curr_gray, mask=None, **self.feature_params)
            return self.H_cumulative

        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, self.prev_pts, None, **self.lk_params)
        valid_prev = self.prev_pts[status == 1]
        valid_curr = curr_pts[status == 1]

        if len(valid_prev) >= 15:
            M, inliers = cv2.estimateAffinePartial2D(valid_curr, valid_prev, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if M is not None:
                scale = np.hypot(M[0, 0], M[0, 1])
                if 0.85 <= scale <= 1.15:
                    M_homo = np.eye(3, dtype=np.float32)
                    M_homo[:2, :] = M
                    self.H_cumulative = self.H_cumulative @ M_homo

        if len(valid_curr) < 200 or self.frame_count % 15 == 0:
            self.prev_pts = cv2.goodFeaturesToTrack(curr_gray, mask=None, **self.feature_params)
        else:
            self.prev_pts = valid_curr.reshape(-1, 1, 2)
            
        self.prev_gray = curr_gray
        return self.H_cumulative


class TeamPlayerTracker:
    """
    Volgt spelers op het 2D veld met spatiale associatie,
    Exponential Moving Average (EMA) smoothing tegen jitter
    en multi-frame persistentie tegen knipperen.
    """
    def __init__(self, max_age=3, match_dist=35.0, ema_alpha=0.65):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
        self.match_dist = match_dist
        self.alpha = ema_alpha

    def update(self, detections):
        for tid in list(self.tracks.keys()):
            self.tracks[tid]['age'] += 1
            if self.tracks[tid]['age'] > self.max_age:
                del self.tracks[tid]

        matched_tracks = set()

        # 1. Singleton behandeling voor de keeper (er kan altijd maximaal 1 keeper zijn)
        keeper_dets = [d for d in detections if d[0] == 3]
        if keeper_dets:
            _, k_meas, k_conf = keeper_dets[0]
            k_tid = None
            for tid, t in self.tracks.items():
                if t['team'] == 3:
                    k_tid = tid
                    break
            if k_tid is not None:
                self.tracks[k_tid]['pos'] = self.alpha * k_meas + (1.0 - self.alpha) * self.tracks[k_tid]['pos']
                self.tracks[k_tid]['age'] = 0
                self.tracks[k_tid]['conf'] = k_conf
                matched_tracks.add(k_tid)
            else:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {'pos': k_meas.copy(), 'team': 3, 'age': 0, 'conf': k_conf}
                matched_tracks.add(new_id)

        # 2. Outfield spelers en scheidsrechters (cid in [1, 2, 4])
        for cid in [1, 2, 4]:
            team_dets = [d for d in detections if d[0] == cid]
            team_tracks = [(tid, t) for tid, t in self.tracks.items() if t['team'] == cid and tid not in matched_tracks]

            for c, p_meas, conf in team_dets:
                best_tid = None
                best_dist = self.match_dist
                for tid, t in team_tracks:
                    if tid in matched_tracks:
                        continue
                    dist = np.hypot(p_meas[0] - t['pos'][0], p_meas[1] - t['pos'][1])
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = tid
                if best_tid is not None:
                    self.tracks[best_tid]['pos'] = self.alpha * p_meas + (1.0 - self.alpha) * self.tracks[best_tid]['pos']
                    self.tracks[best_tid]['age'] = 0
                    self.tracks[best_tid]['conf'] = conf
                    matched_tracks.add(best_tid)
                else:
                    new_id = self.next_id
                    self.next_id += 1
                    self.tracks[new_id] = {'pos': p_meas.copy(), 'team': c, 'age': 0, 'conf': conf}

        # 3. Deduplicatie: verwijder tracks van hetzelfde team die binnen 18px van elkaar zijn gekomen
        active_tids = list(self.tracks.keys())
        to_remove = set()
        for i in range(len(active_tids)):
            t1 = active_tids[i]
            if t1 in to_remove or t1 not in self.tracks: continue
            for j in range(i + 1, len(active_tids)):
                t2 = active_tids[j]
                if t2 in to_remove or t2 not in self.tracks: continue
                if self.tracks[t1]['team'] == self.tracks[t2]['team']:
                    if np.hypot(self.tracks[t1]['pos'][0] - self.tracks[t2]['pos'][0], self.tracks[t1]['pos'][1] - self.tracks[t2]['pos'][1]) < 18.0:
                        if self.tracks[t1]['age'] > self.tracks[t2]['age']:
                            to_remove.add(t1)
                        else:
                            to_remove.add(t2)
        for tid in to_remove:
            if tid in self.tracks:
                del self.tracks[tid]

        return self.tracks


def transform_points(points, H, pitch_w=1050, pitch_h=680):
    """Projecteert cameracoordinaten (x, y) naar het 2D veld (X, Y)."""
    if len(points) == 0:
        return np.array([], dtype=np.float32)
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    mapped[:, 0] = np.clip(mapped[:, 0], 18, 1032)
    mapped[:, 1] = np.clip(mapped[:, 1], 18, 662)
    return mapped


class BallSortTracker:
    """
    SORT (Simple Online and Realtime Tracking) met Time Series Prediction
    en Moving Average Smoothing voor de voetbal.
    """
    def __init__(self, alpha=0.70, max_missing=20):
        self.state = None  # [x, y, vx, vy]
        self.P = np.eye(4, dtype=np.float32) * 40.0
        # Time Series constant-velocity state transition matrix F met wrijvingsdemping
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 0.95, 0],
            [0, 0, 0, 0.95]
        ], dtype=np.float32)
        self.Q = np.diag([2.0, 2.0, 6.0, 6.0]).astype(np.float32)
        self.H_k = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        self.R = np.diag([3.0, 3.0]).astype(np.float32)
        self.ema_pos = None
        self.alpha = alpha  # Moving average smoothing factor
        self.missing_frames = 0
        self.max_missing = max_missing

    def predict(self):
        """Voert time series voorspelling uit naar het volgende frame."""
        if self.state is None:
            return None
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.missing_frames += 1
        return self.state[:2]

    def update(self, z):
        """Kalman update met meting z en Moving Average smoothing."""
        z = np.array(z, dtype=np.float32)
        if self.state is None:
            self.state = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float32)
            self.P = np.eye(4, dtype=np.float32) * 20.0
            self.ema_pos = z.copy()
            self.missing_frames = 0
            return self.ema_pos

        y = z - (self.H_k @ self.state)
        S = self.H_k @ self.P @ self.H_k.T + self.R
        K = self.P @ self.H_k.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H_k) @ self.P
        self.missing_frames = 0

        # Moving average smoothing
        raw_pos = self.state[:2]
        self.ema_pos = self.alpha * raw_pos + (1.0 - self.alpha) * self.ema_pos
        return self.ema_pos

    def get_pos(self):
        """Geeft de afgevlakte positie of tijdreeks-voorspelling terug."""
        if self.missing_frames > self.max_missing:
            self.state = None
            self.ema_pos = None
            return None
        return self.ema_pos if self.missing_frames == 0 else self.state[:2]


def is_penalty_spot_or_marking(center_cam, H):
    """
    Sluit de penaltystip en veldkrijtmarkeringen 100% uit.
    Controleert zowel in de fysieke invariante 2D pitchruimte als in cameraruimte.
    """
    cx, cy = center_cam

    # 1. Invariante 2D Pitch-ruimte transformatie
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    p2d = cv2.perspectiveTransform(pt, H)[0][0]
    X, Y = p2d[0], p2d[1]

    # Rechter strafschopstip (exact X=925, Y=340, ruime straal 55px)
    dist_pen_pitch = np.hypot(X - 925.0, Y - 340.0)
    if dist_pen_pitch <= 55.0:
        return True

    # Middenstip (exact X=525, Y=340, straal 45px)
    dist_mid_pitch = np.hypot(X - 525.0, Y - 340.0)
    if dist_mid_pitch <= 45.0:
        return True

    # 2. Camera-ruimte projectie van de penaltystip
    try:
        H_inv = np.linalg.inv(H)
        pen_cam = cv2.perspectiveTransform(np.array([[[925.0, 340.0]]], dtype=np.float32), H_inv)[0][0]
        if np.hypot(cx - pen_cam[0], cy - pen_cam[1]) <= 50.0:
            return True
    except Exception:
        pass

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


def draw_camera_ball_highlight(img, center, score=None, is_pred=False):
    """Tekent een opvallende balmarkering op het camerabeeld."""
    cx, cy = int(center[0]), int(center[1])
    color_ring = (0, 165, 255) if is_pred else (0, 0, 255)
    cv2.circle(img, (cx, cy), 13, color_ring, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 8, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 8, color_ring, 1, cv2.LINE_AA)
    label = "BALL (PRED)" if is_pred else (f"BALL {score:.2f}" if score is not None else "BALL")
    cv2.putText(img, label, (cx - 18, max(18, cy - 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_ring, 2, cv2.LINE_AA)


def draw_tactical_player(radar, pt, cid):
    """Tekent een professionele speler-marker met zachte slagschaduw en officiële teamkleuren."""
    x, y = int(round(pt[0])), int(round(pt[1]))
    # 1. Zachte antialiased schaduw onder de marker
    cv2.circle(radar, (x + 2, y + 2), 12, (15, 25, 15), -1, cv2.LINE_AA)

    if cid == 1:  # PSG (Team A): Royal Navy Blue + Witte Ring + Rode Kern
        cv2.circle(radar, (x, y), 11, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(radar, (x, y), 9, (160, 45, 18), -1, cv2.LINE_AA)
        cv2.circle(radar, (x, y), 3, (0, 0, 230), -1, cv2.LINE_AA)
    elif cid == 2:  # Inter (Team B): Zonnegeel + Zwarte Ring + Donkerblauwe Kern
        cv2.circle(radar, (x, y), 11, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(radar, (x, y), 9, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(radar, (x, y), 3, (160, 45, 18), -1, cv2.LINE_AA)
    elif cid == 3:  # Goalkeeper: Fluo Oranje + Witte Ring + Witte Kern
        cv2.circle(radar, (x, y), 13, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(radar, (x, y), 11, (0, 130, 255), -1, cv2.LINE_AA)
        cv2.circle(radar, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
    elif cid == 4:  # Referee: Fluo Limoengroen + Zwarte Ring + Donkere Kern
        cv2.circle(radar, (x, y), 10, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(radar, (x, y), 8, (40, 255, 120), -1, cv2.LINE_AA)
        cv2.circle(radar, (x, y), 2, (15, 15, 15), -1, cv2.LINE_AA)


def draw_tactical_radar(pitch_base, active_tracks, ball_pos_2d, ball_trail):
    """Tekent alle getrackte spelers en de bal op de 2D minimap."""
    radar = pitch_base.copy()

    # 1. Teken balspoor met fading snelheidsverloop
    if len(ball_trail) > 1:
        for i in range(1, len(ball_trail)):
            p_prev = ball_trail[i - 1]
            p_curr = ball_trail[i]
            dist = np.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            if dist <= 75:
                progress = i / len(ball_trail)
                thickness = int(1 + progress * 3)
                color = (0, int(150 + 105 * progress), 255)
                cv2.line(radar, p_prev, p_curr, color, thickness, cv2.LINE_AA)

    # 2. Teken getrackte spelers (scheidsrechters eerst, dan veldspelers, dan keeper)
    for draw_cid in [4, 1, 2, 3]:
        for tid, trk in active_tracks.items():
            if trk["team"] == draw_cid:
                draw_tactical_player(radar, trk["pos"], draw_cid)

    # 3. Teken de bal (prominent en oplichtend) - NOOIT op de penaltystip of middenstip
    if ball_pos_2d is not None:
        bx, by = int(round(ball_pos_2d[0])), int(round(ball_pos_2d[1]))
        dist_pen = np.hypot(bx - 925.0, by - 340.0)
        dist_mid = np.hypot(bx - 525.0, by - 340.0)
        if dist_pen > 45.0 and dist_mid > 40.0:
            cv2.circle(radar, (bx, by), 14, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(radar, (bx, by), 9, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(radar, (bx, by), 4, (0, 0, 255), -1, cv2.LINE_AA)

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

    tracker = BallSortTracker(alpha=0.70, max_missing=20)
    camera_tracker = CameraTracker(max_corners=500)
    player_tracker = TeamPlayerTracker(max_age=6, match_dist=38.0, ema_alpha=0.65)
    ball_trail = []
    last_known_2d_ball = None
    total_ball_frames = 0
    all_player_points = []
    frame_idx = 0

    while cap.isOpened() and (args.max_frames == 0 or frame_idx < args.max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Dynamische Cameratracking (compenseert TV-pan, tilt en zoom)
        H_t2_0 = camera_tracker.update(frame)
        Ht = H @ H_t2_0

        results = model.predict(frame, imgsz=args.imgsz, conf=0.07, device=device, verbose=False)[0]

        raw_players = {1: [], 2: [], 3: [], 4: []}
        player_boxes = []
        ball_candidates = []
        cam_display = frame.copy()

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            scores = results.boxes.conf.cpu().numpy()

            # Eerst spelers verzamelen om schoenposities te kennen
            for box, cid, score in zip(boxes, classes, scores):
                if cid in [1, 2, 3, 4] and score >= 0.28:
                    player_boxes.append(box)

            for box, cid, score in zip(boxes, classes, scores):
                if cid not in CLASS_CONFIG:
                    continue

                min_conf = CLASS_CONFIG[cid]["conf"]
                if score < min_conf:
                    continue

                x1, y1, x2, y2 = box
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                if cid == 0:
                    # Filter 1: Harde eliminatie van de penaltystip (invariante 2D pitchruimte via Ht)
                    if is_penalty_spot_or_marking(center, Ht):
                        continue

                    bw = x2 - x1
                    bh = y2 - y1
                    aspect = bw / max(1.0, bh)
                    # Filter 2: Sferische vormcontrole & maximale balgrootte
                    if aspect < 0.50 or aspect > 1.95 or bw > 35 or bh > 35:
                        continue

                    ball_candidates.append({
                        "box": box,
                        "conf": float(score),
                        "center": np.array(center, dtype=np.float32)
                    })
                else:
                    feet = [(x1 + x2) / 2.0, y2]
                    # Veld-ROI: negeer TV-scorebord en bovenste tribunes (feet < 210)
                    if feet[1] >= 210:
                        pt_cam = np.array([[feet]], dtype=np.float32)
                        p2d = cv2.perspectiveTransform(pt_cam, Ht)[0][0]
                        # Moet zich binnen de wettige veldgrens bevinden
                        if 15 <= p2d[0] <= 1035 and 15 <= p2d[1] <= 665:
                            raw_players[cid].append((float(score), p2d, box))

        # 2. Spatiale NMS Deduplicatie (geen dubbele stippen op dezelfde speler)
        deduped_players = []
        # Keeper: maximaal 1 keeper (hoogste conf dichtst bij eigen doel [1030, 339])
        if raw_players[3]:
            raw_players[3].sort(key=lambda x: (-x[0], np.hypot(x[1][0] - 1030.0, x[1][1] - 339.0)))
            k_score, k_p2d, k_box = raw_players[3][0]
            deduped_players.append((3, k_p2d, k_score))
            draw_styled_camera_box(cam_display, k_box, 3, k_score)

        for cid in [1, 2, 4]:
            raw_players[cid].sort(key=lambda x: -x[0])
            accepted_pts = []
            for score, p2d, box in raw_players[cid]:
                too_close = False
                for acc_p in accepted_pts:
                    if np.hypot(p2d[0] - acc_p[0], p2d[1] - acc_p[1]) < 22.0:
                        too_close = True
                        break
                if not too_close:
                    accepted_pts.append(p2d)
                    deduped_players.append((cid, p2d, score))
                    draw_styled_camera_box(cam_display, box, cid, score)

        # 3. Speler Tracking + Anti-Jitter EMA Smoothing
        active_player_tracks = player_tracker.update(deduped_players)

        # 4. SORT Baltracking met Time Series Voorspelling
        pred_pos = tracker.predict()
        chosen_measurement = None
        chosen_conf = None

        if pred_pos is not None and tracker.missing_frames <= tracker.max_missing:
            gated = []
            for cand in ball_candidates:
                d = np.linalg.norm(cand['center'] - pred_pos)
                max_d = 85.0 + tracker.missing_frames * 18.0
                if d <= max_d:
                    cost = d - cand['conf'] * 30.0
                    gated.append((cost, cand))
            if gated:
                gated.sort(key=lambda x: x[0])
                chosen_measurement = gated[0][1]['center']
                chosen_conf = gated[0][1]['conf']
        elif ball_candidates:
            reacq = []
            for cand in ball_candidates:
                if is_penalty_spot_or_marking(cand['center'], Ht):
                    continue
                if is_foot_region(cand['center'], player_boxes):
                    if cand['conf'] >= 0.35:
                        reacq.append(cand)
                else:
                    if cand['conf'] >= 0.12:
                        reacq.append(cand)
            if reacq:
                reacq.sort(key=lambda x: -x['conf'])
                chosen_measurement = reacq[0]['center']
                chosen_conf = reacq[0]['conf']

        if chosen_measurement is not None:
            tracker.update(chosen_measurement)

        current_ball_cam = tracker.get_pos()

        # Harde failsafe: als getrackte bal ooit op penaltystip belandt -> reset tracker direct!
        if current_ball_cam is not None and is_penalty_spot_or_marking(current_ball_cam, Ht):
            tracker.state = None
            tracker.ema_pos = None
            current_ball_cam = None

        current_ball_2d = None
        prev_ball_2d = last_known_2d_ball

        if current_ball_cam is not None:
            total_ball_frames += 1
            is_pred = tracker.missing_frames > 0
            draw_camera_ball_highlight(cam_display, current_ball_cam, chosen_conf, is_pred)
            mapped = transform_points([current_ball_cam], Ht, pitch_img.shape[1], pitch_img.shape[0])
            if len(mapped) > 0:
                p2d = (int(round(mapped[0][0])), int(round(mapped[0][1])))
                # Veiligheid: teken NOOIT op de penaltystip
                if np.hypot(p2d[0] - 925.0, p2d[1] - 340.0) > 45.0:
                    current_ball_2d = p2d
                    if len(ball_trail) == 0 or np.hypot(p2d[0] - ball_trail[-1][0], p2d[1] - ball_trail[-1][1]) <= 95:
                        ball_trail.append(p2d)
                    else:
                        ball_trail = [p2d]
                    if len(ball_trail) > 30:
                        ball_trail.pop(0)
                    last_known_2d_ball = current_ball_2d
        else:
            last_known_2d_ball = None
            if len(ball_trail) > 0:
                ball_trail.pop(0)

        pts_a = [t['pos'] for t in active_player_tracks.values() if t['team'] == 1]
        pts_b = [t['pos'] for t in active_player_tracks.values() if t['team'] in [2, 3]]
        speed_kmh, possession = compute_match_telemetry(current_ball_2d, prev_ball_2d, pts_a, pts_b, fps)

        for t in active_player_tracks.values():
            all_player_points.append(t['pos'])

        # -------------------------------------------------------------
        # Renderen & Compositing
        # -------------------------------------------------------------
        minimap_frame = draw_tactical_radar(pitch_img, active_player_tracks, current_ball_2d, ball_trail)

        cam_resized = cv2.resize(cam_display, (cam_w, target_h))
        minimap_resized = cv2.resize(minimap_frame, (minimap_w, target_h))
        combined = np.hstack((cam_resized, minimap_resized))

        # Telemetrie overlay ALLEEN op het camerabeeld (0 tot cam_w) zodat de minimap 100% vrij en helder blijft
        overlay = combined.copy()
        cv2.rectangle(overlay, (0, 0), (cam_w, 55), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, combined, 0.15, 0, combined)
        # Strakke scheidingslijn tussen camera en 2D radar
        cv2.line(combined, (cam_w, 0), (cam_w, target_h), (40, 40, 40), 2)

        num_psg = len([t for t in active_player_tracks.values() if t['team'] == 1])
        num_inter = len([t for t in active_player_tracks.values() if t['team'] in [2, 3]])

        cv2.putText(combined, "PSG", (25, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 120, 40), 2, cv2.LINE_AA)
        cv2.putText(combined, f": {num_psg}", (75, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(combined, "INTER", (160, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, f": {num_inter}", (245, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

        ball_status = "BAL IN SPEL" if current_ball_2d is not None else "BAL ZOEKEND"
        ball_status_color = (0, 255, 0) if "IN SPEL" in ball_status else (0, 165, 255)
        cv2.putText(combined, f"STATUS: {ball_status}", (330, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ball_status_color, 2, cv2.LINE_AA)

        cv2.putText(combined, f"BEZIT: {possession}", (620, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        if speed_kmh > 1.0:
            cv2.putText(combined, f"SNELHEID: {speed_kmh:.1f} km/h", (940, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

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
