"""
End-to-End Tactical Mapping Pipeline for Dimi.
Combines:
  1. YOLO11s detection (Players + Ball) at imgsz=1280
  2. Homography perspective transformation (Camera -> 2D Pitch)
  3. Side-by-side synchronized tactical video generation
  4. Spatial Occupancy Heatmap generation (Extensie)
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

# Standaard veld-bestemmingen (1050 x 680 px FIFA veld)
DEFAULT_SRC = np.array([
    [320.0, 310.0],   # 16m lijn linksboven
    [210.0, 680.0],   # 16m lijn linksonder
    [980.0, 240.0],   # Middenlijn boven
    [1010.0, 780.0],  # Middenlijn onder
], dtype=np.float32)

DEFAULT_DST = np.array([
    [178.0, 149.0],   # 16m linksboven op kaart
    [178.0, 530.0],   # 16m linksonder op kaart
    [525.0, 20.0],    # Middenlijn boven
    [525.0, 660.0],   # Middenlijn onder
], dtype=np.float32)


def get_homography_matrix():
    """Laadt geijkte Homografie matrix of berekent standaard matrix."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                H = np.array(data["homography_matrix"], dtype=np.float32)
                print("[INFO] Homografie matrix H geladen uit config:", CONFIG_PATH)
                return H
        except Exception as e:
            print("[WARN] Kon config niet laden, gebruik fallback:", e)

    # Fallback default
    H, _ = cv2.findHomography(DEFAULT_SRC, DEFAULT_DST, cv2.RANSAC)
    print("[INFO] Standaard Homografie matrix H berekend.")
    return H


def transform_points(points, H, pitch_w=1050, pitch_h=680):
    """Projecteert cameracoordinaten (x, y) naar 2D veld coordinaten (X, Y)."""
    if len(points) == 0:
        return []
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    # Clip binnen veldgrens
    transformed[:, 0] = np.clip(transformed[:, 0], 5, pitch_w - 5)
    transformed[:, 1] = np.clip(transformed[:, 1], 5, pitch_h - 5)
    return transformed


def draw_minimap(pitch_template, detected_objects, H, ball_trail=None):
    """Tekent de spelers en de bal als gekleurde radar-stipjes op de 2D kaart."""
    minimap = pitch_template.copy()
    pitch_h, pitch_w = minimap.shape[:2]

    # Kleuren per klasse (BGR)
    COLOR_MAP = {
        0: (0, 0, 255),       # Ball: Rood/Wit
        1: (255, 100, 0),     # Player A: Blauw
        2: (0, 220, 255),     # Player B: Geel/Goud
        3: (0, 140, 255),     # Goalkeeper: Oranje
        4: (40, 40, 40),      # Referee: Zwart
    }

    # Transformeer en teken spelers per klasse
    for cid, pts in detected_objects.items():
        if not pts:
            continue
        mapped_pts = transform_points(pts, H, pitch_w, pitch_h)
        color = COLOR_MAP.get(cid, (200, 200, 200))

        for pt in mapped_pts:
            px, py = int(pt[0]), int(pt[1])
            if cid == 0:
                # Bal: Witte cirkel met rode rand
                cv2.circle(minimap, (px, py), 9, (255, 255, 255), -1)
                cv2.circle(minimap, (px, py), 10, (0, 0, 255), 2)
            else:
                # Speler: Gekleurde dot met witte rand en schaduw
                cv2.circle(minimap, (px + 1, py + 2), 11, (10, 10, 10), -1) # Schaduw
                cv2.circle(minimap, (px, py), 10, color, -1)
                cv2.circle(minimap, (px, py), 11, (255, 255, 255), 2)

    # Optioneel: Teken balspoor (trail)
    if ball_trail and len(ball_trail) > 1:
        for i in range(1, len(ball_trail)):
            cv2.line(minimap, ball_trail[i - 1], ball_trail[i], (0, 165, 255), 2)

    return minimap


def generate_heatmap(all_points, pitch_template, output_path):
    """Genereert een Spatial Occupancy Heatmap (Extensie)."""
    h, w = pitch_template.shape[:2]
    density = np.zeros((h, w), dtype=np.float32)

    for pt in all_points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(density, (x, y), 30, 1.0, -1)

    # Blur density om een vloeiende heatmap te maken
    density = cv2.GaussianBlur(density, (61, 61), 0)
    if density.max() > 0:
        density = density / density.max()

    heatmap_color = cv2.applyColorMap((density * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(pitch_template, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite(str(output_path), overlay)
    print(f"[EXTENSIE] Speler Heatmap succesvol opgeslagen in: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Synchronized Tactical Radar Pipeline")
    parser.add_argument("--weights", type=str, default=str(BEST_WEIGHTS))
    parser.add_argument("--video", type=str, default=str(VIDEO_PATH))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max_frames", type=int, default=0, help="0 voor alle frames")
    args = parser.parse_args()

    device = pick_device()
    H = get_homography_matrix()

    if not Path(args.weights).exists():
        print(f"[ERROR] Weights niet gevonden: {args.weights}")
        return

    # Laad 2D pitch template
    pitch_img = cv2.imread(str(PITCH_IMG_PATH))
    if pitch_img is None:
        print(f"[ERROR] Pitch template ontbreekt op: {PITCH_IMG_PATH}. Run scripts/generate_pitch.py!")
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

    # Resoluties instellen voor side-by-side
    target_h = 720
    # Camera feed schalen naar 1280 x 720
    cam_w = 1280
    # Minimap schalen zodat hoogte 720 matches (1050x680 -> ratio 1050/680 * 720 = 1111)
    minimap_w = int(pitch_img.shape[1] * (target_h / pitch_img.shape[0]))
    combined_w = cam_w + minimap_w

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_video_path), fourcc, fps, (combined_w, target_h))

    print("=" * 65)
    print("[PIPELINE] Starten van Synchrone Tactical Radar...")
    print(f"[PIPELINE] Video output: {out_video_path}")
    print(f"[PIPELINE] Resolutie: {combined_w}x{target_h} | FPS: {fps:.1f}")
    print("=" * 65)

    frame_idx = 0
    all_player_points = []
    ball_trail = []

    while cap.isOpened() and (args.max_frames == 0 or frame_idx < args.max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Inferentie met YOLO
        results = model.predict(frame, imgsz=1280, conf=args.conf, device=device, verbose=False)[0]

        detected_objects = {0: [], 1: [], 2: [], 3: [], 4: []}
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)

            for box, cid in zip(boxes, classes):
                x1, y1, x2, y2 = box
                if cid == 0:
                    # Bal: gebruik middelpunt
                    center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
                    detected_objects[0].append(center)
                else:
                    # Speler: gebruik voeten (onderkant van bounding box)
                    feet = [(x1 + x2) / 2.0, y2]
                    if cid in detected_objects:
                        detected_objects[cid].append(feet)

        # 2. Update balspoor
        if detected_objects[0]:
            ball_2d = transform_points([detected_objects[0][0]], H, pitch_img.shape[1], pitch_img.shape[0])
            if len(ball_2d) > 0:
                ball_trail.append((int(ball_2d[0][0]), int(ball_2d[0][1])))
                if len(ball_trail) > 25:
                    ball_trail.pop(0)

        # 3. Verzamel speler-punten voor heatmap extensie
        for cid in [1, 2, 3]:
            for p in detected_objects[cid]:
                p_2d = transform_points([p], H, pitch_img.shape[1], pitch_img.shape[0])
                if len(p_2d) > 0:
                    all_player_points.append(p_2d[0])

        # 4. Render geannoteerd camerabeeld & minimap
        annotated_cam = results.plot()
        minimap_frame = draw_minimap(pitch_img, detected_objects, H, ball_trail)

        # 5. Schaal beide naar target_h en plak naast elkaar
        cam_resized = cv2.resize(annotated_cam, (cam_w, target_h))
        minimap_resized = cv2.resize(minimap_frame, (minimap_w, target_h))
        combined = np.hstack((cam_resized, minimap_resized))

        # 6. Overlay tactische HUD header
        cv2.rectangle(combined, (0, 0), (combined_w, 45), (20, 20, 20), -1)
        hud_text = f"Frame: {frame_idx:04d} | Team A: {len(detected_objects[1])} | Team B: {len(detected_objects[2])} | Bal: {'JA' if detected_objects[0] else 'NEE'}"
        cv2.putText(combined, hud_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, "2D TACTICAL MINIMAP RADAR", (cam_w + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out.write(combined)
        frame_idx += 1
        if frame_idx % 60 == 0 or frame_idx == total_frames:
            print(f"[SYNC] Frame {frame_idx}/{total_frames} verwerkt...")

    cap.release()
    out.release()

    # 7. Genereer Heatmap Extensie
    heatmap_out = output_dir / "spatial_occupancy_heatmap.png"
    generate_heatmap(all_player_points, pitch_img, heatmap_out)

    print("=" * 65)
    print(f"[SUCCESS] Synchrone video gereed: {out_video_path}")
    print(f"[SUCCESS] Heatmap gereed: {heatmap_out}")
    print("=" * 65)


if __name__ == "__main__":
    main()
