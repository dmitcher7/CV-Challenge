"""
Extract frames from a soccer video clip for annotation.
Usage:
    python scripts/extract_frames.py --video data/raw_video/match.mp4 --num_frames 60 --output data/frames/
"""

import os
import argparse
import cv2


def extract_frames(video_path: str, output_dir: str, num_frames: int = 60):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video niet gevonden op pad: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        raise ValueError("Kon geen frames lezen uit het videobestand.")

    duration_sec = total_frames / fps if fps > 0 else 0
    print(f"🎬 Video geladen: {total_frames} frames, {fps:.1f} FPS (~{duration_sec:.1f} seconden).")

    # Bepaal de frame-intervallen
    interval = max(1, total_frames // num_frames)
    extracted_count = 0

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if i % interval == 0 and extracted_count < num_frames:
            frame_filename = os.path.join(output_dir, f"frame_{extracted_count:04d}_idx_{i:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1

    cap.release()
    print(f"✅ Klaar! {extracted_count} frames opgeslagen in '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraheer frames uit videoclip voor labeling.")
    parser.add_argument("--video", type=str, default="data/raw_video/match.mp4", help="Pad naar de .mp4 video")
    parser.add_argument("--output", type=str, default="data/frames", help="Doelmap voor geëxtraheerde frames")
    parser.add_argument("--num_frames", type=int, default=60, help="Aantal frames om te extraheren (standaard: 60)")

    args = parser.parse_args()
    extract_frames(args.video, args.output, args.num_frames)
