"""
Generate a 2D Top-Down Tactical Soccer Pitch Template
and canonical coordinates dictionary for homography projection.
"""

import os
import json
import cv2
import numpy as np


def draw_tactical_pitch(output_path: str = "data/pitch_template/tactical_pitch.png", 
                        width: int = 1050, 
                        height: int = 680):
    """
    Genereert een realistisch 2D FIFA-voetbalveld (105m x 68m geschaald naar pixels).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Groene achtergrond
    pitch = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8) # Forest Green BGR

    # Grasstrepen toevoegen voor tactische look
    stripe_w = width // 10
    for i in range(0, width, stripe_w * 2):
        cv2.rectangle(pitch, (i, 0), (i + stripe_w, height), (40, 150, 40), -1)

    white = (255, 255, 255)
    line_thickness = 3

    # Veldgrenzen
    cv2.rectangle(pitch, (20, 20), (width - 20, height - 20), white, line_thickness)

    # Middenlijn
    mid_x = width // 2
    cv2.line(pitch, (mid_x, 20), (mid_x, height - 20), white, line_thickness)

    # Middencirkel & middenstip
    mid_y = height // 2
    cv2.circle(pitch, (mid_x, mid_y), 91, white, line_thickness)
    cv2.circle(pitch, (mid_x, mid_y), 5, white, -1)

    # 16-meter gebieden (Strafschopgebied: 16.5m diep, 40.3m breed)
    box_w = int(165 * (width - 40) / 1050)
    box_h = int(403 * (height - 40) / 680)
    box_top_y = (height - box_h) // 2

    # Links strafschopgebied
    cv2.rectangle(pitch, (20, box_top_y), (20 + box_w, box_top_y + box_h), white, line_thickness)
    # Rechts strafschopgebied
    cv2.rectangle(pitch, (width - 20 - box_w, box_top_y), (width - 20, box_top_y + box_h), white, line_thickness)

    # 5-meter gebieden (Doelgebied: 5.5m diep, 18.3m breed)
    small_box_w = int(55 * (width - 40) / 1050)
    small_box_h = int(183 * (height - 40) / 680)
    small_box_top = (height - small_box_h) // 2
    cv2.rectangle(pitch, (20, small_box_top), (20 + small_box_w, small_box_top + small_box_h), white, line_thickness)
    cv2.rectangle(pitch, (width - 20 - small_box_w, small_box_top), (width - 20, small_box_top + small_box_h), white, line_thickness)

    # Penalty stippen (11m)
    pen_dist = int(110 * (width - 40) / 1050)
    cv2.circle(pitch, (20 + pen_dist, mid_y), 4, white, -1)
    cv2.circle(pitch, (width - 20 - pen_dist, mid_y), 4, white, -1)

    cv2.imwrite(output_path, pitch)
    print(f"✅ 2D veld sjabloon opgeslagen in: {output_path}")

    # Canonical keypoints voor homography mapping
    keypoints = {
        "top_left_corner": [20, 20],
        "top_right_corner": [width - 20, 20],
        "bottom_left_corner": [20, height - 20],
        "bottom_right_corner": [width - 20, height - 20],
        "midfield_top": [mid_x, 20],
        "midfield_bottom": [mid_x, height - 20],
        "center_spot": [mid_x, mid_y],
        "left_penalty_box_top": [20 + box_w, box_top_y],
        "left_penalty_box_bottom": [20 + box_w, box_top_y + box_h],
        "right_penalty_box_top": [width - 20 - box_w, box_top_y],
        "right_penalty_box_bottom": [width - 20 - box_w, box_top_y + box_h],
    }

    json_path = os.path.splitext(output_path)[0] + "_keypoints.json"
    with open(json_path, "w") as f:
        json.dump(keypoints, f, indent=4)
    print(f"✅ Canonical keypoints opgeslagen in: {json_path}")


if __name__ == "__main__":
    draw_tactical_pitch()
