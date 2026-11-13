"""Which class / group / team is a detection?

The team model (best.pt) already predicts the groups: player_A (PSG), player_B (Inter),
player_B_goal_keeper, referee. For a COCO model that only knows "person" we fall back to
K-means on the jersey colour (assign_kmeans), which gives "team 1" / "team 2" / "other".
"""

import cv2
import numpy as np

OUTLIER_FACTOR = 2.5   # jersey further than this x the typical distance from both teams = "other"

BALL_CLASSES = {"ball", "sports ball"}
GENERIC_PEOPLE = {"person", "player"}   # classes without a team: the team comes from the jersey colour
TEAM_PEOPLE = {"player_A", "player_B", "player_B_goal_keeper", "referee", "goalkeeper"}
PEOPLE_CLASSES = GENERIC_PEOPLE | TEAM_PEOPLE

# Team letter per group (labeling protocol: A = dark blue = PSG, B = yellow = Inter);
# referees and "other" belong to no team ("-")
TEAM_OF = {"player_A": "A", "player_B": "B", "player_B_goal_keeper": "B", "goalkeeper": "B",
           "team 1": "A", "team 2": "B"}
KEEPERS = {"player_B_goal_keeper", "goalkeeper"}


def is_person(d):
    return d["name"] in PEOPLE_CLASSES


def is_ball(d):
    return d["name"] in BALL_CLASSES


def team_of(group):
    """Team letter "A" / "B" of a group, "-" for referee / other."""
    return TEAM_OF.get(group, "-")


def group_of(d):
    """Group of one detection: the K-means team when assigned, else the class of the model."""
    return d.get("kgroup") or d["name"]


def jersey_colour(frame, box):
    """Mean Lab colour of the torso (upper half of the body, middle half of the width), grass removed."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    crop = frame[max(0, int(y1 + 0.15 * h)):max(0, int(y1 + 0.5 * h)),
                 max(0, int(x1 + 0.25 * w)):max(0, int(x2 - 0.25 * w))]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    grass = (hsv[:, 0] > 30) & (hsv[:, 0] < 90) & (hsv[:, 1] > 50)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)[~grass]
    return [round(v, 1) for v in lab.mean(axis=0).tolist()] if len(lab) >= 5 else None


def assign_kmeans(detections):
    """Fallback for COCO models: split all generic persons into 2 teams on jersey colour (K-means, Lab).

    detections: iterable of detection dicts with "name" and "jersey". Sets "kgroup" to "team 1" / "team 2"
    / "other" (a jersey far from both team centres: referee, keeper) on every generic person with a jersey.
    Returns the number of persons that got a group.
    """
    players = [d for d in detections if d["name"] in GENERIC_PEOPLE and d.get("jersey")]
    if len(players) < 2:
        return 0
    data = np.array([d["jersey"] for d in players], np.float32)
    cv2.setRNGSeed(0)   # same teams every run
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, labels, centres = cv2.kmeans(data, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    dist = np.linalg.norm(data[:, None, :] - centres[None], axis=2)  # distance to both team centres
    typical = np.median(dist.min(axis=1))
    for d, label, row in zip(players, labels.ravel(), dist):
        d["kgroup"] = "other" if row.min() > OUTLIER_FACTOR * max(typical, 1e-6) else f"team {label + 1}"
    return len(players)
