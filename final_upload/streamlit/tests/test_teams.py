"""Teams: group -> team letter, K-means fallback on jersey colour for COCO models."""

import numpy as np

from core.teams import assign_kmeans, group_of, is_ball, is_person, jersey_colour, team_of


def test_team_of_groups():
    assert team_of("player_A") == "A"
    assert team_of("player_B") == "B" and team_of("player_B_goal_keeper") == "B"
    assert team_of("referee") == "-" and team_of("other") == "-"


def test_class_helpers():
    assert is_person({"name": "referee"}) and is_person({"name": "person"})
    assert is_ball({"name": "ball"}) and is_ball({"name": "sports ball"})
    assert not is_person({"name": "ball"}) and not is_ball({"name": "player_A"})


def test_group_of_prefers_kmeans_group():
    assert group_of({"name": "person", "kgroup": "team 2"}) == "team 2"
    assert group_of({"name": "player_A"}) == "player_A"


def test_kmeans_splits_two_jersey_colours_and_flags_outlier():
    rng = np.random.default_rng(1)
    reds = [{"name": "person", "jersey": [100 + rng.normal(0, 2), 170 + rng.normal(0, 2), 150 + rng.normal(0, 2)]}
            for _ in range(30)]
    blues = [{"name": "person", "jersey": [90 + rng.normal(0, 2), 120 + rng.normal(0, 2), 80 + rng.normal(0, 2)]}
             for _ in range(30)]
    odd = {"name": "person", "jersey": [250.0, 128.0, 128.0]}   # white: referee / keeper
    people = reds + blues + [odd]
    assert assign_kmeans(people) == 61
    assert len({d["kgroup"] for d in reds}) == 1 and len({d["kgroup"] for d in blues}) == 1
    assert reds[0]["kgroup"] != blues[0]["kgroup"]
    assert odd["kgroup"] == "other"
    assert {reds[0]["kgroup"], blues[0]["kgroup"]} == {"team 1", "team 2"}


def test_kmeans_ignores_team_models_and_needs_two_players():
    people = [{"name": "player_A", "jersey": [1, 2, 3]}, {"name": "person", "jersey": [1, 2, 3]}]
    assert assign_kmeans(people) == 0
    assert "kgroup" not in people[0]


def test_jersey_colour_ignores_grass():
    frame = np.zeros((200, 100, 3), np.uint8)
    frame[:] = (0, 160, 0)                      # grass (BGR)
    frame[20:100, 25:75] = (0, 0, 200)          # red torso
    lab = jersey_colour(frame, [0, 0, 100, 200])
    assert lab is not None and lab[1] > 150      # a (red-green axis) clearly on the red side
    assert jersey_colour(frame, [0, 0, 0, 0]) is None
