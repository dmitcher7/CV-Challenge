"""Smoke test: the real model on ~10 frames (stride 5). Skipped when the model or the video is not there."""

import csv

import pytest

from core.paths import DEFAULT_WEIGHTS, KEYPOINTS_JSON, VIDEO

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(not (DEFAULT_WEIGHTS.exists() and VIDEO.exists() and KEYPOINTS_JSON.exists()),
                       reason="models/best.pt, data/match.mp4 or data/keypoints.json missing (local files)"),
]


def test_pipeline_on_ten_frames(tmp_path):
    from core.pipeline import run, save_frames_csv, save_tracks_csv

    result = run(start=0, end=45, stride=5)
    assert result.ids == list(range(0, 46, 5)) and len(result.ids) == 10
    assert set(result.people) == set(result.ids) and set(result.possession) == set(result.ids)

    rows = [p for ps in result.people.values() for p in ps]
    assert rows, "no tracked people at all"
    assert all(-3 <= p["x_m"] <= 108 and -3 <= p["y_m"] <= 71 for p in rows)
    assert all(p["team"] in ("A", "B", "-") for p in rows)

    frames_csv = save_frames_csv(result, tmp_path / "frames.csv")
    tracks_csv = save_tracks_csv(result, tmp_path / "tracks.csv")
    frames = list(csv.DictReader(open(frames_csv, encoding="utf-8")))
    tracks = list(csv.DictReader(open(tracks_csv, encoding="utf-8")))
    assert list(frames[0]) == ["frame", "time_s", "ball_x_m", "ball_y_m", "ball_state", "possession_team",
                               "possession_track_id"]
    assert list(tracks[0]) == ["frame", "time_s", "track_id", "group", "team", "x_m", "y_m", "conf",
                              "box_x1", "box_y1", "box_x2", "box_y2"]
    assert len(frames) == 10 and len(tracks) == len(rows)
    assert {f["ball_state"] for f in frames} <= {"detected", "interpolated", "none"}
