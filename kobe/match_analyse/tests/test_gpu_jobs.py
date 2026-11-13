"""Pure parts of the GPU jobs (gpu/): no model, no GPU needed."""

import json

import numpy as np
import pytest

from core.detection import DETECT_CONF, cache_path
from gpu.common import (MERGED_ID, best_threshold, build_merged_dataset, find_overlap, frame_number, merged_map,
                        remap_label_lines, smooth)
from gpu.export_unseen_frames import frames_to_export, parse_ranges
from gpu.finetune_ball import attempts
from gpu.nms_sweep import DEFAULT, pick_settings

OUR_NAMES = {0: "ball", 1: "player_A", 2: "player_B", 3: "player_B_goal_keeper", 4: "referee"}
COCO_NAMES = {0: "person", 1: "bicycle", 32: "sports ball", 56: "chair"}


# ---------------------------------------------------------------- remapping to the fair ball/person view

def test_merged_map_of_our_classes_and_coco():
    assert merged_map(OUR_NAMES) == {0: MERGED_ID["ball"], 1: MERGED_ID["person"], 2: MERGED_ID["person"],
                                     3: MERGED_ID["person"], 4: MERGED_ID["person"]}
    # COCO: only 'person' and 'sports ball' count, a chair or bicycle is dropped
    assert merged_map(COCO_NAMES) == {0: MERGED_ID["person"], 32: MERGED_ID["ball"]}


def test_remap_label_lines_keeps_boxes_and_drops_unknown_classes():
    lines = ["0 0.5 0.5 0.01 0.01", "3 0.2 0.3 0.04 0.1", "", "7 0.1 0.1 0.1 0.1", "4 0.9 0.9 0.02 0.05"]
    out = remap_label_lines(lines, merged_map(OUR_NAMES))
    assert out == ["0 0.5 0.5 0.01 0.01", "1 0.2 0.3 0.04 0.1", "1 0.9 0.9 0.02 0.05"]


def test_build_merged_dataset_counts(tmp_path):
    for split in ("test",):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
    (tmp_path / "test" / "images" / "frame_0001.jpg").write_bytes(b"x")
    (tmp_path / "test" / "labels" / "frame_0001.txt").write_text("0 .5 .5 .01 .01\n1 .2 .2 .05 .1\n2 .3 .3 .05 .1\n")
    (tmp_path / "data.yaml").write_text("test: test/images\nnames: [ball, player_A, player_B, player_B_goal_keeper, referee]\n")
    yml, counts = build_merged_dataset(tmp_path / "data.yaml", "test", tmp_path / "merged")
    assert counts == {"images": 1, "ball": 1, "person": 2}
    assert (tmp_path / "merged" / "labels" / "frame_0001.txt").read_text().split("\n")[1].startswith("1 ")
    assert "ball" in yml.read_text() and "person" in yml.read_text()


# ---------------------------------------------------------------- leakage

def test_frame_number_from_roboflow_and_export_names():
    assert frame_number("frame_0600_idx_00600_jpg.rf.470a5116704d.jpg") == 600
    assert frame_number("C:/x/frame_0065.jpg") == 65
    assert frame_number("IMG_1234.jpg") is None


def test_find_overlap_by_frame_number_and_by_content(tmp_path):
    def img(name, content):
        p = tmp_path / name
        p.write_bytes(content)
        return p
    train = [img("frame_0001_jpg.rf.a.jpg", b"a"), img("frame_0002_jpg.rf.b.jpg", b"b"), img("other.jpg", b"same")]
    test = [img("frame_0001_jpg.rf.zz.jpg", b"re-encoded"),   # same frame, other bytes (new Roboflow export)
            img("copy.jpg", b"same"),                         # same bytes, no frame number
            img("frame_0300.jpg", b"new")]                     # unseen
    ov = find_overlap(train, test)
    assert ov["n_overlap"] == 2 and ov["by_frame"] == 1 and ov["by_hash"] == 1
    assert ov["frames"] == [1] and ov["n_test"] == 3
    assert find_overlap(train, [test[2]])["n_overlap"] == 0


def test_shipped_datasets_ball_finetune_does_not_leak_into_notebook_test():
    from gpu.common import DEFAULT_TEST_DATA, DEFAULT_TRAIN_DATA, BEST_TRAIN_DATA, dataset_overlap
    if not (DEFAULT_TEST_DATA.exists() and DEFAULT_TRAIN_DATA.exists()):
        pytest.skip("datasets/ not present")
    assert dataset_overlap(DEFAULT_TRAIN_DATA, DEFAULT_TEST_DATA, "test")["n_overlap"] == 0
    # ... while best.pt's own training frames contain the whole notebook test split (the leakage warning)
    best = dataset_overlap(BEST_TRAIN_DATA, DEFAULT_TEST_DATA, "test", train_splits=("train",))
    assert best["n_overlap"] == best["n_test"] > 0


# ---------------------------------------------------------------- thresholds

def test_best_threshold_on_synthetic_curve():
    px = np.linspace(0, 1, 1001)
    f1 = np.exp(-((px - 0.4) ** 2) / 0.02)          # smooth bump with its maximum at conf 0.4
    p, r = px, 1 - px
    best = best_threshold(px, f1, p, r)
    assert best["conf"] == pytest.approx(0.4, abs=0.01)
    assert best["f1"] == pytest.approx(1.0, abs=1e-3)
    assert best["precision"] == pytest.approx(best["conf"]) and best["recall"] == pytest.approx(1 - best["conf"])


def test_best_threshold_prefers_plateau_over_spike_before_a_cliff():
    px = np.linspace(0, 1, 1001)
    f1 = np.where(px < 0.6, 0.9, 0.0)
    f1[595:600] = 0.95                              # a thin spike right before the cliff
    best = best_threshold(px, f1)
    assert best["conf"] < 0.59                      # smoothed: not on the spike
    assert best_threshold(px, f1, smoothing=0)["conf"] == pytest.approx(0.595)


def test_best_threshold_never_found_class():
    best = best_threshold(np.linspace(0, 1, 100), np.zeros(100))
    assert best["conf"] is None and best["f1"] == 0.0


def test_smooth_keeps_length_and_constant():
    y = np.full(1000, 0.7)
    assert len(smooth(y)) == 1000 and np.allclose(smooth(y), 0.7)


# ---------------------------------------------------------------- detection cache key with nms

def _files(d, video=b"video-bytes", weights=b"weights-bytes"):
    d.mkdir(parents=True, exist_ok=True)
    (d / "match.mp4").write_bytes(video)
    (d / "best.pt").write_bytes(weights)
    return d / "match.mp4", d / "best.pt"


def test_cache_key_is_portable_between_folders(tmp_path, monkeypatch):
    import os
    import core.detection as det
    monkeypatch.setattr(det, "CACHE_DIR", tmp_path / "cache")
    a = _files(tmp_path / "laptop" / "match_analyse")
    b = _files(tmp_path / "gpu_pc" / "elders" / "match_analyse")
    os.utime(b[0], (1, 1))                     # another modification time (copy / zip) must not matter
    args = (960, 0.7, 0, 659, 1)
    assert cache_path(*a, *args) == cache_path(*b, *args)
    assert cache_path(*a, *args, nms=False) == cache_path(*b, *args, nms=False)
    # nms is part of the key only when given; other content (a retrained model) gives another key
    assert cache_path(*a, *args, nms=None) == cache_path(*a, *args)
    assert cache_path(*a, *args, nms=False) != cache_path(*a, *args)
    assert cache_path(*a, *args, nms=False) != cache_path(*a, *args, nms=True)
    c = _files(tmp_path / "retrained", weights=b"other-weights")
    assert cache_path(*c, *args) != cache_path(*a, *args)
    assert cache_path(*a, *args).name.startswith("detections_best_960_")


def test_old_style_cache_file_is_still_used(tmp_path, monkeypatch):
    import hashlib
    import core.detection as det
    monkeypatch.setattr(det, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    video, weights = _files(tmp_path / "m")
    v, w = video.resolve(), weights.resolve()
    old_key = {"video": str(v), "video_mtime": v.stat().st_mtime, "weights": str(w),
               "weights_mtime": w.stat().st_mtime, "imgsz": 960, "iou": 0.7, "start": 0, "end": 10, "stride": 1,
               "conf": DETECT_CONF}
    legacy = tmp_path / "cache" / f"detections_best_960_{hashlib.sha1(json.dumps(old_key, sort_keys=True).encode()).hexdigest()[:10]}.json"
    portable = cache_path(video, weights, 960, 0.7, 0, 10, 1)
    assert portable != legacy and not portable.exists()
    legacy.write_text("{}")
    assert cache_path(video, weights, 960, 0.7, 0, 10, 1) == legacy          # existing old cache keeps working
    assert cache_path(video, weights, 960, 0.7, 0, 10, 1, portable=True) == portable
    portable.write_text("{}")
    assert cache_path(video, weights, 960, 0.7, 0, 10, 1) == portable        # portable file wins once it exists


def test_crop_cache_key_is_portable(tmp_path, monkeypatch):
    import core.ball_refine as br
    monkeypatch.setattr(br, "CACHE_DIR", tmp_path / "cache")
    a = _files(tmp_path / "x")
    b = _files(tmp_path / "y" / "z")
    assert br.crop_cache_path(*a) == br.crop_cache_path(*b)


# ---------------------------------------------------------------- resume, leakage guard, verdict

def test_resume_finds_newest_unfinished_run_with_same_settings(tmp_path):
    from gpu.common import find_resumable, save_state, step_done
    settings = {"quick": True, "epochs": 1}
    for name, st in [("2026-01-01_000000", {"settings": settings, "finished": False}),
                     ("2026-01-02_000000", {"settings": settings, "finished": True}),
                     ("2026-01-03_000000", {"settings": {"quick": False}, "finished": False})]:
        (tmp_path / name).mkdir()
        save_state(tmp_path / name, {**st, "steps": {"2. leakage-overzicht": {"status": "ok"}}})
    assert find_resumable(tmp_path, settings) == tmp_path / "2026-01-01_000000"
    assert find_resumable(tmp_path, {"quick": True, "epochs": 2}) is None
    state = {"steps": {"a": {"status": "ok"}, "b": {"status": "MISLUKT"}}}
    assert step_done(state, "a") and not step_done(state, "b") and not step_done(state, "c")
    assert not step_done(state, "a", force=True)


def test_leakage_only_refused_with_strict(monkeypatch):
    import gpu.finetune_ball as fb
    monkeypatch.setattr(fb, "dataset_overlap", lambda *a, **k: {"n_overlap": 3, "n_test": 15, "by_frame": 3,
                                                                 "by_hash": 3, "frames": [600, 601, 602]})
    assert fb.check_leakage("train.yaml", "test.yaml", "test")["n_overlap"] == 3      # default: warn only
    with pytest.raises(fb.LeakageError):
        fb.check_leakage("train.yaml", "test.yaml", "test", strict=True)


def test_verdict_finetune_vs_best():
    from gpu.summary import verdict
    assert verdict(0.2, 0.5, (10, 3), (14, 3))[0] == "ja"
    assert verdict(0.2, 0.1, (10, 3), (9, 4))[0] == "nee"
    assert verdict(0.2, 0.5, (10, 3), (12, 6))[0] == "twijfel"           # more balls but also more false balls
    assert verdict(None, 0.5, (10, 3), (12, 3))[0] == "onbekend"


# ---------------------------------------------------------------- small helpers

def test_pick_settings_keeps_default_and_best_two():
    rows = [{"_iou": i, "_nms": None, "mAP50": m, "AP50_ball": b}
            for i, m, b in [(0.3, 0.5, 0.1), (0.5, 0.8, 0.4), (0.7, 0.7, 0.2), (0.8, 0.6, 0.6)]]
    rows.append({"_iou": 0.7, "_nms": False, "mAP50": 0.2, "AP50_ball": 0.1})
    assert pick_settings(rows) == [DEFAULT, (0.5, None), (0.8, None)]


def test_finetune_oom_attempts_shrink_and_are_unique():
    a = attempts(1280, 4)
    assert a[0] == (1280, 4) and a[-1] == (960, 1) and len(a) == len(set(a))
    assert all(sz * sz * bs >= a[i + 1][0] ** 2 * a[i + 1][1] for i, (sz, bs) in enumerate(a[:-1]))


def test_unseen_frame_ranges():
    assert parse_ranges("63-199, 246-524") == [(63, 199), (246, 524)]
    ids = frames_to_export(parse_ranges("63-199,246-524"), 5)
    assert ids[0] == 63 and all(63 <= i <= 199 or 246 <= i <= 524 for i in ids)
    assert not set(ids) & set(range(200, 246))


def test_is_resumable_only_for_unfinished_checkpoints(tmp_path):
    """A finished (stripped) last.pt must not be 'resumed': ultralytics would start a default coco8 training."""
    import torch
    from gpu.finetune_ball import is_resumable
    torch.save({"epoch": 3, "optimizer": {"state": {}}}, tmp_path / "unfinished.pt")
    torch.save({"epoch": -1, "optimizer": None}, tmp_path / "finished.pt")
    assert is_resumable(tmp_path / "unfinished.pt")
    assert not is_resumable(tmp_path / "finished.pt")
