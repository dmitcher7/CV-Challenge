"""Reading the match video with OpenCV."""

import cv2


def video_info(path):
    """{"frames", "fps", "width", "height"} of a video, or None if it cannot be opened."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    info = {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


def read_frame(path, idx):
    """One frame (BGR) of the video, or None. Seeking is frame-exact for match.mp4."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def iter_frames(path, ids):
    """Yield (idx, frame) for the frame indices in ids, reading the video once from the first to the last."""
    ids = sorted(ids)
    if not ids:
        return
    wanted = set(ids)
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, ids[0])
    idx = ids[0]
    while idx <= ids[-1]:
        if idx in wanted:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
        elif not cap.grab():   # skip a frame without decoding it to an image
            break
        idx += 1
    cap.release()


def frame_ids(start, end, stride):
    """Frame indices that are processed: start..end (inclusive), every stride-th frame."""
    return list(range(start, end + 1, stride))
