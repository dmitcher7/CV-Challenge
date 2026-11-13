"""Single-ball constant-velocity Kalman tracker for temporal video smoothing."""

import cv2
import numpy as np


class BallTracker:
    """Associate one ball over time and bridge a few missed detections.

    This uses the useful core of SORT (Kalman prediction plus gated data
    association) without multi-object bookkeeping, because a football match has
    one ball. It avoids the lag that a plain moving average introduces.
    """

    def __init__(self, frame_shape, init_conf=0.20, max_missed=4, gate_fraction=0.06):
        height, width = frame_shape[:2]
        self.init_conf = init_conf
        self.max_missed = max_missed
        self.max_distance = gate_fraction * float(np.hypot(width, height))
        self.active = False
        self.missed = 0
        self.confidence = 0.0

        # State: center_x, center_y, velocity_x, velocity_y, width, height.
        self.filter = cv2.KalmanFilter(6, 4)
        self.filter.transitionMatrix = np.array(
            [
                [1, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.filter.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.filter.processNoiseCov = np.diag([4, 4, 16, 16, 1, 1]).astype(np.float32)
        self.filter.measurementNoiseCov = np.diag([9, 9, 4, 4]).astype(np.float32)
        self.filter.errorCovPost = np.eye(6, dtype=np.float32) * 10

    @staticmethod
    def _measurement(box):
        x1, y1, x2, y2 = box
        return np.array(
            [[(x1 + x2) / 2], [(y1 + y2) / 2], [max(x2 - x1, 4)], [max(y2 - y1, 4)]],
            dtype=np.float32,
        )

    @staticmethod
    def _box(state):
        cx, cy, _, _, width, height = state.reshape(-1)
        width, height = max(float(width), 4), max(float(height), 4)
        return [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]

    def _start(self, box, confidence):
        measurement = self._measurement(box)
        cx, cy, width, height = measurement.reshape(-1)
        state = np.array([[cx], [cy], [0], [0], [width], [height]], dtype=np.float32)
        self.filter.statePost = state
        self.filter.statePre = state.copy()
        self.active = True
        self.missed = 0
        self.confidence = confidence
        return {"box": box, "confidence": confidence, "predicted": False}

    def update(self, candidates):
        """Update from ``[(box_xyxy, confidence), ...]`` and return one tracked ball or None."""
        if not self.active:
            eligible = [candidate for candidate in candidates if candidate[1] >= self.init_conf]
            return self._start(*max(eligible, key=lambda item: item[1])) if eligible else None

        prediction = self.filter.predict()
        predicted_center = prediction[:2, 0]
        associated = []
        for box, confidence in candidates:
            center = self._measurement(box)[:2, 0]
            distance = float(np.linalg.norm(center - predicted_center))
            if distance <= self.max_distance:
                # Prefer proximity, using confidence only to break near-ties.
                associated.append((distance - 20 * confidence, box, confidence))

        if associated:
            _, box, confidence = min(associated, key=lambda item: item[0])
            corrected = self.filter.correct(self._measurement(box))
            self.missed = 0
            self.confidence = confidence
            return {"box": self._box(corrected), "confidence": confidence, "predicted": False}

        self.missed += 1
        if self.missed > self.max_missed:
            self.active = False
            self.confidence = 0.0
            return None

        self.confidence *= 0.75
        return {"box": self._box(prediction), "confidence": self.confidence, "predicted": True}
