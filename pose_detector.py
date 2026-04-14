import os
import mediapipe as mp
import cv2
import numpy as np

# Landmark indices (same as legacy API):
# 11 = left shoulder,  12 = right shoulder
# 23 = left hip,       24 = right hip
# 25 = left knee,      26 = right knee
# 27 = left ankle,     28 = right ankle

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task"
)

_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]


class PoseDetector:
    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
    ):
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._frame_ts = 0

    def process(self, frame):
        """Run pose detection on a BGR frame.

        Returns (annotated_frame, list of NormalizedLandmark or empty list).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        landmarks = result.pose_landmarks[0] if result.pose_landmarks else []

        if landmarks:
            self._draw_landmarks(frame, landmarks)

        return frame, landmarks

    def get_landmark(self, landmarks, index: int):
        """Return (x, y, visibility) in normalised coordinates.

        Returns None if no landmarks or visibility < 0.7.
        """
        if not landmarks:
            return None

        lm = landmarks[index]
        if lm.visibility < 0.7:
            return None

        return (lm.x, lm.y, lm.visibility)

    def get_landmark_px(self, landmarks, index: int, frame_w: int, frame_h: int):
        """Return (x, y, visibility) in pixel coordinates.

        Returns None if no landmarks or visibility < 0.7.
        """
        raw = self.get_landmark(landmarks, index)
        if raw is None:
            return None

        x_norm, y_norm, vis = raw
        return (int(x_norm * frame_w), int(y_norm * frame_h), vis)

    @staticmethod
    def _draw_landmarks(frame, landmarks):
        h, w, _ = frame.shape
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for idx, pt in enumerate(pts):
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)

        for a, b in _POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
