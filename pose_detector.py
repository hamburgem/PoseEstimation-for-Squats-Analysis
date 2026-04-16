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

# Full-body set (front / back / auto when not profile)
_RELEVANT_LANDMARKS_FULL = {11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}

_POSE_CONNECTIONS_FULL = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

# One side only: arm + torso to hip + leg (matches side-view reference — no far leg)
_LEFT_PROFILE_IDX = {11, 13, 15, 23, 25, 27}
_RIGHT_PROFILE_IDX = {12, 14, 16, 24, 26, 28}
_POSE_CONNECTIONS_LEFT_PROFILE = [
    (11, 13), (13, 15),
    (11, 23), (23, 25), (25, 27),
]
_POSE_CONNECTIONS_RIGHT_PROFILE = [
    (12, 14), (14, 16),
    (12, 24), (24, 26), (26, 28),
]

_SKELETON_COLOR = (235, 206, 135)  # light blue (BGR)
_KEYPOINT_COLOR = (0, 0, 255)  # red (BGR)


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

    def detect(self, frame):
        """Run pose detection only; does not draw. Returns (frame, landmarks list)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        landmarks = result.pose_landmarks[0] if result.pose_landmarks else []
        return frame, landmarks

    def process(self, frame, draw_view="full", profile_side=None):
        """Detect and optionally draw. Kept for compatibility; prefer detect + draw_landmarks."""
        frame, landmarks = self.detect(frame)
        if landmarks:
            self.draw_landmarks(frame, landmarks, draw_view=draw_view, profile_side=profile_side)
        return frame, landmarks

    def get_landmark(self, landmarks, index: int):
        """Return (x, y, visibility) in normalised coordinates.

        Returns None if no landmarks or visibility < 0.5.
        """
        if not landmarks:
            return None

        lm = landmarks[index]
        if lm.visibility < 0.5:
            return None

        return (lm.x, lm.y, lm.visibility)

    def get_landmark_px(self, landmarks, index: int, frame_w: int, frame_h: int):
        """Return (x, y, visibility) in pixel coordinates.

        Returns None if no landmarks or visibility < 0.5.
        """
        raw = self.get_landmark(landmarks, index)
        if raw is None:
            return None

        x_norm, y_norm, vis = raw
        return (int(x_norm * frame_w), int(y_norm * frame_h), vis)

    def draw_landmarks(self, frame, landmarks, draw_view="full", profile_side=None):
        """draw_view: 'full' | 'side' — side draws only one profile chain."""
        if not landmarks:
            return
        if draw_view == "side" and profile_side in ("left", "right"):
            self._draw_profile_side(frame, landmarks, profile_side)
        else:
            self._draw_full(frame, landmarks)

    @staticmethod
    def _draw_full(frame, landmarks):
        h, w, _ = frame.shape
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for a, b in _POSE_CONNECTIONS_FULL:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], _SKELETON_COLOR, 3)

        for idx in _RELEVANT_LANDMARKS_FULL:
            if idx < len(pts):
                cv2.circle(frame, pts[idx], 8, _KEYPOINT_COLOR, -1)

    @staticmethod
    def _draw_profile_side(frame, landmarks, profile_side: str):
        h, w, _ = frame.shape
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        if profile_side == "left":
            conns = _POSE_CONNECTIONS_LEFT_PROFILE
            idx_set = _LEFT_PROFILE_IDX
        else:
            conns = _POSE_CONNECTIONS_RIGHT_PROFILE
            idx_set = _RIGHT_PROFILE_IDX

        for a, b in conns:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], _SKELETON_COLOR, 3)

        for idx in idx_set:
            if idx < len(pts):
                cv2.circle(frame, pts[idx], 8, _KEYPOINT_COLOR, -1)
