import numpy as np


def calculate_angle(a: tuple, b: tuple, c: tuple) -> float:
    """Compute the angle at vertex b formed by points a → b → c.

    Parameters are (x, y) pixel coordinates.  Returns degrees in [0, 180].
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    angle = np.degrees(np.arctan2(
        np.sqrt(1 - cosine**2),
        cosine,
    ))

    return float(np.clip(angle, 0.0, 180.0))


_REQUIRED_LANDMARKS = {
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_shoulder": 11,
    "right_shoulder": 12,
}


def get_squat_angles(detector, landmarks, frame_w: int, frame_h: int):
    """Extract key squat angles from the current pose.

    Returns a dict with angle data, or None if any required landmark
    is missing (visibility < 0.7).
    """
    pts = {}
    for name, idx in _REQUIRED_LANDMARKS.items():
        lm = detector.get_landmark_px(landmarks, idx, frame_w, frame_h)
        if lm is None:
            return None
        pts[name] = (lm[0], lm[1])

    left_knee_angle = calculate_angle(
        pts["left_hip"], pts["left_knee"], pts["left_ankle"],
    )
    right_knee_angle = calculate_angle(
        pts["right_hip"], pts["right_knee"], pts["right_ankle"],
    )
    back_angle = calculate_angle(
        pts["left_shoulder"], pts["left_hip"], pts["left_knee"],
    )

    return {
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "back_angle": back_angle,
        "symmetry_diff": abs(left_knee_angle - right_knee_angle),
        "knee_x": pts["left_knee"][0],
        "toe_x": pts["left_ankle"][0],
        "knee_angle_min": 180.0,
    }
