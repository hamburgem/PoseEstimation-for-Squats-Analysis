import numpy as np


def calculate_angle(a: tuple, b: tuple, c: tuple) -> float:
    """Compute the angle at vertex b formed by points a -> b -> c.

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


_LEFT_LANDMARKS = {
    "shoulder": 11, "hip": 23, "knee": 25, "ankle": 27,
}
_RIGHT_LANDMARKS = {
    "shoulder": 12, "hip": 24, "knee": 26, "ankle": 28,
}
_NOSE_IDX = 0
# Heel / foot (optional, lower visibility threshold)
_LEFT_FOOT_IDX, _RIGHT_FOOT_IDX = 31, 32


def _lm_px_relaxed(landmarks, idx: int, frame_w: int, frame_h: int, vis_min: float = 0.35):
    if not landmarks or idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    if lm.visibility < vis_min:
        return None
    return (int(lm.x * frame_w), int(lm.y * frame_h), lm.visibility)


def _segment_angle_deg(p1, p2):
    """Angle of p1->p2 from horizontal, degrees [-180, 180]."""
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    return float(np.degrees(np.arctan2(dy, dx)))


def _get_side_pts(detector, landmarks, side_map, frame_w, frame_h):
    """Try to fetch all 4 landmarks for one side. Returns dict or None."""
    pts = {}
    vis_sum = 0.0
    for name, idx in side_map.items():
        lm = detector.get_landmark_px(landmarks, idx, frame_w, frame_h)
        if lm is None:
            return None
        pts[name] = (lm[0], lm[1])
        vis_sum += lm[2]
    pts["_avg_vis"] = vis_sum / len(side_map)
    return pts


def _angle_with_vertical(p1, p2):
    """Return 0..90 angle of segment p1->p2 against vertical axis."""
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-9)))


def _offset_angle_from_nose(left_shoulder, right_shoulder, nose_pt):
    """Angle between nose->left_shoulder and nose->right_shoulder vectors."""
    if nose_pt is None:
        return 0.0
    v1 = np.array(left_shoulder, dtype=float) - np.array(nose_pt, dtype=float)
    v2 = np.array(right_shoulder, dtype=float) - np.array(nose_pt, dtype=float)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    cos_theta = float(np.dot(v1, v2) / denom)
    theta = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return float(theta)


def _pick_profile_side(left, right):
    """Choose the visible profile leg for side view — avoids hallucinated far leg."""
    if left is not None and right is None:
        return "left", left
    if right is not None and left is None:
        return "right", right
    if left is None and right is None:
        return None, None
    vis_l = left["_avg_vis"]
    vis_r = right["_avg_vis"]
    if vis_l > vis_r + 0.06:
        return "left", left
    if vis_r > vis_l + 0.06:
        return "right", right
    dist_l = abs(left["ankle"][1] - left["shoulder"][1])
    dist_r = abs(right["ankle"][1] - right["shoulder"][1])
    if dist_l >= dist_r:
        return "left", left
    return "right", right


def get_squat_angles(
    detector,
    landmarks,
    frame_w: int,
    frame_h: int,
    mode_override: str = "auto",
):
    """Extract key squat angles from the current pose.

    For **side** view (auto-detected or mode S), all leg metrics use a single
    profile side only so the occluded leg does not corrupt angles.

    mode_override: "auto" | "front" | "side" | "back"
    """
    left = _get_side_pts(detector, landmarks, _LEFT_LANDMARKS, frame_w, frame_h)
    right = _get_side_pts(detector, landmarks, _RIGHT_LANDMARKS, frame_w, frame_h)

    if left is None and right is None:
        return None

    def _knee_angle(pts):
        return calculate_angle(pts["hip"], pts["knee"], pts["ankle"])

    def _back_angle(pts):
        """Trunk lean: angle of shoulder->hip segment vs vertical (0 = upright)."""
        return _angle_with_vertical(pts["shoulder"], pts["hip"])

    profile_side, profile_pts = _pick_profile_side(left, right)

    # Geometry-based view scores (only meaningful when both shoulders exist)
    view_scores = {"front": 0.0, "back": 0.0, "side": 0.0}
    view_candidate = "side"

    if left is not None and right is not None:
        lk = _knee_angle(left)
        rk = _knee_angle(right)
        symmetry_diff = abs(lk - rk)
        left_shoulder_x = left["shoulder"][0]
        right_shoulder_x = right["shoulder"][0]
        shoulder_width_ratio = abs(left_shoulder_x - right_shoulder_x) / max(frame_w, 1)

        frontal_strength = float(np.clip(shoulder_width_ratio / 0.12, 0.0, 1.0))
        side_strength = 1.0 - frontal_strength

        front_hint = 1.0 if left_shoulder_x > right_shoulder_x else 0.0
        back_hint = 1.0 - front_hint
        blend = 0.55 + 0.45 * frontal_strength

        view_scores = {
            "front": blend * front_hint,
            "back": blend * back_hint,
            "side": 0.35 * side_strength,
        }
        view_candidate = max(view_scores, key=view_scores.get)
    elif left is not None:
        symmetry_diff = 0.0
        side_conf = 0.8 + 0.2 * left["_avg_vis"]
        view_scores = {"front": 0.0, "back": 0.0, "side": float(np.clip(side_conf, 0.0, 1.0))}
        view_candidate = "side"
    else:
        symmetry_diff = 0.0
        side_conf = 0.8 + 0.2 * right["_avg_vis"]
        view_scores = {"front": 0.0, "back": 0.0, "side": float(np.clip(side_conf, 0.0, 1.0))}
        view_candidate = "side"

    # Effective view respects manual mode
    if mode_override == "side":
        effective_view = "side"
    elif mode_override in ("front", "back"):
        effective_view = mode_override
    else:
        effective_view = view_candidate

    nose_lm = detector.get_landmark_px(landmarks, _NOSE_IDX, frame_w, frame_h)
    nose_pt = (nose_lm[0], nose_lm[1]) if nose_lm is not None else None
    if left is not None and right is not None:
        offset_angle = _offset_angle_from_nose(left["shoulder"], right["shoulder"], nose_pt)
    else:
        offset_angle = 0.0

    # --- Metrics: side / profile uses ONE leg only ---
    if effective_view == "side":
        if profile_pts is None:
            return None
        knee_angle = _knee_angle(profile_pts)
        back_angle = _back_angle(profile_pts)
        knee_x = profile_pts["knee"][0]
        toe_x = profile_pts["ankle"][0]

        hip_vertical_angle = _angle_with_vertical(profile_pts["shoulder"], profile_pts["hip"])
        knee_vertical_angle = _angle_with_vertical(profile_pts["hip"], profile_pts["knee"])
        ankle_vertical_angle = _angle_with_vertical(profile_pts["knee"], profile_pts["ankle"])
    elif left is not None and right is not None:
        lk = _knee_angle(left)
        rk = _knee_angle(right)
        knee_angle = (lk + rk) / 2.0
        back_angle = (_back_angle(left) + _back_angle(right)) / 2.0
        knee_x = (left["knee"][0] + right["knee"][0]) / 2.0
        toe_x = (left["ankle"][0] + right["ankle"][0]) / 2.0

        hip_vertical_angle = 0.0
        knee_vertical_angle = 0.0
        ankle_vertical_angle = 0.0
    else:
        solo = left if left is not None else right
        knee_angle = _knee_angle(solo)
        back_angle = _back_angle(solo)
        knee_x = solo["knee"][0]
        toe_x = solo["ankle"][0]
        hip_vertical_angle = 0.0
        knee_vertical_angle = 0.0
        ankle_vertical_angle = 0.0

    is_frontal = effective_view in ("front", "back")
    if is_frontal and left is not None and right is not None:
        knee_vertical_angle = max(0.0, 180.0 - knee_angle)
        hip_vertical_angle = back_angle
        ankle_vertical_angle = 0.0

    depth_metric = max(0.0, 180.0 - knee_angle)

    # Front/back: knee_angle is unreliable in 2D projection (foreshortened).
    # Use hip-drop ratio: how far hips dropped relative to total body height.
    # Standing ≈ 0%, full squat ≈ 40-55%. Map to 0-90° scale for state machine.
    hip_drop_metric = None
    if is_frontal and left is not None and right is not None:
        shoulder_mid_y = (left["shoulder"][1] + right["shoulder"][1]) / 2.0
        hip_mid_y = (left["hip"][1] + right["hip"][1]) / 2.0
        ankle_mid_y = (left["ankle"][1] + right["ankle"][1]) / 2.0
        body_height = max(ankle_mid_y - shoulder_mid_y, 1.0)
        hip_norm_pos = (hip_mid_y - shoulder_mid_y) / body_height
        # Standing: hip_norm_pos ≈ 0.45-0.55. Full squat: ≈ 0.75-0.90.
        # Map [0.50, 0.90] → [0°, 90°] for state thresholds.
        drop_ratio = max(0.0, (hip_norm_pos - 0.50) / 0.40)
        hip_drop_metric = float(min(drop_ratio * 90.0, 95.0))

    # --- Side-view extras (single profile leg) ---
    knee_forward_norm = 0.0
    hip_below_knee = False
    if effective_view == "side" and profile_pts is not None:
        knee_forward_norm = abs(float(profile_pts["knee"][0] - profile_pts["ankle"][0])) / max(
            frame_w, 1
        )
        # Image y increases downward; hip below knee ≈ butt-wink proxy at bottom
        hip_below_knee = bool(profile_pts["hip"][1] > profile_pts["knee"][1] + 6.0)

    # --- Front / back extras (bilateral) ---
    valgus_left_norm = 0.0
    valgus_right_norm = 0.0
    valgus_max = 0.0
    hip_height_asym_norm = 0.0
    toe_flare_asym_deg = 0.0
    ankle_y_left_norm = None
    ankle_y_right_norm = None
    hip_mid_x_norm = None

    if is_frontal and left is not None and right is not None:
        hip_mid_x_norm = float(left["hip"][0] + right["hip"][0]) / (2.0 * max(frame_w, 1))
        # Facing camera: cave-in = knee moves toward midline vs ankle (per-leg heuristic)
        valgus_left_norm = float(left["ankle"][0] - left["knee"][0]) / max(frame_w, 1)
        valgus_right_norm = float(right["knee"][0] - right["ankle"][0]) / max(frame_w, 1)
        valgus_max = max(max(0.0, valgus_left_norm), max(0.0, valgus_right_norm))

        hip_height_asym_norm = abs(float(left["hip"][1] - right["hip"][1])) / max(frame_h, 1)

        lf = _lm_px_relaxed(landmarks, _LEFT_FOOT_IDX, frame_w, frame_h)
        rf = _lm_px_relaxed(landmarks, _RIGHT_FOOT_IDX, frame_w, frame_h)
        la = _lm_px_relaxed(landmarks, 27, frame_w, frame_h)
        ra = _lm_px_relaxed(landmarks, 28, frame_w, frame_h)
        if la and lf:
            a_l = _segment_angle_deg((la[0], la[1]), (lf[0], lf[1]))
        else:
            a_l = None
        if ra and rf:
            a_r = _segment_angle_deg((ra[0], ra[1]), (rf[0], rf[1]))
        else:
            a_r = None
        if a_l is not None and a_r is not None:
            toe_flare_asym_deg = abs(a_l - a_r)
            if toe_flare_asym_deg > 180.0:
                toe_flare_asym_deg = 360.0 - toe_flare_asym_deg

        if la:
            ankle_y_left_norm = float(la[1]) / max(frame_h, 1)
        if ra:
            ankle_y_right_norm = float(ra[1]) / max(frame_h, 1)

    return {
        "knee_angle": knee_angle,
        "back_angle": back_angle,
        "symmetry_diff": symmetry_diff if left is not None and right is not None else 0.0,
        "knee_x": knee_x,
        "toe_x": toe_x,
        "view_candidate": view_candidate,
        "view_scores": view_scores,
        "view": effective_view,
        "effective_view": effective_view,
        "profile_side": profile_side,
        "offset_angle": offset_angle,
        "hip_vertical_angle": hip_vertical_angle,
        "knee_vertical_angle": knee_vertical_angle,
        "ankle_vertical_angle": ankle_vertical_angle,
        "depth_metric": depth_metric,
        # Counting / feedback (optional)
        "knee_forward_norm": knee_forward_norm,
        "hip_below_knee": hip_below_knee,
        "valgus_left_norm": valgus_left_norm,
        "valgus_right_norm": valgus_right_norm,
        "valgus_max": valgus_max,
        "hip_height_asym_norm": hip_height_asym_norm,
        "toe_flare_asym_deg": toe_flare_asym_deg,
        "ankle_y_left_norm": ankle_y_left_norm,
        "ankle_y_right_norm": ankle_y_right_norm,
        "hip_mid_x_norm": hip_mid_x_norm,
        "hip_drop_metric": hip_drop_metric,
    }
