import time
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 180, 0)
GRAY = (60, 60, 60)


def _overlay_rect(frame, x, y, w, h, color=(0, 0, 0), alpha=0.6):
    """Draw a semi-transparent filled rectangle on frame (in-place)."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _angle_color_knee(angle):
    if angle < 100:
        return GREEN
    if angle < 140:
        return YELLOW
    return WHITE


def _angle_color_back(angle):
    if angle > 60:
        return GREEN
    if angle < 45:
        return RED
    return WHITE


def _angle_color_sym(diff):
    if diff < 10:
        return GREEN
    if diff > 15:
        return RED
    return WHITE


def _draw_stats_panel(frame, sm, angles):
    _overlay_rect(frame, 10, 10, 300, 160)

    y0 = 40
    cv2.putText(frame, f"STATE: {sm.state}", (20, y0), FONT, 0.7, WHITE, 2)

    if angles:
        knee = angles["left_knee_angle"]
        back = angles["back_angle"]
        sym = angles["symmetry_diff"]

        cv2.putText(frame, f"KNEE ANGLE: {knee:.1f}", (20, y0 + 30),
                    FONT, 0.7, _angle_color_knee(knee), 2)
        cv2.putText(frame, f"BACK ANGLE: {back:.1f}", (20, y0 + 60),
                    FONT, 0.7, _angle_color_back(back), 2)
        cv2.putText(frame, f"SYMMETRY: {sym:.1f}", (20, y0 + 90),
                    FONT, 0.7, _angle_color_sym(sym), 2)

    cv2.putText(frame, f"GOOD: {sm.good_reps}", (20, y0 + 120),
                FONT, 0.7, GREEN, 2)
    cv2.putText(frame, f"BAD: {sm.bad_reps}", (180, y0 + 120),
                FONT, 0.7, RED, 2)


def _draw_tempo_bar(frame, sm):
    fh, fw = frame.shape[:2]
    bar_y = fh - 30
    bar_h = 20

    _overlay_rect(frame, 0, bar_y, fw, bar_h, GRAY, 0.5)
    cv2.putText(frame, "TEMPO", (10, bar_y - 6), FONT, 0.5, WHITE, 1)

    if sm.state in (sm.DESCENDING, sm.BOTTOM) and sm.descent_start_time:
        elapsed = time.time() - sm.descent_start_time
        max_time = 4.0
        ratio = min(elapsed / max_time, 1.0)
        bar_w = int(fw * ratio)

        if elapsed < 0.8:
            color = GREEN
        elif elapsed <= 3.0:
            color = BLUE
        else:
            color = RED

        cv2.rectangle(frame, (0, bar_y), (bar_w, bar_y + bar_h), color, -1)


def _draw_feedback(frame, sm):
    if not sm.last_feedback or sm.feedback_timer <= 0:
        return

    fh, fw = frame.shape[:2]
    box_w, box_h = 500, 160
    bx = (fw - box_w) // 2
    by = fh // 2 - 60

    _overlay_rect(frame, bx, by, box_w, box_h, (0, 0, 0), 0.7)

    fb = sm.last_feedback
    if fb["good"]:
        text = fb["message"]
        text_size = cv2.getTextSize(text, FONT, 1.2, 2)[0]
        tx = bx + (box_w - text_size[0]) // 2
        ty = by + (box_h + text_size[1]) // 2
        cv2.putText(frame, text, (tx, ty), FONT, 1.2, GREEN, 2)
    else:
        cv2.putText(frame, "FORM ERRORS:", (bx + 15, by + 30), FONT, 0.7, RED, 2)
        for i, (error, fix) in enumerate(fb["errors"]):
            line_y = by + 58 + i * 22
            cv2.putText(frame, f"  {error}  ->  {fix}", (bx + 15, line_y),
                        FONT, 0.55, WHITE, 1)


def draw_hud(frame, state_machine, angles):
    _draw_stats_panel(frame, state_machine, angles)
    _draw_tempo_bar(frame, state_machine)
    _draw_feedback(frame, state_machine)
