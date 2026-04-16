import time
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = cv2.FONT_HERSHEY_PLAIN

WHITE = (255, 255, 255)
GREEN = (0, 220, 100)
RED = (60, 60, 255)
YELLOW = (0, 220, 255)
CYAN = (220, 200, 50)
DARK = (30, 30, 30)
ACCENT = (235, 206, 135)
BLACK = (0, 0, 0)
BLUE_MSG = (255, 120, 0)


def _rounded_rect_solid(frame, x, y, w, h, color, radius=14):
    """Draw a filled rounded rectangle directly on the frame (no transparency)."""
    cv2.rectangle(frame, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(frame, (x, y + radius), (x + w, y + h - radius), color, -1)
    cv2.circle(frame, (x + radius, y + radius), radius, color, -1)
    cv2.circle(frame, (x + w - radius, y + radius), radius, color, -1)
    cv2.circle(frame, (x + radius, y + h - radius), radius, color, -1)
    cv2.circle(frame, (x + w - radius, y + h - radius), radius, color, -1)


def _rounded_rect(frame, x, y, w, h, color=DARK, alpha=0.7, radius=12):
    overlay = frame.copy()
    _rounded_rect_solid(overlay, x, y, w, h, color, radius)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _overlay_rect(frame, x, y, w, h, color=DARK, alpha=0.7):
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    sub = frame[y1:y2, x1:x2]
    rect = np.full_like(sub, color, dtype=np.uint8)
    cv2.addWeighted(rect, alpha, sub, 1 - alpha, 0, sub)
    frame[y1:y2, x1:x2] = sub


def _angle_color_knee(angle):
    if 60 <= angle <= 100:
        return GREEN
    if angle < 140:
        return YELLOW
    return WHITE


def _angle_color_back(angle):
    if angle <= 50:
        return GREEN
    if angle <= 65:
        return YELLOW
    return RED


def _angle_color_sym(diff):
    if diff < 10:
        return GREEN
    if diff > 20:
        return RED
    return YELLOW


def _state_color(state):
    if state in ("S3", "BOTTOM"):
        return GREEN
    if state in ("S2", "DESCENDING", "ASCENDING"):
        return YELLOW
    return WHITE


# ─── Rep counters: top-left (incorrect) & top-right (correct) ────────────────

def _draw_rep_counters(frame, sm):
    _, fw = frame.shape[:2]
    box_w, box_h = 240, 42
    margin = 14
    radius = 10

    lx, ly = margin, margin
    _rounded_rect_solid(frame, lx, ly, box_w, box_h, RED, radius)
    cv2.putText(frame, f"X INCORRECT: {sm.bad_reps}", (lx + 14, ly + 28),
                FONT, 0.7, WHITE, 2)

    rx, ry = fw - box_w - margin, margin
    _rounded_rect_solid(frame, rx, ry, box_w, box_h, GREEN, radius)
    cv2.putText(frame, f"+ CORRECT: {sm.good_reps}", (rx + 14, ry + 28),
                FONT, 0.7, WHITE, 2)


# ─── Stats panel: bottom-left ────────────────────────────────────────────────

def _draw_stats_panel(frame, sm, angles):
    fh, fw = frame.shape[:2]
    panel_w, panel_h = 260, 150
    px, py = 10, fh - panel_h - 14

    _rounded_rect(frame, px, py, panel_w, panel_h, DARK, 0.75, 10)

    x0 = px + 14
    y0 = py + 24

    state_col = _state_color(sm.state)
    cv2.putText(frame, sm.state, (x0, y0), FONT, 0.6, state_col, 2)

    view_label = sm.current_view.upper()
    lock_label = "LOCK" if sm.view_locked else "AUTO"
    view_conf = int(sm.view_confidence * 100)
    cv2.putText(frame, f"{view_label} {view_conf}% {lock_label}",
                (x0 + 90, y0), FONT_SMALL, 1.0, CYAN, 1)

    cv2.line(frame, (x0, y0 + 8), (x0 + panel_w - 28, y0 + 8), (60, 60, 60), 1)

    if angles:
        knee = angles["knee_angle"]
        back = angles["back_angle"]

        cv2.putText(frame, "KNEE", (x0, y0 + 34), FONT_SMALL, 1.0, ACCENT, 1)
        cv2.putText(frame, f"{knee:.0f}", (x0 + 70, y0 + 34),
                    FONT, 0.55, _angle_color_knee(knee), 2)

        cv2.putText(frame, "BACK", (x0, y0 + 56), FONT_SMALL, 1.0, ACCENT, 1)
        cv2.putText(frame, f"{back:.0f}", (x0 + 70, y0 + 56),
                    FONT, 0.55, _angle_color_back(back), 2)

        if angles["view"] == "front":
            sym = angles["symmetry_diff"]
            cv2.putText(frame, "SYM", (x0, y0 + 78), FONT_SMALL, 1.0, ACCENT, 1)
            cv2.putText(frame, f"{sym:.0f}", (x0 + 70, y0 + 78),
                        FONT, 0.55, _angle_color_sym(sym), 2)
        else:
            label = "BACK VIEW" if angles["view"] == "back" else "SIDE VIEW"
            cv2.putText(frame, label, (x0, y0 + 78), FONT_SMALL, 1.0, CYAN, 1)

    cv2.line(frame, (x0, y0 + 90), (x0 + panel_w - 28, y0 + 90), (60, 60, 60), 1)

    status_text = "SET ACTIVE" if sm.set_active else "WAITING"
    status_col = CYAN if sm.set_active else YELLOW
    cv2.putText(frame, status_text, (x0, y0 + 112), FONT_SMALL, 1.1, status_col, 1)


# ─── Tempo bar: bottom edge ──────────────────────────────────────────────────

def _draw_tempo_bar(frame, sm):
    fh, fw = frame.shape[:2]
    bar_h = 6
    bar_y = fh - bar_h - 4

    elapsed = 0.0
    active = sm.state not in (sm.STANDING, "S1")

    if active and sm.descent_start_time:
        elapsed = time.time() - sm.descent_start_time

    max_time = 6.0
    ratio = min(elapsed / max_time, 1.0) if active else 0.0
    bar_w = int(fw * ratio)

    _overlay_rect(frame, 0, bar_y, fw, bar_h, (50, 50, 50), 0.5)

    if bar_w > 0:
        if elapsed < 0.6:
            color = YELLOW
        elif elapsed <= 4.0:
            color = GREEN
        else:
            color = RED
        cv2.rectangle(frame, (0, bar_y), (bar_w, bar_y + bar_h), color, -1)

    if active and elapsed > 0:
        label = f"{elapsed:.1f}s"
        cv2.putText(frame, label, (8, bar_y - 6), FONT_SMALL, 1.0, WHITE, 1)


# ─── Feedback popups: center (good) / right (bad) ────────────────────────────

def _draw_feedback(frame, sm):
    if not sm.last_feedback or not sm.feedback_active:
        return

    elapsed = time.time() - sm.feedback_time
    fade_start = sm.FEEDBACK_DURATION - 1.0
    if elapsed > fade_start:
        alpha = max(0.0, 1.0 - (elapsed - fade_start) / 1.0)
    else:
        alpha = 1.0

    fh, fw = frame.shape[:2]
    fb = sm.last_feedback

    if fb["good"]:
        text = fb.get("message", "Good rep, keep it up...")
        text_size = cv2.getTextSize(text, FONT, 0.7, 2)[0]
        box_w = max(240, text_size[0] + 36)
        box_h = text_size[1] + 24
        right_counter_x = fw - 240 - 14
        bx = right_counter_x + (240 - box_w) // 2
        by = 62

        overlay = frame.copy()
        _rounded_rect_solid(overlay, bx, by, box_w, box_h, BLUE_MSG, 10)
        cv2.addWeighted(overlay, 0.85 * alpha, frame, 1 - 0.85 * alpha, 0, frame)

        col = tuple(int(c * alpha) for c in WHITE)
        tx = bx + (box_w - text_size[0]) // 2
        ty = by + (box_h + text_size[1]) // 2 - 3
        cv2.putText(frame, text, (tx, ty), FONT, 0.7, col, 2)
    else:
        errs = fb["errors"][:3]
        n_errors = len(errs)
        line_font = 0.48
        line_h = 24
        max_text_w = 0
        for error, fix in errs:
            line = f"- {error}: {fix}"
            tw = cv2.getTextSize(line, FONT, line_font, 1)[0][0]
            max_text_w = max(max_text_w, tw)
        box_w = min(max(240, max_text_w + 24), 440)
        box_h = 16 + n_errors * line_h
        bx = 14
        by = 62

        overlay = frame.copy()
        _rounded_rect_solid(overlay, bx, by, box_w, box_h, YELLOW, 10)
        cv2.addWeighted(overlay, 0.85 * alpha, frame, 1 - 0.85 * alpha, 0, frame)

        col_t = tuple(int(c * alpha) for c in BLACK)

        for i, (error, fix) in enumerate(errs):
            line_y = by + 18 + i * line_h
            cv2.putText(frame, f"- {error}: {fix}", (bx + 8, line_y),
                        FONT, line_font, col_t, 1)


# ─── Controls panel: bottom-right ────────────────────────────────────────────

def _draw_controls(frame, sm):
    fh, fw = frame.shape[:2]

    panel_w, panel_h = 280, 132
    px = fw - panel_w - 10
    py = fh - panel_h - 14

    _rounded_rect(frame, px, py, panel_w, panel_h, DARK, 0.78, 10)

    x0 = px + 14
    y0 = py + 24

    cv2.putText(frame, "Mode Keys", (x0, y0), FONT, 0.5, CYAN, 1)
    active = sm.mode_override
    auto_col = GREEN if active == "auto" else WHITE
    front_col = GREEN if active == "front" else WHITE
    side_col = GREEN if active == "side" else WHITE
    back_col = GREEN if active == "back" else WHITE
    cv2.putText(frame, "A-Auto", (x0, y0 + 20), FONT_SMALL, 1.0, auto_col, 1)
    cv2.putText(frame, "F-Front", (x0 + 108, y0 + 20), FONT_SMALL, 1.0, front_col, 1)
    cv2.putText(frame, "S-Side", (x0, y0 + 38), FONT_SMALL, 1.0, side_col, 1)
    cv2.putText(frame, "B-Back", (x0 + 108, y0 + 38), FONT_SMALL, 1.0, back_col, 1)

    mode_name = sm.mode_override.upper() if sm.mode_override != "auto" else "AUTO"
    cv2.putText(frame, f"Current: {mode_name}", (x0, y0 + 60), FONT_SMALL, 1.0, CYAN, 1)

    btn_w, btn_h = panel_w - 28, 34
    bx = x0
    by = py + panel_h - btn_h - 10

    if sm.set_active:
        btn_color = RED
        btn_text = "END SET"
    else:
        btn_color = GREEN
        btn_text = "START SET"

    _rounded_rect_solid(frame, bx, by, btn_w, btn_h, btn_color, 8)
    text_size = cv2.getTextSize(btn_text, FONT, 0.62, 2)[0]
    tx = bx + (btn_w - text_size[0]) // 2
    ty = by + (btn_h + text_size[1]) // 2 - 3
    cv2.putText(frame, btn_text, (tx, ty), FONT, 0.62, WHITE, 2)

    return {"start_set_button": (bx, by, btn_w, btn_h)}


# ─── Main draw entry point ───────────────────────────────────────────────────

def draw_hud(frame, state_machine, angles):
    _draw_rep_counters(frame, state_machine)
    _draw_stats_panel(frame, state_machine, angles)
    _draw_tempo_bar(frame, state_machine)
    _draw_feedback(frame, state_machine)
    return _draw_controls(frame, state_machine)
