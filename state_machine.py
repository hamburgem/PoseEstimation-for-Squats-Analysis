import time
import random


class SquatStateMachine:
    STANDING = "STANDING"
    DESCENDING = "DESCENDING"
    BOTTOM = "BOTTOM"
    ASCENDING = "ASCENDING"

    FEEDBACK_DURATION = 10.0
    VIEW_SWITCH_MARGIN = 0.08
    VIEW_SWITCH_STREAK = 3

    _BEGINNER_THRESHOLDS = {
        "STATE_THRESH": {"s1_max": 32.0, "s2_min": 35.0, "s2_max": 65.0, "s3_min": 70.0, "s3_max": 95.0},
        "FEEDBACK_THRESH": {
            "hip_low": 10.0, "hip_high": 50.0,
            "knee_mid_low": 50.0, "knee_mid_high": 70.0,
            "knee_deep_high": 95.0, "ankle_high": 58.0,
        },
        "REP_RULES": {
            "shallow_knee_deg": 105.0,
            "torso_back_max_deg": 60.0,
            "knee_forward_norm": 0.13,
            "descent_fast_s": 0.25,
            "ascent_slow_s": 7.0,
            "valgus_norm": 0.065,
            "hip_height_asym": 0.060,
            "heel_rise_norm": 0.038,
            "lateral_drift": 0.085,
            "toe_flare_asym_deg": 30.0,
        },
        "OFFSET_THRESH": 35.0,
        "INACTIVE_THRESH": 15.0,
    }
    _PRO_THRESHOLDS = {
        "STATE_THRESH": {"s1_max": 32.0, "s2_min": 35.0, "s2_max": 65.0, "s3_min": 80.0, "s3_max": 95.0},
        "FEEDBACK_THRESH": {
            "hip_low": 15.0, "hip_high": 50.0,
            "knee_mid_low": 50.0, "knee_mid_high": 80.0,
            "knee_deep_high": 95.0, "ankle_high": 52.0,
        },
        "REP_RULES": {
            "shallow_knee_deg": 100.0,
            "torso_back_max_deg": 55.0,
            "knee_forward_norm": 0.12,
            "descent_fast_s": 0.22,
            "ascent_slow_s": 6.0,
            "valgus_norm": 0.058,
            "hip_height_asym": 0.050,
            "heel_rise_norm": 0.032,
            "lateral_drift": 0.075,
            "toe_flare_asym_deg": 25.0,
        },
        "OFFSET_THRESH": 35.0,
        "INACTIVE_THRESH": 15.0,
    }

    def __init__(self):
        self.state = self.STANDING
        self.good_reps = 0
        self.bad_reps = 0
        self.descent_start_time = None
        self.ascent_start_time = None
        self.knee_angle_min = 180.0
        self.last_feedback = None
        self.feedback_time = None
        self.current_view = "side"
        self.view_confidence = 0.0
        self.view_locked = False
        self.mode_override = "auto"
        self.set_active = False
        self._candidate_streak = 0
        self._last_candidate = None
        self.state_sequence = []
        self.prev_state = "s1"
        self.current_state = "s1"
        self.inactive_since = None
        self.incorrect_posture = False
        self.rep_errors = []
        self.skill_level = "beginner"
        self._t_first_s3 = None
        self._t_ascent_start = None
        self._rep_knee_angle_min = 180.0
        self._rep_back_angle_max = 0.0
        self._rep_butt_wink = False
        self._rep_knee_forward_max = 0.0
        self._rep_max_valgus = 0.0
        self._rep_max_hip_asym = 0.0
        self._rep_max_heel_rise = 0.0
        self._rep_max_lateral_drift = 0.0
        self._rep_max_toe_asym = 0.0
        self._rep_hip_mid_x0 = None
        self._rep_ankle_y_avg0 = None
        self._apply_skill_thresholds(self.skill_level)

    def _apply_skill_thresholds(self, level: str):
        cfg = self._BEGINNER_THRESHOLDS if level == "beginner" else self._PRO_THRESHOLDS
        self.STATE_THRESH = dict(cfg["STATE_THRESH"])
        self.FEEDBACK_THRESH = dict(cfg["FEEDBACK_THRESH"])
        self.REP_RULES = dict(cfg["REP_RULES"])
        self.OFFSET_THRESH = float(cfg["OFFSET_THRESH"])
        self.INACTIVE_THRESH = float(cfg["INACTIVE_THRESH"])

    def set_skill_level(self, level: str):
        if level not in ("beginner", "pro"):
            return
        self.skill_level = level
        self._apply_skill_thresholds(level)

    @property
    def feedback_active(self):
        if self.feedback_time is None:
            return False
        return (time.time() - self.feedback_time) < self.FEEDBACK_DURATION

    def _update_view(self, angles: dict):
        """Stabilize auto-view detection with hysteresis and streak switching."""
        if self.mode_override != "auto":
            self.current_view = self.mode_override
            self.view_confidence = 1.0
            angles["view"] = self.current_view
            angles["view_confidence"] = self.view_confidence
            return

        scores = angles.get("view_scores") or {}
        if not scores:
            return

        candidate = max(scores, key=scores.get)
        candidate_conf = float(scores[candidate])

        if self.view_locked:
            angles["view"] = self.current_view
            angles["view_confidence"] = self.view_confidence
            return

        if self.current_view is None:
            self.current_view = candidate
            self.view_confidence = candidate_conf
            self._candidate_streak = 0
            self._last_candidate = candidate
        elif candidate == self.current_view:
            self.view_confidence = 0.75 * self.view_confidence + 0.25 * candidate_conf
            self._candidate_streak = 0
            self._last_candidate = candidate
        else:
            if self._last_candidate == candidate:
                self._candidate_streak += 1
            else:
                self._last_candidate = candidate
                self._candidate_streak = 1

            if (
                self._candidate_streak >= self.VIEW_SWITCH_STREAK
                and candidate_conf > (self.view_confidence + self.VIEW_SWITCH_MARGIN)
            ):
                self.current_view = candidate
                self.view_confidence = candidate_conf
                self._candidate_streak = 0
            else:
                self.view_confidence = 0.85 * self.view_confidence + 0.15 * scores.get(
                    self.current_view, 0.0
                )

        angles["view"] = self.current_view
        angles["view_confidence"] = self.view_confidence

    def set_mode(self, mode: str):
        if mode not in ("auto", "front", "side", "back"):
            return
        self.mode_override = mode
        if mode != "auto":
            self.current_view = mode
            self.view_confidence = 1.0
            self._candidate_streak = 0
            self._last_candidate = mode

    def start_set(self):
        self.set_active = True
        self.state = self.STANDING
        self.good_reps = 0
        self.bad_reps = 0
        self.descent_start_time = None
        self.ascent_start_time = None
        self.knee_angle_min = 180.0
        self.last_feedback = None
        self.feedback_time = None
        self.view_locked = False
        self.state_sequence = []
        self.prev_state = "s1"
        self.current_state = "s1"
        self.inactive_since = None
        self.incorrect_posture = False
        self.rep_errors = []
        self._reset_rep_metrics()

    def end_set(self):
        self._finalize_rep_if_needed()
        self.set_active = False
        self.state = self.STANDING
        self.state_sequence = []
        self.prev_state = "s1"
        self.current_state = "s1"
        self.descent_start_time = None
        self.inactive_since = None
        self.incorrect_posture = False
        self.rep_errors = []
        self._reset_rep_metrics()

    def _reset_rep_metrics(self):
        self._t_first_s3 = None
        self._t_ascent_start = None
        self._rep_knee_angle_min = 180.0
        self._rep_back_angle_max = 0.0
        self._rep_butt_wink = False
        self._rep_knee_forward_max = 0.0
        self._rep_max_valgus = 0.0
        self._rep_max_hip_asym = 0.0
        self._rep_max_heel_rise = 0.0
        self._rep_max_lateral_drift = 0.0
        self._rep_max_toe_asym = 0.0
        self._rep_hip_mid_x0 = None
        self._rep_ankle_y_avg0 = None

    def _set_feedback(self, good: bool, message=None, errors=None):
        if good:
            self.last_feedback = {
                "good": True,
                "message": message or random.choice([
                    "Nice one!", "Great job!", "Perfect rep!", "Keep it up!",
                ]),
            }
        else:
            self.last_feedback = {"good": False, "errors": errors or [("Incorrect rep", "Adjust form")]}
        self.feedback_time = time.time()

    def _state_from_knee_vertical(self, angle):
        if angle <= self.STATE_THRESH["s1_max"]:
            return "s1"
        if self.STATE_THRESH["s2_min"] <= angle <= self.STATE_THRESH["s2_max"]:
            return "s2"
        if self.STATE_THRESH["s3_min"] <= angle <= self.STATE_THRESH["s3_max"]:
            return "s3"
        return None

    @staticmethod
    def _frontal_depth_metric(knee_angle: float) -> float:
        """Map frontal knee joint angle into squat-depth style metric."""
        return max(0.0, 180.0 - float(knee_angle))

    def _depth_metric_for_view(self, angles: dict):
        """Pick the right depth signal depending on camera angle."""
        if self.current_view in ("front", "back"):
            hdm = angles.get("hip_drop_metric")
            if hdm is not None:
                return float(hdm)
        dm = angles.get("depth_metric")
        if dm is not None:
            return float(dm)
        knee = angles.get("knee_angle")
        if knee is None:
            return None
        return self._frontal_depth_metric(knee)

    def _on_inactive_tick(self):
        now = time.time()
        if self.inactive_since is None:
            self.inactive_since = now
            return
        if (now - self.inactive_since) > self.INACTIVE_THRESH:
            self.good_reps = 0
            self.bad_reps = 0
            self.state_sequence = []
            self.current_state = "s1"
            self.prev_state = "s1"
            self.state = self.STANDING
            self.incorrect_posture = False
            self.inactive_since = now

    def on_no_detection(self):
        self._on_inactive_tick()

    def update(self, angles: dict):
        self._update_view(angles)
        if not self.set_active:
            return
        offset = angles.get("offset_angle", 0.0)
        is_side_mode = self.current_view == "side"
        both_shoulders_visible = angles.get("view_candidate") in ("front", "back")
        strict_side_mode = self.mode_override == "side"
        if strict_side_mode and is_side_mode and both_shoulders_visible and offset > self.OFFSET_THRESH:
            self._on_inactive_tick()
            return

        depth_metric = self._depth_metric_for_view(angles)
        if depth_metric is None:
            self._on_inactive_tick()
            return

        current = self._state_from_knee_vertical(depth_metric)
        if current is None:
            # Out of defined bins -> keep previous state without counting.
            return

        self.current_state = current
        self.state = current.upper()

        if current != self.prev_state:
            self.inactive_since = None
        else:
            self._on_inactive_tick()

        if current in ("s2", "s3"):
            if self.prev_state == "s1" and current == "s2":
                self._reset_rep_metrics()
                self.descent_start_time = time.time()
                self.incorrect_posture = False
                self.rep_errors = []
                aln = angles.get("ankle_y_left_norm")
                arn = angles.get("ankle_y_right_norm")
                if aln is not None and arn is not None:
                    self._rep_ankle_y_avg0 = (float(aln) + float(arn)) / 2.0
                if angles.get("hip_mid_x_norm") is not None:
                    self._rep_hip_mid_x0 = float(angles["hip_mid_x_norm"])
            if self.prev_state in ("s1", "s2") and current == "s3" and self._t_first_s3 is None:
                self._t_first_s3 = time.time()
            if self.prev_state == "s3" and current == "s2":
                self._t_ascent_start = time.time()
            self._accumulate_rep_metrics(angles, current)
            self._compute_feedback_for_active_states(angles, self.prev_state, current)
            self._update_state_sequence(current)
        elif current == "s1":
            self._finalize_rep_if_needed()
            self.descent_start_time = None
            self._t_ascent_start = None

        self.prev_state = current

    def _update_state_sequence(self, current):
        if current == "s2":
            if not self.state_sequence:
                self.state_sequence.append("s2")
            elif self.state_sequence == ["s2", "s3"]:
                self.state_sequence.append("s2")
        elif current == "s3":
            if self.state_sequence == ["s2"]:
                self.state_sequence.append("s3")

        if len(self.state_sequence) > 3:
            self.state_sequence = self.state_sequence[-3:]

    def _compute_feedback_for_active_states(self, angles, prev_state, current_state):
        errors = []
        transition_ok = False
        knee_vert = self._depth_metric_for_view(angles) or 0.0
        is_frontal = self.current_view in ("front", "back")
        symmetry = angles.get("symmetry_diff", 0.0)

        # F3: "lower hips" cue only on s1 -> s2 transition.
        if (
            prev_state == "s1"
            and current_state == "s2"
            and self.FEEDBACK_THRESH["knee_mid_low"] < knee_vert < self.FEEDBACK_THRESH["knee_mid_high"]
        ):
            transition_ok = True

        if is_frontal:
            if symmetry > 40:
                self.incorrect_posture = True
                errors.append(("Uneven leg depth", "Balance both sides"))
        # Side: depth / torso / knee travel / tempo are evaluated at rep end (REP_RULES)
        # so we do not set incorrect_posture here (avoids double-counting with finalize).

        # Keep only current tips while moving; final counting happens in s1.
        if errors:
            for err in errors:
                if err not in self.rep_errors:
                    self.rep_errors.append(err)
            self._set_feedback(False, errors=errors[:3])
        elif transition_ok:
            self._set_feedback(True, message="Good transition, keep going...")

    def _accumulate_rep_metrics(self, angles: dict, current_state: str):
        ka = angles.get("knee_angle")
        if ka is not None:
            self._rep_knee_angle_min = min(self._rep_knee_angle_min, float(ka))
        ba = angles.get("back_angle")
        if ba is not None:
            self._rep_back_angle_max = max(self._rep_back_angle_max, float(ba))

        view = self.current_view
        if view == "side":
            kf = float(angles.get("knee_forward_norm") or 0.0)
            self._rep_knee_forward_max = max(self._rep_knee_forward_max, kf)
            if current_state == "s3" and angles.get("hip_below_knee"):
                self._rep_butt_wink = True
        elif view in ("front", "back"):
            self._rep_max_valgus = max(
                self._rep_max_valgus, float(angles.get("valgus_max") or 0.0)
            )
            self._rep_max_hip_asym = max(
                self._rep_max_hip_asym, float(angles.get("hip_height_asym_norm") or 0.0)
            )
            self._rep_max_toe_asym = max(
                self._rep_max_toe_asym, float(angles.get("toe_flare_asym_deg") or 0.0)
            )
            if self._rep_hip_mid_x0 is not None and angles.get("hip_mid_x_norm") is not None:
                drift = abs(float(angles["hip_mid_x_norm"]) - float(self._rep_hip_mid_x0))
                self._rep_max_lateral_drift = max(self._rep_max_lateral_drift, drift)
            if self._rep_ankle_y_avg0 is not None:
                aln = angles.get("ankle_y_left_norm")
                arn = angles.get("ankle_y_right_norm")
                if aln is not None and arn is not None:
                    cur = (float(aln) + float(arn)) / 2.0
                    rise = float(self._rep_ankle_y_avg0) - cur
                    if rise > 0:
                        self._rep_max_heel_rise = max(self._rep_max_heel_rise, rise)

    @staticmethod
    def _merge_error_lists(a, b):
        seen = set()
        out = []
        for lst in (a, b):
            for e in lst or []:
                if not e:
                    continue
                key = e[0]
                if key not in seen:
                    seen.add(key)
                    out.append(e)
        return out

    def _collect_rep_rule_violations(self) -> list:
        """End-of-rep rules: side vs front metrics (MediaPipe-limited, conservative thresholds)."""
        R = self.REP_RULES
        out = []
        view = self.current_view
        now = time.time()

        descent_dur = None
        if self.descent_start_time is not None and self._t_first_s3 is not None:
            descent_dur = self._t_first_s3 - self.descent_start_time

        ascent_dur = None
        if self._t_ascent_start is not None:
            ascent_dur = now - self._t_ascent_start

        meaningful = self._rep_knee_angle_min < 125.0

        if view == "side":
            if self._rep_knee_angle_min > R["shallow_knee_deg"]:
                out.append(("Shallow depth", "Lower hips more"))
            if self._rep_back_angle_max > R["torso_back_max_deg"]:
                out.append(("Torso too far fwd", "Keep chest up"))
            if self._rep_butt_wink:
                out.append(("Pelvis tuck at bottom", "Reduce depth slightly"))
            if self._rep_knee_forward_max > R["knee_forward_norm"]:
                out.append(("Knees past toes", "Sit hips back more"))
            if (
                descent_dur is not None
                and descent_dur < R["descent_fast_s"]
                and meaningful
            ):
                out.append(("Dropping too fast", "Control the descent"))
            if (
                ascent_dur is not None
                and ascent_dur > R["ascent_slow_s"]
                and meaningful
            ):
                out.append(("Rising too slow", "Drive up smoothly"))

        elif view in ("front", "back"):
            if self._rep_max_valgus > R["valgus_norm"]:
                out.append(("Knees caving in", "Push knees outward"))
            if self._rep_max_hip_asym > R["hip_height_asym"]:
                out.append(("Uneven hip height", "Keep hips level"))
            if self._rep_max_heel_rise > R["heel_rise_norm"]:
                out.append(("Heels lifting", "Keep heels planted"))
            if self._rep_max_lateral_drift > R["lateral_drift"]:
                out.append(("Hips shifting side", "Drop straight down"))
            if self._rep_max_toe_asym > R["toe_flare_asym_deg"]:
                out.append(("Feet angled unevenly", "Point feet the same"))

        return out

    def _finalize_rep_if_needed(self):
        if not self.state_sequence:
            return

        full_sequence = "s2" in self.state_sequence and "s3" in self.state_sequence
        rule_errors = self._collect_rep_rule_violations()
        rules_fail = len(rule_errors) > 0

        rep_ok = (
            full_sequence
            and not self.incorrect_posture
            and not rules_fail
        )

        if rep_ok:
            self.good_reps += 1
            self._set_feedback(True, message="Good rep, keep it up...")
        else:
            self.bad_reps += 1
            merged = self._merge_error_lists(self.rep_errors, rule_errors)
            if not full_sequence:
                merged = self._merge_error_lists(
                    merged,
                    [("Incomplete range", "Go deeper before standing")],
                )
            if not merged:
                merged = [("Form needs work", "Focus on control & depth")]
            self._set_feedback(False, errors=merged[:3])

        self.state_sequence = []
        self.incorrect_posture = False
        self.rep_errors = []
