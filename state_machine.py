import time
import random


class SquatStateMachine:
    STANDING = "STANDING"
    DESCENDING = "DESCENDING"
    BOTTOM = "BOTTOM"
    ASCENDING = "ASCENDING"

    def __init__(self):
        self.state = self.STANDING
        self.good_reps = 0
        self.bad_reps = 0
        self.descent_start_time = None
        self.ascent_start_time = None
        self.knee_angle_min = 180.0
        self.last_feedback = None
        self.feedback_timer = 0

    def update(self, angles: dict):
        knee = angles["left_knee_angle"]

        if self.feedback_timer > 0:
            self.feedback_timer -= 1

        # Track deepest point during descent and bottom
        if self.state in (self.DESCENDING, self.BOTTOM):
            self.knee_angle_min = min(self.knee_angle_min, knee)

        if self.state == self.STANDING and knee < 160:
            self.state = self.DESCENDING
            self.descent_start_time = time.time()

        elif self.state == self.DESCENDING and knee < 90:
            self.state = self.BOTTOM

        elif self.state == self.BOTTOM and knee > 100:
            self.state = self.ASCENDING
            self.ascent_start_time = time.time()

        elif self.state == self.ASCENDING and knee > 160:
            self._evaluate_rep(angles)
            self.state = self.STANDING

    def _evaluate_rep(self, angles: dict):
        descent_time = self.ascent_start_time - self.descent_start_time
        ascent_time = time.time() - self.ascent_start_time

        errors = []

        if self.knee_angle_min > 100:
            errors.append(("Too shallow", "Go deeper"))
        if angles["knee_x"] > angles["toe_x"] + 20:
            errors.append(("Knee over toe", "Push knees out"))
        if angles["back_angle"] < 45:
            errors.append(("Leaning forward", "Chest up"))
        if angles["symmetry_diff"] > 15:
            errors.append(("Uneven weight", "Balance your load"))
        if descent_time < 0.8:
            errors.append(("Too fast down", "Control descent"))
        if ascent_time > 3.0:
            errors.append(("Too slow up", "Drive up faster"))

        if not errors:
            self.good_reps += 1
            self.last_feedback = {
                "good": True,
                "message": random.choice([
                    "Nice one!", "Great job!", "Perfect rep!", "Keep it up!",
                ]),
            }
        else:
            self.bad_reps += 1
            self.last_feedback = {"good": False, "errors": errors}

        self.feedback_timer = 180
        self.knee_angle_min = 180.0
