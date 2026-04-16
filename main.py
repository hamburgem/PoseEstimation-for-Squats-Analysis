import cv2
from pose_detector import PoseDetector
from angles import get_squat_angles
from state_machine import SquatStateMachine
from hud import draw_hud


_UI_HITBOXES = {}


def _on_mouse(event, x, y, flags, userdata):
    del flags
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    sm = userdata
    if sm is None:
        return

    rect = _UI_HITBOXES.get("start_set_button")
    if rect is None:
        return

    bx, by, bw, bh = rect
    if bx <= x <= bx + bw and by <= y <= by + bh:
        if sm.set_active:
            sm.end_set()
        else:
            sm.start_set()


def main():
    print("Squat Analyzer running — press Q to quit")
    detector = PoseDetector()
    sm = SquatStateMachine()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: cannot open webcam")
        return

    window_name = "Squat Analyzer"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _on_mouse, sm)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: failed to read frame")
            break

        frame, landmarks = detector.detect(frame)
        h, w, _ = frame.shape

        angles = get_squat_angles(
            detector, landmarks, w, h, mode_override=sm.mode_override
        )
        if angles:
            sm.update(angles)
        else:
            sm.on_no_detection()

        if landmarks:
            if angles and angles.get("view") == "side" and angles.get("profile_side"):
                detector.draw_landmarks(
                    frame,
                    landmarks,
                    draw_view="side",
                    profile_side=angles["profile_side"],
                )
            else:
                detector.draw_landmarks(frame, landmarks, draw_view="full")

        global _UI_HITBOXES
        _UI_HITBOXES = draw_hud(frame, sm, angles)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("a"):
            sm.set_mode("auto")
        elif key == ord("f"):
            sm.set_mode("front")
        elif key == ord("s"):
            sm.set_mode("side")
        elif key == ord("b"):
            sm.set_mode("back")
        elif key == ord(" "):
            if sm.set_active:
                sm.end_set()
            else:
                sm.start_set()
        elif key == ord("1"):
            sm.set_skill_level("beginner")
        elif key == ord("2"):
            sm.set_skill_level("pro")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
