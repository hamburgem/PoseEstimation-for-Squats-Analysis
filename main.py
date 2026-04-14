import cv2
from pose_detector import PoseDetector
from angles import get_squat_angles
from state_machine import SquatStateMachine
from hud import draw_hud


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

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: failed to read frame")
            break

        frame, landmarks = detector.process(frame)
        h, w, _ = frame.shape

        angles = get_squat_angles(detector, landmarks, w, h)
        if angles:
            sm.update(angles)

        draw_hud(frame, sm, angles)
        cv2.imshow("Squat Analyzer", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
