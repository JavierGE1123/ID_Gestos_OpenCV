import cv2

from core import webcam_controller as CAM
from core.hand_detector import HandDetector


def main():
    video = CAM.WebcamController()
    detector = HandDetector()

    while True:
        ret, frame = video.get_frame()
        if not ret:
            break

        hands = detector.detect(frame)
        detector.draw(frame, hands)

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
