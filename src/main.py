import cv2

from core import webcam_controller as CAM
from core.hand_detector import HandDetector

WINDOW = 'Webcam'
MASK_WINDOW = 'Skin mask'


class SkinTuner:
    """
    Real-time skin tone adjustment:
      - left click on the hand to sample its skin tone,
      - sliders to fine-tune luminosity (center and range) and chroma tolerance,
      - 'r' resets the skin model, 'm' toggles the skin mask view, 'q' quits.
    """

    def __init__(self, detector):
        self.detector = detector
        self.last_frame = None
        self.show_mask = False
        self._updating = False

        cv2.namedWindow(WINDOW)
        cv2.createTrackbar('Luminosity', WINDOW, detector.skin_center[0], 255, self._on_luma_center)
        cv2.createTrackbar('Lum range', WINDOW, detector.luma_tolerance, 128, self._on_luma_range)
        cv2.createTrackbar('Chroma tol', WINDOW, detector.chroma_tolerance, 60, self._on_chroma)
        cv2.setMouseCallback(WINDOW, self._on_mouse)

    def _on_luma_center(self, value):
        if not self._updating:
            self.detector.set_luminosity(center=value)

    def _on_luma_range(self, value):
        if not self._updating:
            self.detector.set_luminosity(tolerance=value)

    def _on_chroma(self, value):
        if not self._updating:
            self.detector.set_chroma_tolerance(value)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.last_frame is not None:
            self.detector.calibrate_skin(self.last_frame, (x, y))
            self._sync_trackbars()

    def _sync_trackbars(self):
        self._updating = True
        cv2.setTrackbarPos('Luminosity', WINDOW, self.detector.skin_center[0])
        cv2.setTrackbarPos('Lum range', WINDOW, self.detector.luma_tolerance)
        cv2.setTrackbarPos('Chroma tol', WINDOW, self.detector.chroma_tolerance)
        self._updating = False

    def handle_key(self, key):
        if key == ord('r'):
            self.detector.reset_skin()
            self._sync_trackbars()
        elif key == ord('m'):
            self.show_mask = not self.show_mask
            if not self.show_mask:
                cv2.destroyWindow(MASK_WINDOW)

    def show(self, frame):
        if self.show_mask and self.detector.last_skin_mask is not None:
            cv2.imshow(MASK_WINDOW, self.detector.last_skin_mask)
        cv2.putText(frame, "Click on your hand to calibrate skin | m: mask | r: reset | q: quit",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow(WINDOW, frame)


def main():
    video = CAM.WebcamController()
    detector = HandDetector()
    tuner = SkinTuner(detector)

    while True:
        ret, frame = video.get_frame()
        if not ret:
            break

        tuner.last_frame = frame.copy()  # clean copy, so clicks sample the real colors
        hands = detector.detect(frame)
        detector.draw(frame, hands)

        tuner.show(frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        tuner.handle_key(key)

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
