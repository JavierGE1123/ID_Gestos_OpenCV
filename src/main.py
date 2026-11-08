import cv2

from core import webcam_controller as CAM
from core.hand_detector import HandDetector
from core.filter import FilterManager

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
        # [MODIFICADO] ayuda incluye la nueva tecla 'w'
        cv2.putText(frame,
                    "Click calibrate | m: mask | c: contours | w: gesture filter | r: reset | e: clear | q: quit",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow(WINDOW, frame)


def main():
    video = CAM.WebcamController()
    detector = HandDetector()
    tuner = SkinTuner(detector)
    filter_manager = FilterManager()                 # [NUEVO]

    show_contours = True

    while True:
        ret, frame = video.get_frame()
        if not ret:
            break

        tuner.last_frame = frame.copy()
        hands = detector.detect(frame)
        detector.draw(frame, hands)

        if show_contours:
            detector.draw_contours(frame)

        # [NUEVO] 1) comprueba si hay que activar filtro (solo si armado)
        #        2) aplica el filtro activo al frame de visualización
        filter_manager.update(detector)
        filter_manager.apply(frame)

        # [NUEVO] overlay de estado (se dibuja DESPUÉS del filtro -> en color)
        if filter_manager.armed:
            cv2.putText(frame, "ESPERANDO: Muestra dedos para activar filtros",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif filter_manager.is_active:
            cv2.putText(frame, f"ACTIVAR FILTRO: {filter_manager.active_filter}",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        tuner.show(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):                        # [NUEVO] armar filtro
            filter_manager.arm()
        elif key == ord('c'):
            show_contours = not show_contours
        elif key == ord('e'):
            filter_manager.clear()
        else:
            tuner.handle_key(key)                    # 'r' resetea piel, 'm' máscara

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
