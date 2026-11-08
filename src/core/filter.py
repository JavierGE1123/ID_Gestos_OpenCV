"""
Gestión de filtros de imagen activados por gestos de la mano.

Flujo:
  - `arm()`   -> el usuario pulsa 'w': se borra el filtro actual y se queda
                 esperando a que el detector vea un gesto con un nº de dedos
                 soportado en `SUPPORTED_FINGERS`.
  - `update()`-> se llama cada frame. Si está armado y alguna mano del
                 `detector.last_contours` tiene un nº de dedos válido,
                 aplica el filtro correspondiente y se desarma.
  - `apply()` -> aplica el filtro activo al frame (in-place).
  - `clear()` -> 'r': elimina el filtro y desarma.
"""
import cv2


class FilterManager:
    """Filtros de imagen activados por conteo de dedos (OpenCV puro)."""

    #: nº de dedos -> nombre del filtro
    SUPPORTED_FINGERS = {
        5: "blanco&negro",
        4: "canny",
        3: "blur",
    }

    def __init__(self):
        self.active_filter = None   # None | "grayscale"
        self.armed = False          # True mientras esperamos un gesto

    # ---------------------------------------------------------- control
    def arm(self):
        """Entra en modo espera: borra el filtro actual y espera gesto."""
        self.active_filter = None
        self.armed = True

    def disarm(self):
        self.armed = False

    def clear(self):
        """Quita el filtro y sale de modo espera (tecla 'r')."""
        self.active_filter = None
        self.armed = False

    @property
    def is_active(self) -> bool:
        return self.active_filter is not None

    # ---------------------------------------------------------- lógica
    def update(self, detector):
        """
        Comprueba si hay que aplicar un filtro. Se llama cada frame.

        `detector.last_contours` es un dict {box: HandContour} que rellena
        `HandDetector.detect()` cuando `analyze_contours=True`.
        """
        if not self.armed:
            return
        contours = getattr(detector, "last_contours", None) or {}
        for hc in contours.values():
            name = self.SUPPORTED_FINGERS.get(hc.finger_count)
            if name is not None:
                self.active_filter = name
                self.armed = False
                return

    # ---------------------------------------------------------- efecto
    def apply(self, frame):
        """Aplica `self.active_filter` al frame in-place. Devuelve el frame."""
        if self.active_filter == "blanco&negro":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame[:] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif self.active_filter == "canny":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            frame[:] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif self.active_filter == "blur":
            frame[:] = cv2.GaussianBlur(frame, (21, 21), 0)
        return frame