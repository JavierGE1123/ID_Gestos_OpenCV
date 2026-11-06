"""
Análisis geométrico de la mano a partir de la skin_mask (OpenCV puro, sin IA).

Pipeline por cada detección:
  1. Recorta la skin_mask a la caja de la detección y la limpia morfológicamente.
  2. Toma el contorno más grande -> contorno de la mano.
  3. Calcula su convex hull y los convexity defects (las "V" entre dedos).
  4. Filtra defectos por profundidad y ángulo -> picos válidos.
  5. Cuenta dedos y etiqueta el gesto usando nº de defectos + solidez.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# Cada defecto se guarda como (start, end, far, depth) ya en coordenadas del frame.
Defect = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], float]


@dataclass
class HandContour:
    """Resultado del análisis de contorno para una mano detectada."""
    box: Tuple[int, int, int, int]   # (x, y, w, h) ajustado al contorno real
    contour: np.ndarray              # (N, 1, 2) en coordenadas del frame
    hull: np.ndarray                 # (M, 1, 2) en coordenadas del frame
    defects: List[Defect]            # defectos válidos (ya filtrados)
    finger_count: int                # dedos extendidos estimados
    solidity: float                  # area(contorno) / area(hull)  en [0, 1]
    gesture: str                     # "fist" | "open" | "partial" | "unknown"


class ContourAnalyzer:
    """
    Refina la caja Haar con el contorno de piel y estima un gesto estático
    a partir de la geometría de la mano (hull + convexity defects).
    """

    def __init__(self,
                 min_area: int = 2500,
                 defect_depth_ratio: float = 0.10,
                 defect_angle_max: float = 90.0,
                 min_solidity: float = 0.55):
        """
        min_area:            área mínima del contorno para considerarlo mano.
        defect_depth_ratio:  profundidad mínima de un defecto, relativa al lado mayor
                             de la caja Haar (evita ruido en los bordes).
        defect_angle_max:    ángulo máximo (grados) en el punto más profundo del
                             defecto para que cuente como "valle entre dedos".
        min_solidity:        umbral informativo; la clasificación usa 0.85 fijo
                             para "fist" (ver _classify).
        """
        self.min_area = min_area
        self.defect_depth_ratio = defect_depth_ratio
        self.defect_angle_max = defect_angle_max
        self.min_solidity = min_solidity

    # ------------------------------------------------------------------ API
    def analyze(self,
                frame,
                detection,
                skin_mask: Optional[np.ndarray] = None) -> Optional[HandContour]:
        """Devuelve un `HandContour` o None si no hay contorno válido."""
        if skin_mask is None:
            return None

        x0, y0, w0, h0 = detection.box
        roi = skin_mask[y0:y0 + h0, x0:x0 + w0]
        if roi.size == 0:
            return None

        # Cerrar huecos (los dedos se separan por zonas oscuras) y limpiar ruido.
        kernel = np.ones((5, 5), np.uint8)
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=2)
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < self.min_area:
            return None

        # Hull con índices (necesario para convexityDefects)
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if hull_indices is None or len(hull_indices) < 3:
            return None

        hull_points = cv2.convexHull(contour, returnPoints=True)
        hull_area = cv2.contourArea(hull_points)
        contour_area = cv2.contourArea(contour)
        solidity = float(contour_area / hull_area) if hull_area > 0 else 0.0

        # Umbral de profundidad relativo al tamaño de la mano
        depth_threshold = self.defect_depth_ratio * max(w0, h0)

        valid_defects: List[Defect] = []
        defects_raw = cv2.convexityDefects(contour, hull_indices)
        if defects_raw is not None:
            for i in range(defects_raw.shape[0]):
                s, e, f, d = defects_raw[i, 0]
                depth = float(d) / 256.0
                if depth < depth_threshold:
                    continue
                start = tuple(int(v) for v in contour[s][0])
                end = tuple(int(v) for v in contour[e][0])
                far = tuple(int(v) for v in contour[f][0])
                angle = self._angle(start, far, end)
                if angle > self.defect_angle_max:
                    continue
                # Coordenadas locales (ROI) -> coordenadas del frame
                valid_defects.append(((start[0] + x0, start[1] + y0),
                                      (end[0] + x0, end[1] + y0),
                                      (far[0] + x0, far[1] + y0),
                                      depth))

        finger_count, gesture = self._classify(solidity, len(valid_defects))

        # Caja ajustada al contorno real (más ceñida que la de Haar)
        bx, by, bw, bh = cv2.boundingRect(contour)

        # Desplazar contorno y hull a coordenadas del frame
        shift = np.array([[x0, y0]], dtype=contour.dtype)
        contour_frame = contour + shift
        hull_frame = hull_points + shift

        return HandContour(
            box=(x0 + bx, y0 + by, bw, bh),
            contour=contour_frame,
            hull=hull_frame,
            defects=valid_defects,
            finger_count=finger_count,
            solidity=solidity,
            gesture=gesture,
        )

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _angle(a, b, c) -> float:
        """Ángulo en el vértice b formado por a-b-c, en grados [0, 180]."""
        ba = np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)
        bc = np.array(c, dtype=np.float32) - np.array(b, dtype=np.float32)
        denom = float(np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
        cos = float(np.dot(ba, bc) / denom)
        cos = max(-1.0, min(1.0, cos))
        return float(np.degrees(np.arccos(cos)))

    @staticmethod
    def _classify(solidity: float, num_defects: int) -> Tuple[int, str]:
        """
        Heurística de gesto estático:
          - Puño cerrado: solidez muy alta y ningún valle entre dedos.
          - Palma abierta: 3-4 valles -> 4-5 dedos extendidos.
          - 1-2 valles: gesto parcial (victoria, señalando, etc.).
        """
        if solidity > 0.85 and num_defects == 0:
            return 0, "fist"
        if num_defects >= 3:
            return min(num_defects + 1, 5), "open"
        if num_defects in (1, 2):
            return num_defects + 1, "partial"
        return 0, "unknown"

    # ------------------------------------------------------------- dibujo
    @staticmethod
    def draw(frame, hc: HandContour, color=(255, 0, 255)):
        """Dibuja contorno (magenta), hull (naranja), defectos (rojo) y etiqueta."""
        cv2.drawContours(frame, [hc.contour], -1, color, 2)
        cv2.drawContours(frame, [hc.hull], -1, (0, 200, 255), 2)
        for (_s, _e, far, _d) in hc.defects:
            cv2.circle(frame, far, 6, (0, 0, 255), -1)

        x, y, _w, _h = hc.box
        text = f"{hc.gesture} | dedos={hc.finger_count} | sol={hc.solidity:.2f}"
        cv2.putText(frame, text, (x, max(y - 10, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame