from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .contour_analyzer import ContourAnalyzer  # [NUEVO]

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

DEFAULT_HAND_CASCADES = {
    "rpalm": MODELS_DIR / "rpalm.xml",
    "lpalm": MODELS_DIR / "lpalm.xml",
    "right": MODELS_DIR / "right.xml",
    "left": MODELS_DIR / "left.xml",
    "fist": MODELS_DIR / "fist.xml",
    "hand_dataset": MODELS_DIR / "Hand.Cascade.1.xml",
    "palm": MODELS_DIR / "palm.xml",
}
DEFAULT_FACE_CASCADE = MODELS_DIR / "haarcascade_frontalface_alt.xml"

# Modelo de piel en YCrCb (orden OpenCV: Y, Cr, Cb): centro ± tolerancia.
DEFAULT_SKIN_CENTER = (128, 153, 102)
DEFAULT_LUMA_TOLERANCE = 128
DEFAULT_CHROMA_TOLERANCE = 25


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    w: int
    h: int
    label: str
    score: float = 0.0

    @property
    def box(self):
        return self.x, self.y, self.w, self.h

    @property
    def area(self):
        return self.w * self.h

    @property
    def center(self):
        return self.x + self.w // 2, self.y + self.h // 2


class HandDetector:
    def __init__(self, hand_cascades=None, face_cascade=DEFAULT_FACE_CASCADE,
                 skin_center=DEFAULT_SKIN_CENTER, luma_tolerance=DEFAULT_LUMA_TOLERANCE,
                 chroma_tolerance=DEFAULT_CHROMA_TOLERANCE,
                 min_area=5000, min_skin_ratio=0.3, scale_factor=1.1, min_neighbors=4,
                 iou_threshold=0.3, exclude_faces=True, use_skin_filter=True,
                 refine_boxes=True,
                 analyze_contours=True):                      # [NUEVO]
        hand_cascades = DEFAULT_HAND_CASCADES if hand_cascades is None else hand_cascades
        self.hand_cascades = {name: self._load_cascade(path) for name, path in hand_cascades.items()}
        if not self.hand_cascades:
            raise ValueError("At least one hand cascade is required")

        self.face_cascade = self._load_cascade(face_cascade) if exclude_faces and face_cascade else None
        self.skin_center = tuple(int(v) for v in skin_center)
        self.luma_tolerance = int(luma_tolerance)
        self.chroma_tolerance = int(chroma_tolerance)
        self.refine_boxes = refine_boxes
        self.last_skin_mask = None
        self.min_area = min_area
        self.min_skin_ratio = min_skin_ratio
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.iou_threshold = iou_threshold
        self.use_skin_filter = use_skin_filter

        # [NUEVO] analizador de contornos + caché de resultados
        self.analyze_contours = analyze_contours
        self.contour_analyzer = ContourAnalyzer() if analyze_contours else None
        self.last_contours = {}   # dict: box_tuple -> HandContour

    # --------------------------------------------------------------- skin
    @property
    def skin_lower(self):
        y, cr, cb = self.skin_center
        return np.array([y - self.luma_tolerance, cr - self.chroma_tolerance,
                         cb - self.chroma_tolerance]).clip(0, 255).astype(np.uint8)

    @property
    def skin_upper(self):
        y, cr, cb = self.skin_center
        return np.array([y + self.luma_tolerance, cr + self.chroma_tolerance,
                         cb + self.chroma_tolerance]).clip(0, 255).astype(np.uint8)

    def set_luminosity(self, center=None, tolerance=None):
        if center is not None:
            self.skin_center = (int(center),) + self.skin_center[1:]
        if tolerance is not None:
            self.luma_tolerance = int(tolerance)

    def set_chroma_tolerance(self, tolerance):
        self.chroma_tolerance = int(tolerance)

    def calibrate_skin(self, frame, point, radius=8, luma_tolerance=40):
        h, w = frame.shape[:2]
        x = min(max(int(point[0]), 0), w - 1)
        y = min(max(int(point[1]), 0), h - 1)
        patch = frame[max(y - radius, 0):y + radius + 1, max(x - radius, 0):x + radius + 1]
        ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
        self.skin_center = tuple(int(v) for v in np.median(ycrcb, axis=0))
        self.luma_tolerance = int(luma_tolerance)
        return self.skin_center

    def reset_skin(self):
        self.skin_center = DEFAULT_SKIN_CENTER
        self.luma_tolerance = DEFAULT_LUMA_TOLERANCE
        self.chroma_tolerance = DEFAULT_CHROMA_TOLERANCE

    # ---------------------------------------------------------- utilidades
    @staticmethod
    def _load_cascade(path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Cascade file not found: {path}")
        cascade = cv2.CascadeClassifier(str(path))
        if cascade.empty():
            raise ValueError(f"Could not load cascade: {path}")
        return cascade

    # ---------------------------------------------------------- detección
    def detect(self, frame):
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale = cv2.equalizeHist(grayscale)

        faces = self.detect_faces(grayscale)
        skin_mask = self._skin_mask(frame, faces) if self.use_skin_filter else None
        self.last_skin_mask = skin_mask

        candidates = []
        for label, cascade in self.hand_cascades.items():
            rects, _, weights = cascade.detectMultiScale3(
                grayscale, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
                outputRejectLevels=True)
            for (x, y, w, h), weight in zip(rects, np.ravel(weights)):
                detection = Detection(int(x), int(y), int(w), int(h), label, float(weight))
                if self._is_valid(detection, faces, skin_mask):
                    detection = self._refine(detection, skin_mask)
                    if detection is not None:
                        candidates.append(detection)

        kept = self._non_max_suppression(candidates)

        # [NUEVO] análisis de contorno sobre las detecciones que sobreviven al NMS
        self.last_contours = {}
        if self.contour_analyzer is not None and self.last_skin_mask is not None:
            for det in kept:
                hc = self.contour_analyzer.analyze(frame, det, self.last_skin_mask)
                if hc is not None:
                    self.last_contours[det.box] = hc

        return kept

    # [NUEVO]
    def draw_contours(self, frame, color=(255, 0, 255)):
        if self.contour_analyzer is None:
            return frame
        for hc in self.last_contours.values():
            self.contour_analyzer.draw(frame, hc, color)
        return frame

    def detect_faces(self, grayscale):
        if self.face_cascade is None:
            return []
        return [tuple(int(v) for v in f)
                for f in self.face_cascade.detectMultiScale(grayscale, 1.1, 5)]

    def _skin_mask(self, frame, faces):
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.skin_lower, self.skin_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        for (fx, fy, fw, fh) in faces:
            mask[fy:fy + fh, fx:fx + fw] = 0
        return mask

    def _refine(self, detection, skin_mask):
        if not self.refine_boxes or skin_mask is None:
            return detection
        x0, y0 = detection.x, detection.y
        roi = skin_mask[y0:y0 + detection.h, x0:x0 + detection.w]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        if w * h < self.min_area * 0.25:
            return None
        return Detection(x0 + x, y0 + y, w, h, detection.label, detection.score)

    def _is_valid(self, detection, faces, skin_mask):
        if detection.area < self.min_area:
            return False
        for face in faces:
            if self._overlap_over_smaller(detection.box, face) > 0.5:
                return False
        if skin_mask is not None:
            roi = skin_mask[detection.y:detection.y + detection.h,
                            detection.x:detection.x + detection.w]
            if roi.size == 0 or cv2.countNonZero(roi) / roi.size < self.min_skin_ratio:
                return False
        return True

    @staticmethod
    def _intersection(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        iw = min(ax + aw, bx + bw) - max(ax, bx)
        ih = min(ay + ah, by + bh) - max(ay, by)
        return max(iw, 0) * max(ih, 0)

    def _overlap_over_smaller(self, a, b):
        return self._intersection(a, b) / max(min(a[2] * a[3], b[2] * b[3]), 1)

    def _iou(self, a, b):
        inter = self._intersection(a, b)
        return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1)

    def _non_max_suppression(self, detections):
        kept = []
        for det in sorted(detections, key=lambda d: d.score, reverse=True):
            if all(self._iou(det.box, k.box) < self.iou_threshold for k in kept):
                kept.append(det)
        return kept

    @staticmethod
    def draw(frame, detections, color=(0, 255, 0)):
        for det in detections:
            cv2.rectangle(frame, (det.x, det.y),
                          (det.x + det.w, det.y + det.h), color, 2)
            cv2.putText(frame, det.label, (det.x, max(det.y - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame