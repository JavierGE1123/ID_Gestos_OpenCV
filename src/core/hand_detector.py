from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

DEFAULT_HAND_CASCADES = {
    "hand": MODELS_DIR / "hand.xml",
    "hand_dataset": MODELS_DIR / "Hand.Cascade.1.xml",
    "palm": MODELS_DIR / "palm.xml",
    "fist": MODELS_DIR / "fist.xml",
}
DEFAULT_FACE_CASCADE = MODELS_DIR / "haarcascade_frontalface_alt.xml"

# Skin range in YCrCb color space (OpenCV order: Y, Cr, Cb)
DEFAULT_SKIN_LOWER = (0, 133, 77)
DEFAULT_SKIN_UPPER = (255, 173, 127)


@dataclass(frozen=True)
class Detection:
    """A single hand detection. `label` is the cascade that produced it (hand, palm, fist...)."""
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
    """
    Hand detector that combines several Haar cascades (hand, palm, fist...) and
    refines their output with:
      - face exclusion (cascades often fire on faces),
      - skin color validation (YCrCb),
      - minimum area filtering,
      - non-maximum suppression across all cascades.
    """

    def __init__(self, hand_cascades=None, face_cascade=DEFAULT_FACE_CASCADE,
                 skin_lower=DEFAULT_SKIN_LOWER, skin_upper=DEFAULT_SKIN_UPPER,
                 min_area=5000, min_skin_ratio=0.3, scale_factor=1.1, min_neighbors=4,
                 iou_threshold=0.3, exclude_faces=True, use_skin_filter=True):
        hand_cascades = DEFAULT_HAND_CASCADES if hand_cascades is None else hand_cascades
        self.hand_cascades = {name: self._load_cascade(path) for name, path in hand_cascades.items()}
        if not self.hand_cascades:
            raise ValueError("At least one hand cascade is required")

        self.face_cascade = self._load_cascade(face_cascade) if exclude_faces and face_cascade else None
        self.skin_lower = np.array(skin_lower, dtype=np.uint8)
        self.skin_upper = np.array(skin_upper, dtype=np.uint8)
        self.min_area = min_area
        self.min_skin_ratio = min_skin_ratio
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.iou_threshold = iou_threshold
        self.use_skin_filter = use_skin_filter

    @staticmethod
    def _load_cascade(path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Cascade file not found: {path}")
        cascade = cv2.CascadeClassifier(str(path))
        if cascade.empty():
            raise ValueError(f"Could not load cascade: {path}")
        return cascade

    def detect(self, frame):
        """Returns a list of `Detection` for the given BGR frame."""
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale = cv2.equalizeHist(grayscale)

        faces = self.detect_faces(grayscale)
        skin_mask = self._skin_mask(frame, faces) if self.use_skin_filter else None

        candidates = []
        for label, cascade in self.hand_cascades.items():
            rects, _, weights = cascade.detectMultiScale3(
                grayscale, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
                outputRejectLevels=True)
            for (x, y, w, h), weight in zip(rects, np.ravel(weights)):
                detection = Detection(int(x), int(y), int(w), int(h), label, float(weight))
                if self._is_valid(detection, faces, skin_mask):
                    candidates.append(detection)

        return self._non_max_suppression(candidates)

    def detect_faces(self, grayscale):
        if self.face_cascade is None:
            return []
        return [tuple(int(v) for v in f) for f in self.face_cascade.detectMultiScale(grayscale, 1.1, 5)]

    def _skin_mask(self, frame, faces):
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.skin_lower, self.skin_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        for (fx, fy, fw, fh) in faces:  # the face is skin too, it must not validate hands
            mask[fy:fy + fh, fx:fx + fw] = 0
        return mask

    def _is_valid(self, detection, faces, skin_mask):
        if detection.area < self.min_area:
            return False
        for face in faces:
            if self._overlap_over_smaller(detection.box, face) > 0.5:
                return False
        if skin_mask is not None:
            roi = skin_mask[detection.y:detection.y + detection.h, detection.x:detection.x + detection.w]
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
            cv2.rectangle(frame, (det.x, det.y), (det.x + det.w, det.y + det.h), color, 2)
            cv2.putText(frame, det.label, (det.x, max(det.y - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
