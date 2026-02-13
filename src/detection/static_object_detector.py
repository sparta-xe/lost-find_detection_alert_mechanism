import cv2
import numpy as np
from collections import defaultdict
from .base import Detection


class StaticObjectDetector:
    """
    Detects objects that remain stationary for multiple frames.
    Filters out motion noise, shadows, and passing people.
    """

    def __init__(self, min_area=1500, persistence_frames=25, iou_thresh=0.5):
        self.bg_model = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=25,
            detectShadows=False
        )

        self.min_area = min_area
        self.persistence_frames = persistence_frames
        self.iou_thresh = iou_thresh

        # store how long a region stays stable
        self.candidates = defaultdict(int)
        self.box_memory = {}

    def _iou(self, a, b):
        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])

        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (a[2]-a[0])*(a[3]-a[1])
        areaB = (b[2]-b[0])*(b[3]-b[1])
        union = areaA + areaB - inter

        return inter / union if union else 0

    def detect(self, frame):
        fgmask = self.bg_model.apply(frame)

        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        current_boxes = []

        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            current_boxes.append((x, y, x+w, y+h))

        detections = []

        # match boxes to previous ones
        for box in current_boxes:
            matched = False

            for key, prev_box in list(self.box_memory.items()):
                if self._iou(box, prev_box) > self.iou_thresh:
                    self.candidates[key] += 1
                    self.box_memory[key] = box
                    matched = True

                    if self.candidates[key] > self.persistence_frames:
                        detections.append(
                            Detection(
                                bbox=box,
                                confidence=0.6,
                                label="static_object"
                            )
                        )
                    break

            if not matched:
                new_id = len(self.box_memory) + 1
                self.box_memory[new_id] = box
                self.candidates[new_id] = 1

        return detections
