"""
Detector Abstraction Layer

This allows us to swap detection backends (YOLO, DINO, etc.)
without touching the pipeline.
"""

from .yolo_detector import YOLODetector


class Detector:
    """
    Unified detection interface used by pipeline.
    Internally uses YOLO for now.
    """

    def __init__(self, conf: float = 0.25):
        self.model = YOLODetector(conf=conf)

    def detect(self, frame):
        """
        Returns list of Detection objects.
        """
        return self.model.detect(frame)
