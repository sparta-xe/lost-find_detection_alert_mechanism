from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    confidence: float
    label: str


class BaseDetector:
    def detect(self, frame) -> List[Detection]:
        raise NotImplementedError
