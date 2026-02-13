# src/ingestion/camera_sync.py

from typing import Dict


class CameraSynchronizer:
    """
    Normalizes timestamps across cameras using configurable offsets.
    """

    def __init__(self, camera_offsets: Dict[str, float]):
        """
        camera_offsets example:
        {
            "cam_1": 0.0,
            "cam_2": -1.4,
            "cam_3": +0.8
        }
        """
        self.offsets = camera_offsets

    def normalize(self, camera_id: str, timestamp: float) -> float:
        offset = self.offsets.get(camera_id, 0.0)
        return timestamp + offset
