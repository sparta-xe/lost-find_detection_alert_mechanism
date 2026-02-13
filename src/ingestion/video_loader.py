# src/ingestion/video_loader.py

import cv2
import time
from typing import Generator, Tuple


class VideoLoader:
    """
    Loads frames from a video file or stream.
    Acts as a generator so pipeline can pull frames lazily.
    """

    def __init__(self, source: str, camera_id: str, simulate_realtime: bool = True):
        self.source = source
        self.camera_id = camera_id
        self.simulate_realtime = simulate_realtime

        # Handle different source types
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            # Camera index
            camera_index = int(source)
            self.cap = cv2.VideoCapture(camera_index)
        else:
            # File path
            self.cap = cv2.VideoCapture(str(source))
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.frame_delay = 1.0 / self.fps

    def frames(self) -> Generator[Tuple[str, float, any], None, None]:
        """
        Yields:
            (camera_id, timestamp, frame)
        """

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            timestamp = time.time()

            yield self.camera_id, timestamp, frame

            if self.simulate_realtime:
                time.sleep(self.frame_delay)

    def release(self):
        self.cap.release()
