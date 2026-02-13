# src/ingestion/frame_buffer.py

from queue import Queue, Full, Empty


class FrameBuffer:
    """
    Thread-safe bounded buffer.
    Prevents pipeline stalls & memory blowups.
    """

    def __init__(self, max_size: int = 128):
        self.queue = Queue(maxsize=max_size)

    def push(self, item):
        try:
            self.queue.put(item, block=False)
        except Full:
            # Drop oldest frame to maintain freshness
            self.queue.get_nowait()
            self.queue.put(item)

    def pop(self):
        try:
            return self.queue.get(timeout=1)
        except Empty:
            return None

    def size(self):
        return self.queue.qsize()
