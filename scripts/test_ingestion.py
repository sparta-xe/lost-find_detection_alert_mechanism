from pathlib import Path

from src.ingestion.video_loader import VideoLoader
from src.ingestion.frame_buffer import FrameBuffer
from src.ingestion.camera_sync import CameraSynchronizer


# Resolve project root dynamically (works on every machine)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

video_path = PROJECT_ROOT / "data" / "test_clips" / "cam1.mp4"

print(f"Loading video from: {video_path}")

loader = VideoLoader(str(video_path), camera_id="cam_1")
buffer = FrameBuffer(max_size=32)

sync = CameraSynchronizer({"cam_1": 0.0})


for cam_id, ts, frame in loader.frames():
    ts = sync.normalize(cam_id, ts)
    buffer.push((cam_id, ts, frame))

    item = buffer.pop()
    if item:
        print(f"Frame from {item[0]} @ {item[1]:.3f}")
