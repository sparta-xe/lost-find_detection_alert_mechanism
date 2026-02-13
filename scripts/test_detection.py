from src.ingestion.video_loader import VideoLoader
from src.detection.detector import MockDetector
from src.detection.object_filter import ObjectFilter

loader = VideoLoader("data/test_clips/cam1.mp4", "cam_1", simulate_realtime=False)
detector = MockDetector()
obj_filter = ObjectFilter()

for cam_id, ts, frame in loader.frames():
    detections = detector.detect(frame)
    filtered = obj_filter.filter(detections)

    if filtered:
        print(f"[{cam_id}] {len(filtered)} tracked objects detected:")
        for d in filtered:
            print("   ", d.to_dict())
