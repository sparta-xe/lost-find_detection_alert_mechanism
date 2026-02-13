from pathlib import Path

from src.ingestion.video_loader import VideoLoader
from src.detection.yolo_detector import YOLODetector
from src.detection.object_filter import ObjectFilter
from src.tracking.tracker import SimpleTracker
from src.escalation.trigger_logic import AbandonmentTrigger

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
video_path = PROJECT_ROOT / "data" / "test_clips" / "cam1.mp4"

# Initialize pipeline
loader = VideoLoader(str(video_path), camera_id="cam_1")
detector = YOLODetector()
filterer = ObjectFilter()
tracker = SimpleTracker()
trigger = AbandonmentTrigger()

print("\n--- TRACKING + ABANDONMENT ANALYSIS STARTED ---\n")

for cam_id, ts, frame in loader.frames():

    # Step 1 — Detect everything
    detections = detector.detect(frame, cam_id, ts)

    # Step 2 — Separate persons vs objects
    persons = [d for d in detections if d["label"] == "person"]
    objects = filterer.filter(detections)

    # Step 3 — Update tracker
    tracker.current_frame = frame
    tracker.update(objects, ts)

    active_tracks = list(tracker.memory.active_tracks.values())

    # Step 4 — Run abandonment reasoning
    alerts = trigger.update(active_tracks, persons, ts)

    # Debug print active tracks
    for t in active_tracks:
        print(f"[ACTIVE] ID={t.id} label={t.label} missed={t.missed_frames}")

    # Step 5 — Print alerts
    for a in alerts:
        print(f"\n🚨 ABANDONMENT ALERT → Track {a.id}\n")
