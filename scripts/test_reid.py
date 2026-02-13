"""
Re-Identification Test

Flow:
Cam1 → Track → Lose → Create LostEvent → Extract Embedding → Store
If Cam2 exists → Search for match
"""

import os

from src.ingestion.video_loader import VideoLoader
from src.detection.yolo_detector import YOLODetector
from src.detection.object_filter import ObjectFilter
from src.tracking.tracker import SimpleTracker
from src.tracking.lost_event import LostEvent

from src.reidentification.embedding_engine import EmbeddingEngine
from src.reidentification.similarity_search import SimilaritySearch
from src.storage.vector_store import VectorStore


# ============================================================
# CAMERA RUNNER
# ============================================================

def run_camera(camera_path, camera_id, tracker, detector, obj_filter):
    loader = VideoLoader(
        camera_path,
        camera_id=camera_id,
        simulate_realtime=False
    )

    for cam_id, ts, frame in loader.frames():

        detections = obj_filter.filter(detector.detect(frame))
        active_tracks, lost_tracks = tracker.update(detections)

        for lost_track in lost_tracks:
            print(f"\n⚠ LOST object {lost_track.id} on {cam_id}")

            return LostEvent(
                track=lost_track,
                camera_id=cam_id,
                timestamp=ts
            )

    return None


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n▶ Running Re-Identification Test\n")

    cam1_path = "data/test_clips/cam1.mp4"
    cam2_path = "data/test_clips/cam2.mp4"

    multi_camera = os.path.exists(cam2_path)

    if multi_camera:
        print("📡 Cam2 detected → Multi-camera mode\n")
    else:
        print("📷 Single-camera mode\n")

    # --------------------------------------------------------
    # Core Modules
    # --------------------------------------------------------

    detector = YOLODetector(conf=0.20)
    obj_filter = ObjectFilter()
    tracker = SimpleTracker(iou_threshold=0.3)

    embedder = EmbeddingEngine()
    vector_store = VectorStore()
    search_engine = SimilaritySearch(vector_store)

    # --------------------------------------------------------
    # 1️⃣ Run Camera 1
    # --------------------------------------------------------

    lost_event = run_camera(
        cam1_path,
        "cam_1",
        tracker,
        detector,
        obj_filter
    )

    if not lost_event:
        print("❌ No object lost in Cam1")
        return

    # --------------------------------------------------------
    # 2️⃣ Extract Embedding
    # --------------------------------------------------------

    print("🧠 Extracting embedding from lost object...")

    embedding = embedder.from_track(lost_event.track)

    vector_store.add(lost_event.track_id, embedding)

    print("📦 Stored in Vector Store\n")

    # --------------------------------------------------------
    # 3️⃣ If Cam2 exists → Search
    # --------------------------------------------------------

    if not multi_camera:
        print("ℹ No Cam2 available → ReID skipped")
        return

    print("🔎 Searching in Cam2...\n")

    # Fresh tracker for cam2
    tracker_cam2 = SimpleTracker(iou_threshold=0.3)

    loader = VideoLoader(
        cam2_path,
        camera_id="cam_2",
        simulate_realtime=False
    )

    for cam_id, ts, frame in loader.frames():

        detections = obj_filter.filter(detector.detect(frame))
        active_tracks, _ = tracker_cam2.update(detections)

        for track in active_tracks:

            emb = embedder.from_track(track)

            match_id, score = search_engine.search(emb)

            if match_id is not None and score > 0.75:
                print(f"✅ MATCH FOUND in {cam_id} (score={score:.3f})")
                print("🚚 Camera handoff successful\n")
                return

    print("❌ No match found in Cam2\n")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
