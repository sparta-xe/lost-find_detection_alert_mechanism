"""
Unit Tests for Real-Time Object Tracking & Loss Detection System

Tests covering:
- Detection pipeline
- Tracking accuracy
- Loss event detection
- Statistics collection
- Edge cases and error handling
"""

import sys
import logging
import unittest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_detection_live import RealTimeTrackingPipeline
from src.tracking.tracker import SimpleTracker, iou
from src.tracking.lost_event import LostEvent
from src.tracking.track_memory import Track, TrackMemory


# ============================================================================
# TEST SUITE 1: BASIC FUNCTIONALITY
# ============================================================================

class TestIOUCalculation(unittest.TestCase):
    """Test Intersection over Union (IoU) calculation for bounding boxes."""
    
    def test_perfect_overlap(self):
        """IoU should be 1.0 for identical boxes."""
        box = (10, 20, 100, 120)
        self.assertEqual(iou(box, box), 1.0)
    
    def test_no_overlap(self):
        """IoU should be 0.0 for non-overlapping boxes."""
        box1 = (10, 20, 100, 120)
        box2 = (200, 300, 400, 500)
        self.assertEqual(iou(box1, box2), 0.0)
    
    def test_partial_overlap(self):
        """IoU should be between 0 and 1 for partial overlap."""
        box1 = (10, 20, 100, 120)
        box2 = (50, 60, 150, 160)
        result = iou(box1, box2)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)
    
    def test_contained_box(self):
        """IoU should reflect area of smaller box."""
        large_box = (0, 0, 200, 200)
        small_box = (50, 50, 100, 100)
        result = iou(large_box, small_box)
        # Small box area = 50*50 = 2500
        # Large box area = 200*200 = 40000
        # IoU should be 2500 / 40000 = 0.0625
        self.assertAlmostEqual(result, 2500 / 40000, places=4)
    
    def test_symmetry(self):
        """IoU(A, B) should equal IoU(B, A)."""
        box1 = (10, 20, 100, 120)
        box2 = (50, 60, 150, 160)
        self.assertEqual(iou(box1, box2), iou(box2, box1))


class TestTrackMemory(unittest.TestCase):
    """Test track memory management."""
    
    def setUp(self):
        """Initialize track memory before each test."""
        self.memory = TrackMemory()
    
    def test_create_track(self):
        """Test creating a new track."""
        detection = {
            "label": "person",
            "bbox": (10, 20, 100, 120),
            "timestamp": 1.0
        }
        feature = np.random.rand(128)
        
        track = self.memory.create_track(detection, feature)
        
        self.assertIsNotNone(track)
        self.assertEqual(track.label, "person")
        self.assertEqual(len(self.memory.active_tracks), 1)
    
    def test_update_track(self):
        """Test updating an existing track."""
        # Create initial track
        detection = {
            "label": "person",
            "bbox": (10, 20, 100, 120),
            "timestamp": 1.0
        }
        feature = np.random.rand(128)
        track = self.memory.create_track(detection, feature)
        track_id = track.id
        
        # Update track
        new_detection = {
            "label": "person",
            "bbox": (15, 25, 105, 125),
            "timestamp": 2.0
        }
        new_feature = np.random.rand(128)
        self.memory.update_track(track_id, new_detection, new_feature)
        
        updated_track = self.memory.active_tracks[track_id]
        self.assertEqual(updated_track.bbox, (15, 25, 105, 125))
        self.assertEqual(updated_track.missed_frames, 0)
    
    def test_mark_missed(self):
        """Test marking track as missed."""
        detection = {
            "label": "person",
            "bbox": (10, 20, 100, 120),
            "timestamp": 1.0
        }
        feature = np.random.rand(128)
        track = self.memory.create_track(detection, feature)
        track_id = track.id
        
        # Mark as missed multiple times
        for i in range(3):
            self.memory.mark_missed(track_id)
        
        self.assertEqual(self.memory.active_tracks[track_id].missed_frames, 3)
    
    def test_delete_track(self):
        """Test deleting a track."""
        detection = {
            "label": "person",
            "bbox": (10, 20, 100, 120),
            "timestamp": 1.0
        }
        feature = np.random.rand(128)
        track = self.memory.create_track(detection, feature)
        track_id = track.id
        
        self.assertEqual(len(self.memory.active_tracks), 1)
        self.memory.delete_track(track_id)
        self.assertEqual(len(self.memory.active_tracks), 0)


class TestLostEvent(unittest.TestCase):
    """Test loss event creation and representation."""
    
    def test_lost_event_creation(self):
        """Test creating a loss event."""
        track = Track(
            id="track_123",
            label="person",
            bbox=(10, 20, 100, 120),
            feature=np.random.rand(128),
            first_seen=1.0,
            last_seen=2.0
        )
        
        event = LostEvent(track, "cam_1", timestamp=2.5)
        
        self.assertEqual(event.track_id, "track_123")
        self.assertEqual(event.label, "person")
        self.assertEqual(event.camera_id, "cam_1")
        self.assertEqual(event.timestamp, 2.5)
    
    def test_lost_event_representation(self):
        """Test string representation of loss event."""
        track = Track(
            id="track_456",
            label="backpack",
            bbox=(50, 60, 150, 160),
            feature=np.random.rand(128),
            first_seen=1.0,
            last_seen=3.0
        )
        
        event = LostEvent(track, "cam_2")
        repr_str = repr(event)
        
        self.assertIn("track_456", repr_str)
        self.assertIn("cam_2", repr_str)


# ============================================================================
# TEST SUITE 2: TRACKING PIPELINE
# ============================================================================

class TestSimpleTracker(unittest.TestCase):
    """Test the tracking algorithm."""
    
    def setUp(self):
        """Initialize tracker before each test."""
        self.tracker = SimpleTracker(
            iou_threshold=0.3,
            appearance_threshold=0.7,
            max_missed=5
        )
        # Create a dummy frame for feature extraction
        self.tracker.current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_new_track_creation(self):
        """Test that new objects create new tracks."""
        detections = [
            {
                "label": "person",
                "bbox": (10, 20, 100, 120),
                "confidence": 0.95,
                "timestamp": 1.0
            }
        ]
        
        lost_tracks = self.tracker.update(detections, 1.0)
        
        self.assertEqual(len(self.tracker.memory.active_tracks), 1)
        self.assertEqual(len(lost_tracks), 0)
    
    def test_track_continuation(self):
        """Test that tracks continue across frames."""
        # First frame - new object
        detections1 = [
            {
                "label": "person",
                "bbox": (10, 20, 100, 120),
                "confidence": 0.95,
                "timestamp": 1.0
            }
        ]
        lost_tracks1 = self.tracker.update(detections1, 1.0)
        track_ids_1 = set(self.tracker.memory.active_tracks.keys())
        
        # Second frame - same object moved slightly
        detections2 = [
            {
                "label": "person",
                "bbox": (15, 25, 105, 125),
                "confidence": 0.95,
                "timestamp": 2.0
            }
        ]
        lost_tracks2 = self.tracker.update(detections2, 2.0)
        track_ids_2 = set(self.tracker.memory.active_tracks.keys())
        
        # Track should continue with same ID
        self.assertEqual(track_ids_1, track_ids_2)
        self.assertEqual(len(lost_tracks2), 0)
    
    def test_track_loss_detection(self):
        """Test detection of lost tracks."""
        # Create a track
        detections1 = [
            {
                "label": "person",
                "bbox": (10, 20, 100, 120),
                "confidence": 0.95,
                "timestamp": 1.0
            }
        ]
        self.tracker.update(detections1, 1.0)
        track_id = list(self.tracker.memory.active_tracks.keys())[0]
        
        # Skip frames (no detection)
        lost_tracks = None
        for frame in range(2, 8):
            lost_tracks = self.tracker.update([], float(frame))
        
        # After enough missed frames, track should be lost
        self.assertGreater(len(lost_tracks), 0)
        self.assertNotIn(track_id, self.tracker.memory.active_tracks)
    
    def test_multiple_objects(self):
        """Test tracking multiple objects simultaneously."""
        # First detection - should create track 1
        detections1 = [
            {
                "label": "person",
                "bbox": (10, 20, 100, 120),
                "confidence": 0.95,
                "timestamp": 1.0
            }
        ]
        lost1 = self.tracker.update(detections1, 1.0)
        self.assertEqual(len(self.tracker.memory.active_tracks), 1)
        
        # Add more detections at different frames to build up multiple tracks
        detections2 = [
            {
                "label": "person",
                "bbox": (200, 250, 300, 400),
                "confidence": 0.92,
                "timestamp": 2.0
            }
        ]
        lost2 = self.tracker.update(detections2, 2.0)
        # Should have 2 tracks now (previous one marked as missed)
        self.assertGreaterEqual(len(self.tracker.memory.active_tracks), 1)
        
        # Continue first object from frame 1
        detections3 = [
            {
                "label": "person",
                "bbox": (15, 25, 105, 125),  # Slightly moved from first detection
                "confidence": 0.95,
                "timestamp": 3.0
            }
        ]
        lost3 = self.tracker.update(detections3, 3.0)
        # Should still be tracking at least the first object
        self.assertGreater(len(self.tracker.memory.active_tracks), 0)


# ============================================================================
# TEST SUITE 3: PIPELINE INTEGRATION
# ============================================================================

class TestPipelineStatistics(unittest.TestCase):
    """Test pipeline statistics collection."""
    
    @patch('scripts.test_detection_live.VideoLoader')
    @patch('scripts.test_detection_live.YOLODetector')
    @patch('scripts.test_detection_live.ObjectFilter')
    def test_statistics_initialization(self, mock_filter, mock_detector, mock_loader):
        """Test that statistics are properly initialized."""
        pipeline = RealTimeTrackingPipeline(
            video_source="dummy.mp4",
            camera_id="cam_test"
        )
        
        stats = pipeline.get_statistics()
        
        self.assertEqual(stats["frames_processed"], 0)
        self.assertEqual(stats["total_detections"], 0)
        self.assertEqual(stats["lost_objects"], 0)
    
    @patch('scripts.test_detection_live.VideoLoader')
    @patch('scripts.test_detection_live.YOLODetector')
    @patch('scripts.test_detection_live.ObjectFilter')
    def test_statistics_accumulation(self, mock_filter, mock_detector, mock_loader):
        """Test that statistics accumulate correctly."""
        pipeline = RealTimeTrackingPipeline(
            video_source="dummy.mp4",
            camera_id="cam_test"
        )
        
        # Mock frame processing
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.detector.detect = Mock(return_value=[
            {"label": "person", "bbox": (10, 20, 100, 120), "confidence": 0.95, "timestamp": 1.0}
        ])
        pipeline.filterer.filter = Mock(return_value=[
            {"label": "person", "bbox": (10, 20, 100, 120), "confidence": 0.95, "timestamp": 1.0}
        ])
        
        # Process frame
        pipeline.process_frame(dummy_frame, 1.0)
        
        stats = pipeline.get_statistics()
        
        self.assertEqual(stats["frames_processed"], 1)
        self.assertGreater(stats["total_detections"], 0)


# ============================================================================
# TEST SUITE 4: ERROR HANDLING
# ============================================================================

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_empty_detection_list(self):
        """Test handling of empty detections."""
        tracker = SimpleTracker()
        tracker.current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Should not raise error with empty detections
        lost_tracks = tracker.update([], 1.0)
        
        self.assertEqual(len(lost_tracks), 0)
    
    def test_none_detections(self):
        """Test handling of None detections."""
        tracker = SimpleTracker()
        tracker.current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Should not raise error with None
        lost_tracks = tracker.update(None or [], 1.0)
        
        self.assertEqual(len(lost_tracks), 0)
    
    def test_invalid_bbox(self):
        """Test handling of invalid bounding boxes."""
        tracker = SimpleTracker()
        tracker.current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detections = [
            {
                "label": "person",
                "bbox": (100, 20, 10, 120),  # Invalid: x1 > x2
                "confidence": 0.95,
                "timestamp": 1.0
            }
        ]
        
        # Should handle gracefully
        lost_tracks = tracker.update(detections, 1.0)
        self.assertIsInstance(lost_tracks, list)


# ============================================================================
# TEST SUITE 5: PERFORMANCE
# ============================================================================

class TestPipelinePerformance(unittest.TestCase):
    """Test pipeline performance characteristics."""
    
    def test_iou_computation_speed(self):
        """Test that IoU computation is fast."""
        import time
        
        box1 = (10, 20, 100, 120)
        box2 = (50, 60, 150, 160)
        
        start = time.time()
        for _ in range(10000):
            iou(box1, box2)
        elapsed = time.time() - start
        
        # Should compute 10k IoUs in under 1 second
        self.assertLess(elapsed, 1.0)
    
    def test_track_update_speed(self):
        """Test that track updates are fast."""
        import time
        
        memory = TrackMemory()
        
        # Create 100 tracks
        track_ids = []
        for i in range(100):
            detection = {
                "label": "person",
                "bbox": (i*10, i*10, i*10+100, i*10+100),
                "timestamp": 1.0
            }
            track = memory.create_track(detection, np.random.rand(128))
            track_ids.append(track.id)
        
        # Update all tracks
        start = time.time()
        for track_id in track_ids:
            detection = {
                "label": "person",
                "bbox": (100, 100, 200, 200),
                "timestamp": 2.0
            }
            memory.update_track(track_id, detection, np.random.rand(128))
        elapsed = time.time() - start
        
        # Should update 100 tracks in under 0.1 seconds
        self.assertLess(elapsed, 0.1)


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_tests(verbosity=2):
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestIOUCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestTrackMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestLostEvent))
    suite.addTests(loader.loadTestsFromTestCase(TestSimpleTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineStatistics))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelinePerformance))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    result = run_tests(verbosity=2)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
