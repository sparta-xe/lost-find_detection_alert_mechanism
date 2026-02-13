"""
Real-Time Object Tracking & Crowd-Based Loss Detection System
Test Script for Live Detection

This script demonstrates:
- Real-time object detection from video streams
- Multi-object tracking with appearance-based re-identification
- Loss event detection for crowd-based tracking
- Performance monitoring and statistics
- Configurable thresholds and parameters
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.video_loader import VideoLoader
from src.detection.yolo_detector import YOLODetector
from src.detection.object_filter import ObjectFilter
from src.tracking.tracker import SimpleTracker
from src.tracking.lost_event import LostEvent


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging for the detection pipeline.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("ObjectTracking")
    logger.setLevel(log_level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    
    # Add handler
    if not logger.handlers:
        logger.addHandler(ch)
    
    return logger


# ============================================================================
# REAL-TIME TRACKING PIPELINE
# ============================================================================

class RealTimeTrackingPipeline:
    """
    Complete real-time object tracking pipeline with loss detection.
    
    Features:
    - Real-time frame ingestion from video sources
    - YOLO-based object detection
    - Multi-object tracking with re-identification
    - Loss event detection and statistics
    - Performance monitoring
    """
    
    def __init__(
        self,
        video_source: str,
        camera_id: str = "cam_1",
        detection_conf: float = 0.25,
        iou_threshold: float = 0.3,
        appearance_threshold: float = 0.7,
        max_missed_frames: int = 12,
        simulate_realtime: bool = True,
        log_level: int = logging.INFO
    ):
        """
        Initialize the tracking pipeline.
        
        Args:
            video_source: Path to video file or camera stream
            camera_id: Unique identifier for this camera
            detection_conf: Confidence threshold for detections [0-1]
            iou_threshold: IoU threshold for spatial matching
            appearance_threshold: Threshold for appearance-based matching
            max_missed_frames: Max frames to keep track before removal
            simulate_realtime: Whether to simulate real-time playback speed
            log_level: Logging verbosity level
        """
        self.logger = setup_logging(log_level)
        self.logger.info(f"Initializing tracking pipeline for camera: {camera_id}")
        
        # Configuration
        self.camera_id = camera_id
        self.video_source = video_source
        self.simulate_realtime = simulate_realtime
        
        # Components
        self.logger.debug("Initializing video loader...")
        self.loader = VideoLoader(
            source=video_source,
            camera_id=camera_id,
            simulate_realtime=simulate_realtime
        )
        
        self.logger.debug("Initializing detector...")
        self.detector = YOLODetector()
        self.detection_conf = detection_conf
        
        self.logger.debug("Initializing object filter...")
        self.filterer = ObjectFilter()
        
        self.logger.debug("Initializing tracker...")
        self.tracker = SimpleTracker(
            iou_threshold=iou_threshold,
            appearance_threshold=appearance_threshold,
            max_missed=max_missed_frames
        )
        
        # Statistics tracking
        self.stats = {
            "frames_processed": 0,
            "total_detections": 0,
            "filtered_detections": 0,
            "active_tracks": 0,
            "lost_objects": 0,
            "last_update": None,
        }
        
        # Lost events tracking for crowd detection
        self.lost_events: List[LostEvent] = []
        self.lost_objects_by_label: Dict[str, int] = defaultdict(int)
        self.track_history: Dict[str, Dict] = {}
        
        self.logger.info(
            f"Pipeline initialized successfully. "
            f"Detection conf={self.detection_conf}, "
            f"IOU thresh={iou_threshold}, "
            f"Appearance thresh={appearance_threshold}"
        )
    
    def _format_detection(self, detection) -> Dict:
        """Convert detection object to dictionary format."""
        if hasattr(detection, "__dict__"):
            return detection.__dict__
        elif isinstance(detection, dict):
            return detection
        else:
            # Handle tuple/list format (bbox, conf, label)
            return {
                "bbox": detection[0] if len(detection) > 0 else None,
                "confidence": detection[1] if len(detection) > 1 else 0,
                "label": detection[2] if len(detection) > 2 else "unknown",
                "timestamp": datetime.utcnow().timestamp()
            }
    
    def process_frame(
        self,
        frame,
        timestamp: float
    ) -> Tuple[List[Dict], List[LostEvent], int]:
        """
        Process a single frame through the entire pipeline.
        
        Args:
            frame: Input video frame (numpy array)
            timestamp: Frame timestamp
        
        Returns:
            Tuple of (active_tracks, lost_events, num_active_tracks)
        """
        self.stats["frames_processed"] += 1
        
        # Step 1: Detection
        self.logger.debug(f"Frame {self.stats['frames_processed']}: Running detection...")
        detections = self.detector.detect(frame, self.camera_id, timestamp)
        self.stats["total_detections"] += len(detections) if detections else 0
        
        # Step 2: Filtering
        self.logger.debug(f"Frame {self.stats['frames_processed']}: Filtering detections...")
        filtered_detections = self.filterer.filter(detections)
        self.stats["filtered_detections"] += len(filtered_detections) if filtered_detections else 0
        
        # Convert detections to proper format for tracker
        formatted_detections = []
        for det in (filtered_detections or []):
            formatted_det = self._format_detection(det)
            formatted_det["timestamp"] = timestamp
            formatted_detections.append(formatted_det)
        
        # Step 3: Tracking with appearance features
        self.logger.debug(f"Frame {self.stats['frames_processed']}: Updating tracker...")
        self.tracker.current_frame = frame  # Inject frame for feature extraction
        lost_tracks = self.tracker.update(formatted_detections, timestamp)
        
        # Step 4: Loss Event Processing (Crowd-Based Detection)
        lost_events = []
        for lost_track in lost_tracks:
            lost_event = LostEvent(
                track=lost_track,
                camera_id=self.camera_id,
                timestamp=timestamp
            )
            lost_events.append(lost_event)
            self.lost_events.append(lost_event)
            self.stats["lost_objects"] += 1
            self.lost_objects_by_label[lost_track.label] += 1
            
            self.logger.warning(
                f"LOSS DETECTED: Track {lost_track.id} ({lost_track.label}) "
                f"lost at {timestamp:.2f}"
            )
        
        # Update statistics
        num_active_tracks = len(self.tracker.memory.active_tracks)
        self.stats["active_tracks"] = num_active_tracks
        self.stats["last_update"] = datetime.now()
        
        # Log active tracks periodically
        if self.stats["frames_processed"] % 30 == 0:  # Every ~1 second at 30fps
            self.logger.info(
                f"Frame {self.stats['frames_processed']}: "
                f"Active tracks: {num_active_tracks}, "
                f"Detections: {len(formatted_detections)}, "
                f"Lost events: {len(lost_events)}"
            )
        
        return list(self.tracker.memory.active_tracks.values()), lost_events, num_active_tracks
    
    def print_frame_summary(
        self,
        active_tracks: List,
        lost_events: List[LostEvent],
        num_active_tracks: int
    ):
        """Print summary information for the current frame."""
        if active_tracks:
            print(f"\n{'='*70}")
            print(f"FRAME {self.stats['frames_processed']} SUMMARY")
            print(f"{'='*70}")
            print(f"Active Tracks: {num_active_tracks}")
            
            print(f"\nTRACKABLE OBJECTS ({len(active_tracks)} active):")
            for i, track in enumerate(active_tracks, 1):
                bbox_str = f"[{track.bbox[0]:.0f}, {track.bbox[1]:.0f}, {track.bbox[2]:.0f}, {track.bbox[3]:.0f}]"
                print(f"  {i}. ID: {track.id:8s} | Label: {track.label:15s} | BBox: {bbox_str} | Missed: {track.missed_frames}")
        
        if lost_events:
            print(f"\n{'─'*70}")
            print(f"LOSS EVENTS ({len(lost_events)} detected):")
            for event in lost_events:
                print(f"  • Track {event.track_id:8s} ({event.label:15s}) lost at {event.timestamp}")
    
    def get_statistics(self) -> Dict:
        """Retrieve current pipeline statistics."""
        return {
            **self.stats,
            "lost_objects_by_label": dict(self.lost_objects_by_label),
            "avg_detection_rate": (
                self.stats["filtered_detections"] / max(self.stats["frames_processed"], 1)
            ),
            "avg_active_tracks": (
                self.stats["total_detections"] / max(self.stats["frames_processed"], 1)
            ),
            "loss_rate": (
                self.stats["lost_objects"] / max(self.stats["filtered_detections"], 1)
                if self.stats["filtered_detections"] > 0 else 0
            ),
        }
    
    def print_statistics(self):
        """Print comprehensive pipeline statistics."""
        stats = self.get_statistics()
        print(f"\n{'='*70}")
        print(f"PIPELINE STATISTICS (Updated: {stats['last_update']})")
        print(f"{'='*70}")
        print(f"  Frames Processed:        {stats['frames_processed']:>10d}")
        print(f"  Total Detections:        {stats['total_detections']:>10d}")
        print(f"  Filtered Detections:     {stats['filtered_detections']:>10d}")
        print(f"  Active Tracks:           {stats['active_tracks']:>10d}")
        print(f"  Lost Objects:            {stats['lost_objects']:>10d}")
        print(f"  Avg Detection Rate:      {stats['avg_detection_rate']:>10.2f}")
        print(f"  Avg Active Tracks:       {stats['avg_active_tracks']:>10.2f}")
        print(f"  Loss Rate:               {stats['loss_rate']*100:>10.2f}%")
        
        if stats["lost_objects_by_label"]:
            print(f"\n  Lost Objects by Label:")
            for label, count in stats["lost_objects_by_label"].items():
                print(f"    - {label}: {count}")
        print(f"{'='*70}\n")
    
    def run(self, max_frames: Optional[int] = None, verbose: bool = True):
        """
        Run the tracking pipeline on video stream.
        
        Args:
            max_frames: Maximum number of frames to process (None = all)
            verbose: Whether to print frame summaries
        """
        self.logger.info(f"Starting real-time tracking pipeline...")
        self.logger.info(f"Video source: {self.video_source}")
        self.logger.info(f"Camera ID: {self.camera_id}")
        
        try:
            for cam_id, ts, frame in self.loader.frames():
                # Process frame
                active_tracks, lost_events, num_active = self.process_frame(frame, ts)
                
                # Print summary
                if verbose and (active_tracks or lost_events):
                    self.print_frame_summary(active_tracks, lost_events, num_active)
                
                # Check max frames limit
                if max_frames and self.stats["frames_processed"] >= max_frames:
                    self.logger.info(f"Reached max frames limit: {max_frames}")
                    break
        
        except KeyboardInterrupt:
            self.logger.warning("Pipeline interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Pipeline error: {type(e).__name__}: {e}", exc_info=True)
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up resources...")
        if self.loader:
            self.loader.release()
        
        self.logger.info("Pipeline shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point for the tracking pipeline."""
    
    # Determine video path
    video_path = PROJECT_ROOT / "data" / "test_clips" / "cam1.mp4"
    
    # Validate video file exists
    if not video_path.exists():
        print(f"ERROR: Video file not found at {video_path}")
        print(f"Please ensure the test video exists in {video_path.parent}")
        return 1
    
    # Initialize and run pipeline
    with RealTimeTrackingPipeline(
        video_source=str(video_path),
        camera_id="cam_1",
        detection_conf=0.25,
        iou_threshold=0.3,
        appearance_threshold=0.7,
        max_missed_frames=12,
        simulate_realtime=True,
        log_level=logging.INFO
    ) as pipeline:
        # Run for testing (limit to 300 frames)
        pipeline.run(max_frames=None, verbose=True)
        
        # Print final statistics
        pipeline.print_statistics()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
