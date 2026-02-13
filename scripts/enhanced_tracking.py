"""
Enhanced Real-Time Tracking with Lost Item Re-Identification

This script extends the tracking pipeline with lost item identification capability.
Allows uploading images of lost items that will be identified in real-time camera
feeds, even with low resolution.

Usage:
    python scripts/enhanced_tracking.py --video <path> --lost-item <image_path> --name <item_name>
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from argparse import ArgumentParser

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.video_loader import VideoLoader
from src.detection.yolo_detector import YOLODetector
from src.detection.object_filter import ObjectFilter
from src.tracking.tracker import SimpleTracker
from src.tracking.lost_event import LostEvent
from src.escalation.lost_item_service import LostItemService, LostItemReporter


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """Configure logging for the detection pipeline."""
    logger = logging.getLogger("EnhancedTracking")
    logger.setLevel(log_level)
    
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(ch)
    
    return logger


# ============================================================================
# ENHANCED TRACKING PIPELINE WITH LOST ITEM MATCHING
# ============================================================================

class EnhancedTrackingPipeline:
    """
    Real-time tracking pipeline with lost item re-identification.
    
    Features:
    - Real-time object detection and tracking
    - Lost item matching using multi-modal features
    - Low-resolution image matching capability
    - Comprehensive statistics and alerts
    """
    
    def __init__(
        self,
        video_source: str,
        camera_id: str = "cam_1",
        detection_conf: float = 0.25,
        iou_threshold: float = 0.3,
        appearance_threshold: float = 0.7,
        max_missed_frames: int = 12,
        lost_item_threshold: float = 0.6,
        simulate_realtime: bool = True,
        log_level: int = logging.INFO
    ):
        """
        Initialize enhanced tracking pipeline.
        
        Args:
            video_source: Path to video file
            camera_id: Camera identifier
            detection_conf: Detection confidence threshold
            iou_threshold: IoU threshold for tracking
            appearance_threshold: Appearance matching threshold
            max_missed_frames: Frames before track removal
            lost_item_threshold: Confidence threshold for lost item matches
            simulate_realtime: Simulate real-time playback
            log_level: Logging level
        """
        self.logger = setup_logging(log_level)
        self.logger.info("Initializing Enhanced Tracking Pipeline")
        
        # Configuration
        self.camera_id = camera_id
        self.video_source = video_source
        self.simulate_realtime = simulate_realtime
        
        # Components
        self.loader = VideoLoader(
            source=video_source,
            camera_id=camera_id,
            simulate_realtime=simulate_realtime
        )
        
        self.detector = YOLODetector()
        self.detection_conf = detection_conf
        
        self.filterer = ObjectFilter()
        
        self.tracker = SimpleTracker(
            iou_threshold=iou_threshold,
            appearance_threshold=appearance_threshold,
            max_missed=max_missed_frames
        )
        
        # Lost item service
        self.lost_item_service = LostItemService(
            match_threshold=lost_item_threshold
        )
        self.lost_item_reporter = LostItemReporter(self.lost_item_service)
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "total_detections": 0,
            "filtered_detections": 0,
            "active_tracks": 0,
            "lost_objects": 0,
            "lost_item_matches": 0,
            "last_update": None,
        }
        
        self.lost_events: List[LostEvent] = []
        self.lost_objects_by_label: Dict[str, int] = defaultdict(int)
        
        self.logger.info("Enhanced tracking pipeline ready")
    
    def add_lost_item(self, image_path: str, name: str, 
                     description: str = "") -> bool:
        """Add a lost item to track."""
        success, result = self.lost_item_service.upload_lost_item(
            image_path, name, description
        )
        if success:
            self.logger.info(f"Added lost item: {result} ({name})")
        return success
    
    def _format_detection(self, detection) -> Dict:
        """Convert detection to dictionary format."""
        if hasattr(detection, "__dict__"):
            return detection.__dict__
        elif isinstance(detection, dict):
            return detection
        else:
            return {
                "bbox": detection[0] if len(detection) > 0 else None,
                "confidence": detection[1] if len(detection) > 1 else 0,
                "label": detection[2] if len(detection) > 2 else "unknown",
                "timestamp": datetime.now().timestamp()
            }
    
    def process_frame(
        self,
        frame,
        timestamp: float
    ) -> Tuple[List[Dict], List[LostEvent], List[Dict]]:
        """
        Process frame with live object detection and lost item matching.
        
        Args:
            frame: Input video frame
            timestamp: Frame timestamp
        
        Returns:
            Tuple of (active_tracks, lost_events, lost_item_matches)
        """
        self.stats["frames_processed"] += 1
        
        # Step 1: Detection
        detections = self.detector.detect(frame, self.camera_id, timestamp)
        self.stats["total_detections"] += len(detections) if detections else 0
        
        # Step 2: Filtering
        filtered_detections = self.filterer.filter(detections)
        self.stats["filtered_detections"] += len(filtered_detections) if filtered_detections else 0
        
        # Convert to proper format
        formatted_detections = []
        for det in (filtered_detections or []):
            formatted_det = self._format_detection(det)
            formatted_det["timestamp"] = timestamp
            formatted_detections.append(formatted_det)
        
        # Step 3: Tracking
        self.tracker.current_frame = frame
        lost_tracks = self.tracker.update(formatted_detections, timestamp)
        
        # Step 4: Loss Event Processing
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
                f"LOSS DETECTED: Track {lost_track.id} ({lost_track.label})"
            )
        
        # Step 5: Lost Item Matching
        lost_item_matches = []
        lost_items = self.lost_item_service.get_lost_items()
        
        if lost_items:
            for det in formatted_detections:
                bbox = det.get("bbox", None)
                if bbox is None:
                    continue
                
                x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), 
                                  int(bbox[2]), int(bbox[3]))
                
                # Ensure valid bbox
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    # Crop detection
                    crop = frame[y1:y2, x1:x2]
                    
                    # Match against lost items
                    matches = self.lost_item_service.matcher.match_detection(
                        crop,
                        detection_id=f"det_{self.stats['frames_processed']}",
                        camera_id=self.camera_id,
                        bbox=(x1, y1, x2, y2),
                        frame_number=self.stats['frames_processed'],
                        timestamp=timestamp
                    )
                    
                    if matches:
                        lost_item_matches.extend(matches)
                        self.stats["lost_item_matches"] += len(matches)
                        
                        for match in matches:
                            self.logger.warning(
                                f"🎯 LOST ITEM MATCH FOUND: {match.lost_item_id} "
                                f"({match.confidence:.1%} confidence)"
                            )
        
        # Update statistics
        self.stats["active_tracks"] = len(self.tracker.memory.active_tracks)
        self.stats["last_update"] = datetime.now()
        
        # Log periodically
        if self.stats["frames_processed"] % 30 == 0:
            self.logger.info(
                f"Frame {self.stats['frames_processed']}: "
                f"Tracks={self.stats['active_tracks']}, "
                f"Detections={len(formatted_detections)}, "
                f"Lost items matched={self.stats['lost_item_matches']}"
            )
        
        return (
            list(self.tracker.memory.active_tracks.values()),
            lost_events,
            lost_item_matches
        )
    
    def print_frame_summary(
        self,
        active_tracks: List,
        lost_events: List[LostEvent],
        lost_item_matches: List[Dict]
    ):
        """Print frame summary with lost item matches."""
        if active_tracks or lost_events or lost_item_matches:
            print(f"\n{'='*70}")
            print(f"FRAME {self.stats['frames_processed']} SUMMARY")
            print(f"{'='*70}")
            
            if active_tracks:
                print(f"\nActive Tracks: {len(active_tracks)}")
                for track in active_tracks:
                    print(f"  • {track.id}: {track.label}")
            
            if lost_events:
                print(f"\n⚠️  Loss Events: {len(lost_events)}")
                for event in lost_events:
                    print(f"  • {event.track_id} ({event.label})")
            
            if lost_item_matches:
                print(f"\n🎯 LOST ITEMS FOUND: {len(lost_item_matches)}")
                for match in lost_item_matches:
                    print(f"  • {match['lost_item_id']}: {match.get('confidence', 'N/A'):.1%}")
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics."""
        return {
            **self.stats,
            "lost_objects_by_label": dict(self.lost_objects_by_label),
            "avg_detection_rate": (
                self.stats["filtered_detections"] / max(self.stats["frames_processed"], 1)
            ),
            "loss_rate": (
                self.stats["lost_objects"] / max(self.stats["filtered_detections"], 1)
                if self.stats["filtered_detections"] > 0 else 0
            ),
        }
    
    def print_statistics(self):
        """Print comprehensive statistics."""
        stats = self.get_statistics()
        print(f"\n{'='*70}")
        print(f"PIPELINE STATISTICS")
        print(f"{'='*70}")
        print(f"  Frames Processed:        {stats['frames_processed']:>10d}")
        print(f"  Total Detections:        {stats['total_detections']:>10d}")
        print(f"  Filtered Detections:     {stats['filtered_detections']:>10d}")
        print(f"  Active Tracks:           {stats['active_tracks']:>10d}")
        print(f"  Lost Objects:            {stats['lost_objects']:>10d}")
        print(f"  Lost Items Matched:      {stats['lost_item_matches']:>10d}")
        print(f"  Loss Rate:               {stats['loss_rate']*100:>10.2f}%")
        print(f"{'='*70}\n")
    
    def run(self, max_frames: Optional[int] = None, verbose: bool = True):
        """Run the enhanced tracking pipeline."""
        self.logger.info(f"Starting enhanced tracking on {self.video_source}")
        
        try:
            for cam_id, ts, frame in self.loader.frames():
                active_tracks, lost_events, lost_item_matches = self.process_frame(frame, ts)
                
                if verbose and (active_tracks or lost_events or lost_item_matches):
                    self.print_frame_summary(active_tracks, lost_events, lost_item_matches)
                
                if max_frames and self.stats["frames_processed"] >= max_frames:
                    break
        
        except KeyboardInterrupt:
            self.logger.warning("Pipeline interrupted")
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.loader:
            self.loader.release()
        self.logger.info("Cleanup complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main entry point with CLI support."""
    parser = ArgumentParser(description="Enhanced Real-Time Tracking with Lost Item Matching")
    parser.add_argument("--video", type=str, default="data/test_clips/cam1.mp4",
                       help="Video file path")
    parser.add_argument("--camera", type=str, default="cam_1",
                       help="Camera ID")
    parser.add_argument("--lost-item", type=str,
                       help="Image path of lost item to track")
    parser.add_argument("--name", type=str,
                       help="Name of lost item")
    parser.add_argument("--description", type=str, default="",
                       help="Description of lost item")
    parser.add_argument("--confidence", type=float, default=0.6,
                       help="Lost item match confidence threshold")
    parser.add_argument("--max-frames", type=int,
                       help="Maximum frames to process")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress frame summaries")
    parser.add_argument("--export", type=str,
                       help="Export results to JSON file")
    
    args = parser.parse_args()
    
    # Validate video
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {args.video}")
        return 1
    
    # Initialize pipeline
    pipeline = EnhancedTrackingPipeline(
        video_source=str(video_path),
        camera_id=args.camera,
        detection_conf=0.25,
        iou_threshold=0.3,
        appearance_threshold=0.7,
        max_missed_frames=12,
        lost_item_threshold=args.confidence
    )
    
    # Add lost item if provided
    if args.lost_item and args.name:
        if not pipeline.add_lost_item(args.lost_item, args.name, args.description):
            print("ERROR: Failed to add lost item")
            return 1
    
    # Run pipeline
    try:
        pipeline.run(max_frames=args.max_frames, verbose=not args.quiet)
        pipeline.print_statistics()
        
        # Export if requested
        if args.export:
            pipeline.lost_item_reporter.export_matches(args.export)
            print(f"\n✅ Results exported to {args.export}")
        
        # Print report
        if pipeline.lost_item_service.get_lost_items():
            print("\n" + pipeline.lost_item_reporter.report_matches())
    
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
