"""
Real-Time Object Tracking Visualization Script

Visualizes:
- Bounding boxes for detected objects
- Track IDs with color-coded trails
- Confidence scores
- Object labels
- Loss events with alerts
- Real-time statistics overlay
- Multi-camera support
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

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


# ============================================================================
# COLOR MANAGEMENT
# ============================================================================

class ColorManager:
    """Manages consistent colors for track IDs."""
    
    # Color palette (BGR format for OpenCV)
    COLORS = [
        (0, 255, 0),       # Green
        (255, 0, 0),       # Blue
        (0, 0, 255),       # Red
        (255, 255, 0),     # Cyan
        (255, 0, 255),     # Magenta
        (0, 255, 255),     # Yellow
        (128, 0, 255),     # Orange
        (255, 128, 0),     # Sky Blue
        (128, 255, 0),     # Spring Green
        (0, 128, 255),     # Gold
        (255, 0, 128),     # Pink
        (128, 255, 255),   # Light Cyan
    ]
    
    def __init__(self):
        self.color_map: Dict[str, Tuple[int, int, int]] = {}
        self.color_counter = 0
    
    def get_color(self, track_id: str) -> Tuple[int, int, int]:
        """Get consistent color for a track ID."""
        if track_id not in self.color_map:
            color = self.COLORS[self.color_counter % len(self.COLORS)]
            self.color_map[track_id] = color
            self.color_counter += 1
        return self.color_map[track_id]
    
    def reset(self):
        """Reset color assignments."""
        self.color_map.clear()
        self.color_counter = 0


# ============================================================================
# VISUALIZATION ENGINE
# ============================================================================

class TrackingVisualizer:
    """Renders tracking results on video frames."""
    
    def __init__(
        self,
        video_source: str,
        camera_id: str = "cam_1",
        output_path: Optional[str] = None,
        detection_conf: float = 0.25,
        draw_trails: bool = True,
        trail_length: int = 20,
        draw_stats: bool = True,
        log_level: int = logging.INFO
    ):
        """
        Initialize the visualization engine.
        
        Args:
            video_source: Input video file or stream
            camera_id: Camera identifier
            output_path: Path to save output video (None = display only)
            detection_conf: Detection confidence threshold
            draw_trails: Whether to draw track trails
            trail_length: Length of track trail history
            draw_stats: Whether to draw statistics overlay
            log_level: Logging verbosity
        """
        self.logger = logging.getLogger("TrackingVisualizer")
        self.logger.setLevel(log_level)
        
        self.camera_id = camera_id
        self.video_source = video_source
        self.output_path = output_path
        self.draw_trails = draw_trails
        self.trail_length = trail_length
        self.draw_stats = draw_stats
        
        # Initialize components
        self.logger.info(f"Initializing visualization for {camera_id}")
        
        self.loader = VideoLoader(video_source, camera_id, simulate_realtime=False)
        self.detector = YOLODetector()
        self.detection_conf = detection_conf
        self.filterer = ObjectFilter()
        self.tracker = SimpleTracker(
            iou_threshold=0.3,
            appearance_threshold=0.7,
            max_missed=12
        )
        
        # Get video properties
        cap = cv2.VideoCapture(video_source)
        self.fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Visualization setup
        self.color_manager = ColorManager()
        
        # Track trails for visualization
        self.track_trails: Dict[str, deque] = defaultdict(lambda: deque(maxlen=trail_length))
        
        # Video writer setup
        self.video_writer = None
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_path,
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height)
            )
            self.logger.info(f"Video output configured: {output_path}")
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "total_tracks": set(),
            "lost_events": [],
            "max_concurrent_tracks": 0,
        }
        
        self.logger.info(
            f"Visualizer initialized: {self.frame_width}x{self.frame_height} @ {self.fps}fps"
        )
    
    def _format_detection(self, detection) -> Dict:
        """Convert detection object to dictionary format."""
        if isinstance(detection, dict):
            return detection
        elif hasattr(detection, "__dict__"):
            return detection.__dict__
        return {}
    
    def draw_bounding_box(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        track_id: str,
        label: str,
        confidence: float,
        is_lost: bool = False
    ) -> np.ndarray:
        """Draw bounding box with label and track ID."""
        x1, y1, x2, y2 = [int(p) for p in bbox]
        color = self.color_manager.get_color(track_id)
        thickness = 3 if is_lost else 2
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        label_text = f"ID:{track_id[:4]} {label} {confidence:.2f}"
        
        # Get text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, text_thickness
        )
        
        # Draw label background
        label_bg_color = (255, 0, 0) if is_lost else color  # Red for lost
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 5, y1),
            label_bg_color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label_text,
            (x1 + 2, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness
        )
        
        return frame
    
    def draw_track_trail(
        self,
        frame: np.ndarray,
        track_id: str,
        centroid: Tuple[int, int]
    ) -> np.ndarray:
        """Draw trail showing track history."""
        if not self.draw_trails:
            return frame
        
        # Store centroid
        self.track_trails[track_id].append(centroid)
        trail = self.track_trails[track_id]
        
        if len(trail) < 2:
            return frame
        
        # Draw trail line
        color = self.color_manager.get_color(track_id)
        points = np.array(list(trail), np.int32)
        cv2.polylines(frame, [points], False, color, 2, cv2.LINE_AA)
        
        # Draw centroid
        cv2.circle(frame, centroid, 3, color, -1)
        
        return frame
    
    def draw_loss_event(
        self,
        frame: np.ndarray,
        event: LostEvent
    ) -> np.ndarray:
        """Draw visual alert for loss event."""
        # Draw red alert box in corner
        height, width = frame.shape[:2]
        alert_height = 60
        
        # Draw semi-transparent red overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height - alert_height), (width, height), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw alert text
        alert_text = f"LOSS ALERT: {event.label} (ID:{event.track_id[:4]}) lost"
        cv2.putText(
            frame,
            alert_text,
            (10, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        
        return frame
    
    def draw_statistics_overlay(
        self,
        frame: np.ndarray,
        active_tracks: int,
        num_detections: int,
        num_lost: int,
        fps_actual: float
    ) -> np.ndarray:
        """Draw statistics overlay on frame."""
        if not self.draw_stats:
            return frame
        
        height, width = frame.shape[:2]
        
        # Prepare stats text
        stats_lines = [
            f"Camera: {self.camera_id}",
            f"Active Tracks: {active_tracks}",
            f"Detections: {num_detections}",
            f"Lost Events: {num_lost}",
            f"FPS: {fps_actual:.1f}",
            f"Frame: {self.stats['frames_processed']}/{self.frame_count}",
        ]
        
        # Draw semi-transparent background
        overlay = frame.copy()
        stats_height = len(stats_lines) * 25 + 10
        cv2.rectangle(overlay, (10, 10), (300, 10 + stats_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw stats text
        y_offset = 30
        for line in stats_lines:
            cv2.putText(
                frame,
                line,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1
            )
            y_offset += 25
        
        return frame
    
    def process_and_visualize_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        fps_actual: float
    ) -> Tuple[np.ndarray, List[LostEvent]]:
        """
        Process frame through tracking pipeline and visualize results.
        
        Args:
            frame: Input frame
            timestamp: Frame timestamp
            fps_actual: Actual processing FPS
        
        Returns:
            Tuple of (visualized_frame, lost_events)
        """
        self.stats["frames_processed"] += 1
        
        # Detection
        detections = self.detector.detect(frame, self.camera_id, timestamp)
        filtered_detections = self.filterer.filter(detections)
        
        # Format detections
        formatted_detections = []
        for det in (filtered_detections or []):
            formatted_det = self._format_detection(det)
            formatted_det["timestamp"] = timestamp
            formatted_detections.append(formatted_det)
        
        # Tracking
        self.tracker.current_frame = frame
        lost_tracks = self.tracker.update(formatted_detections, timestamp)
        
        # Process lost events
        lost_events = []
        for lost_track in lost_tracks:
            event = LostEvent(lost_track, self.camera_id, timestamp)
            lost_events.append(event)
            self.stats["lost_events"].append(event)
        
        # Visualization
        vis_frame = frame.copy()
        
        # Draw bounding boxes and trails
        for track in self.tracker.memory.active_tracks.values():
            self.stats["total_tracks"].add(track.id)
            
            # Get confidence from the detection
            confidence = 0.0
            for det in formatted_detections:
                if det.get("bbox") == track.bbox:
                    confidence = det.get("confidence", 0.0)
                    break
            
            # Draw bbox and trail
            vis_frame = self.draw_bounding_box(
                vis_frame,
                track.bbox,
                track.id,
                track.label,
                confidence,
                is_lost=False
            )
            
            centroid = (
                int((track.bbox[0] + track.bbox[2]) / 2),
                int((track.bbox[1] + track.bbox[3]) / 2)
            )
            vis_frame = self.draw_track_trail(vis_frame, track.id, centroid)
        
        # Draw loss events
        for event in lost_events:
            vis_frame = self.draw_loss_event(vis_frame, event)
        
        # Draw statistics
        num_active = len(self.tracker.memory.active_tracks)
        self.stats["max_concurrent_tracks"] = max(
            self.stats["max_concurrent_tracks"],
            num_active
        )
        
        vis_frame = self.draw_statistics_overlay(
            vis_frame,
            num_active,
            len(formatted_detections),
            len(lost_events),
            fps_actual
        )
        
        return vis_frame, lost_events
    
    def run(self, display: bool = True, max_frames: Optional[int] = None):
        """
        Run visualization on video stream.
        
        Args:
            display: Whether to display frames in window
            max_frames: Maximum frames to process (None = all)
        """
        import time
        
        self.logger.info(f"Starting visualization...")
        
        frame_times = deque(maxlen=30)  # For FPS calculation
        
        try:
            for cam_id, ts, frame in self.loader.frames():
                frame_start = time.time()
                
                # Process and visualize
                frame_time = time.time()
                vis_frame, lost_events = self.process_and_visualize_frame(
                    frame, ts, len(frame_times) / (sum(frame_times) + 1e-6) if frame_times else 0
                )
                
                frame_times.append(time.time() - frame_time)
                fps_actual = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
                
                # Save to output video
                if self.video_writer:
                    self.video_writer.write(vis_frame)
                
                # Display
                if display:
                    cv2.imshow(f"Tracking - {self.camera_id}", vis_frame)
                    
                    # Press 'q' to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.info("Visualization stopped by user")
                        break
                
                # Progress update
                if self.stats["frames_processed"] % 30 == 0:
                    progress = (self.stats["frames_processed"] / self.frame_count) * 100
                    self.logger.info(
                        f"Progress: {progress:.1f}% "
                        f"({self.stats['frames_processed']}/{self.frame_count})"
                    )
                
                # Check max frames
                if max_frames and self.stats["frames_processed"] >= max_frames:
                    break
        
        except Exception as e:
            self.logger.error(f"Error during visualization: {e}", exc_info=True)
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up visualization resources...")
        
        if self.loader:
            self.loader.release()
        
        if self.video_writer:
            self.video_writer.release()
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        self.print_summary()
    
    def print_summary(self):
        """Print visualization summary."""
        print(f"\n{'='*70}")
        print(f"VISUALIZATION SUMMARY")
        print(f"{'='*70}")
        print(f"  Frames Processed:     {self.stats['frames_processed']:>10d}")
        print(f"  Total Unique Tracks:  {len(self.stats['total_tracks']):>10d}")
        print(f"  Max Concurrent Tracks:{self.stats['max_concurrent_tracks']:>10d}")
        print(f"  Loss Events:          {len(self.stats['lost_events']):>10d}")
        
        if self.stats["total_tracks"]:
            print(f"\n  Tracked Labels:")
            label_counts = defaultdict(int)
            for track_id in self.stats["total_tracks"]:
                # Count from lost events
                for event in self.stats["lost_events"]:
                    if event.track_id == track_id:
                        label_counts[event.label] += 1
            
            for label, count in label_counts.items():
                print(f"    - {label}: {count}")
        
        if self.output_path:
            print(f"\n  Output saved to: {self.output_path}")
        
        print(f"{'='*70}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point for visualization."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
    )
    
    # Determine video path
    video_path = PROJECT_ROOT / "data" / "test_clips" / "cam1.mp4"
    
    if not video_path.exists():
        print(f"ERROR: Video file not found at {video_path}")
        return 1
    
    # Output path
    output_path = PROJECT_ROOT / "runs" / "detect" / "predict" / "output.mp4"
    
    # Create visualizer
    visualizer = TrackingVisualizer(
        video_source=str(video_path),
        camera_id="cam_1",
        output_path=str(output_path),
        detection_conf=0.25,
        draw_trails=True,
        trail_length=20,
        draw_stats=True,
        log_level=logging.INFO
    )
    
    # Run visualization
    visualizer.run(display=True, max_frames=None)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
