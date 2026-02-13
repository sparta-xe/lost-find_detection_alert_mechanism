"""
Enhanced Tracking Pipeline V2 with Person-Object Interaction Detection

This enhanced version includes:
- Person detection and tracking
- Object state detection (dropped, carried, stationary)
- Person-object interaction detection
- Theft/pickup attempt alerts
- Improved lost item matching
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
from src.detection.enhanced_object_detector import EnhancedObjectDetector
from src.escalation.lost_item_service import LostItemService, LostItemReporter


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """Configure logging for the detection pipeline."""
    logger = logging.getLogger("EnhancedTrackingV2")
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


class EnhancedTrackingPipelineV2:
    """
    Advanced tracking pipeline with person-object interaction detection.
    
    Features:
    - Person detection and tracking
    - Object state tracking (dropped, carried, stationary)
    - Person-object interaction detection
    - Theft/pickup attempt alerts
    - Enhanced lost item matching
    """
    
    def __init__(
        self,
        video_source: str,
        camera_id: str = "cam_1",
        lost_item_threshold: float = 0.6,
        stationary_threshold: float = 3.0,
        proximity_threshold: float = 100.0,
        interaction_threshold: float = 50.0,
        simulate_realtime: bool = True,
        log_level: int = logging.INFO
    ):
        """
        Initialize enhanced tracking pipeline.
        
        Args:
            video_source: Path to video file or camera index
            camera_id: Camera identifier
            lost_item_threshold: Confidence threshold for lost item matches
            stationary_threshold: Time before object is considered stationary
            proximity_threshold: Distance for person-object proximity
            interaction_threshold: Distance for interaction detection
            simulate_realtime: Simulate real-time playback
            log_level: Logging level
        """
        self.logger = setup_logging(log_level)
        self.logger.info("Initializing Enhanced Tracking Pipeline V2")
        
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
        
        self.detector = EnhancedObjectDetector(
            stationary_threshold=stationary_threshold,
            proximity_threshold=proximity_threshold,
            interaction_threshold=interaction_threshold
        )
        
        # Lost item service
        self.lost_item_service = LostItemService(
            match_threshold=lost_item_threshold
        )
        self.lost_item_reporter = LostItemReporter(self.lost_item_service)
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "objects_detected": 0,
            "persons_detected": 0,
            "interactions_detected": 0,
            "pickup_attempts": 0,
            "items_picked_up": 0,
            "active_alerts": 0,
            "lost_item_matches": 0,
            "stationary_objects": 0,
            "last_update": None,
        }
        
        self.alerts = []
        self.person_tracking_data = {}  # Store detailed person tracking info
        
        self.logger.info("Enhanced tracking pipeline V2 ready")
    
    def add_lost_item(self, image_path: str, name: str, 
                     description: str = "") -> bool:
        """Add a lost item to track."""
        success, result = self.lost_item_service.upload_lost_item(
            image_path, name, description
        )
        if success:
            self.logger.info(f"Added lost item: {result} ({name})")
        return success
    
    def process_frame(
        self,
        frame,
        timestamp: float
    ) -> Tuple[List, List, List, List]:
        """
        Process frame with enhanced detection and interaction analysis.
        
        Args:
            frame: Input video frame
            timestamp: Frame timestamp
        
        Returns:
            Tuple of (objects, persons, interactions, lost_item_matches)
        """
        self.stats["frames_processed"] += 1
        
        # Enhanced detection and tracking
        objects, persons, interactions = self.detector.detect_and_track(
            frame, self.camera_id, timestamp
        )
        
        # Update statistics
        self.stats["objects_detected"] += len(objects)
        self.stats["persons_detected"] += len(persons)
        self.stats["interactions_detected"] += len(interactions)
        self.stats["pickup_attempts"] += len([i for i in interactions 
                                            if i.interaction_type == 'pickup_attempt'])
        self.stats["items_picked_up"] += len([i for i in interactions 
                                            if i.interaction_type == 'item_picked_up'])
        self.stats["stationary_objects"] = len([o for o in objects 
                                              if o.state == 'stationary'])
        
        # Get active alerts
        active_alerts = self.detector.get_active_alerts()
        self.stats["active_alerts"] = len(active_alerts)
        
        # Update person tracking data
        for person in persons:
            tracking_info = self.detector.get_person_tracking_info(person.person_id)
            if tracking_info:
                self.person_tracking_data[person.person_id] = tracking_info
        
        # Process alerts for interactions
        for interaction in interactions:
            if interaction.interaction_type == 'pickup_attempt':
                alert = {
                    'type': 'pickup_attempt',
                    'message': f"🚨 PICKUP ATTEMPT: Person {interaction.person_id} "
                              f"trying to take {interaction.object_id}",
                    'timestamp': timestamp,
                    'person_id': interaction.person_id,
                    'object_id': interaction.object_id,
                    'confidence': interaction.confidence,
                    'alert_level': interaction.alert_level
                }
                self.alerts.append(alert)
                self.logger.warning(alert['message'])
            
            elif interaction.interaction_type == 'item_picked_up':
                alert = {
                    'type': 'item_picked_up',
                    'message': f"🚨 ITEM PICKED UP: Person {interaction.person_id} "
                              f"has taken {interaction.object_id}",
                    'timestamp': timestamp,
                    'person_id': interaction.person_id,
                    'object_id': interaction.object_id,
                    'confidence': interaction.confidence,
                    'alert_level': interaction.alert_level
                }
                self.alerts.append(alert)
                self.logger.critical(alert['message'])
        
        # Process pickup alerts from detector
        for pickup_alert in active_alerts:
            if not any(a.get('alert_id') == pickup_alert.alert_id for a in self.alerts):
                alert = {
                    'type': 'pickup_alert',
                    'alert_id': pickup_alert.alert_id,
                    'message': pickup_alert.alert_message,
                    'timestamp': timestamp,
                    'person_id': pickup_alert.person_id,
                    'object_id': pickup_alert.object_id,
                    'severity': pickup_alert.severity,
                    'alert_level': 'critical'
                }
                self.alerts.append(alert)
                self.logger.critical(alert['message'])
        
        # Lost item matching
        lost_item_matches = []
        lost_items = self.lost_item_service.get_lost_items()
        
        if lost_items:
            for obj in objects:
                x1, y1, x2, y2 = obj.bbox
                
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
                        detection_id=obj.object_id,
                        camera_id=self.camera_id,
                        bbox=(x1, y1, x2, y2),
                        frame_number=self.stats['frames_processed'],
                        timestamp=timestamp
                    )
                    
                    if matches:
                        lost_item_matches.extend(matches)
                        self.stats["lost_item_matches"] += len(matches)
                        
                        for match in matches:
                            # Check if someone is near the matched lost item
                            nearby_persons = [p for p in persons 
                                            if obj.object_id in p.near_objects]
                            
                            alert_msg = f"🎯 LOST ITEM FOUND: {match.lost_item_id} "
                            alert_msg += f"({match.confidence:.1%} confidence)"
                            
                            if nearby_persons:
                                alert_msg += f" - Person nearby: {nearby_persons[0].person_id}"
                                if obj.pickup_attempt:
                                    alert_msg += " ⚠️ PICKUP ATTEMPT DETECTED!"
                            
                            if obj.state == 'stationary':
                                alert_msg += " (Object is stationary/dropped)"
                            
                            alert = {
                                'type': 'lost_item_found',
                                'message': alert_msg,
                                'timestamp': timestamp,
                                'match': match,
                                'object_state': obj.state,
                                'nearby_persons': len(nearby_persons),
                                'pickup_attempt': obj.pickup_attempt
                            }
                            self.alerts.append(alert)
                            
                            self.logger.warning(alert_msg)
        
        # Update statistics
        self.stats["last_update"] = datetime.now()
        
        # Log periodically
        if self.stats["frames_processed"] % 30 == 0:
            self.logger.info(
                f"Frame {self.stats['frames_processed']}: "
                f"Objects={len(objects)}, Persons={len(persons)}, "
                f"Interactions={len(interactions)}, "
                f"Lost items matched={len(lost_item_matches)}"
            )
        
        return objects, persons, interactions, lost_item_matches
    
    def draw_enhanced_overlay(self, frame: np.ndarray, objects: List, 
                            persons: List, interactions: List, 
                            lost_item_matches: List) -> np.ndarray:
        """Draw enhanced overlay with all detection information."""
        overlay_frame = frame.copy()
        
        # Draw objects
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox
            
            # Color based on state
            if obj.state == 'picked_up':
                color = (255, 0, 255)  # Magenta for picked up items
            elif obj.state == 'stationary':
                color = (0, 255, 255)  # Yellow for stationary
            elif obj.state == 'dropped':
                color = (0, 165, 255)  # Orange for dropped
            elif obj.pickup_attempt:
                color = (0, 0, 255)    # Red for pickup attempt
            else:
                color = (0, 255, 0)    # Green for normal
            
            # Draw bounding box
            thickness = 4 if obj.state == 'picked_up' else 3 if obj.pickup_attempt else 2
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with state
            label = f"{obj.label} ({obj.state})"
            if obj.person_nearby:
                label += " [Person nearby]"
            if obj.pickup_attempt:
                label += " [PICKUP ATTEMPT!]"
            if hasattr(obj, 'picked_up_by') and obj.picked_up_by:
                label += f" [TAKEN BY: {obj.picked_up_by}]"
            if hasattr(obj, 'tracking_priority') and obj.tracking_priority:
                label += " [HIGH PRIORITY]"
            
            cv2.putText(overlay_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw original position if item was picked up
            if obj.state == 'picked_up' and hasattr(obj, 'original_position') and obj.original_position:
                orig_x, orig_y = obj.original_position
                cv2.circle(overlay_frame, (orig_x, orig_y), 10, (255, 0, 255), 2)
                cv2.putText(overlay_frame, "ORIGINAL POS", (orig_x-30, orig_y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # Draw persons
        for person in persons:
            x1, y1, x2, y2 = person.bbox
            
            # Color based on behavior
            color = (255, 0, 0) if person.suspicious_behavior else (255, 255, 0)  # Red if suspicious, cyan if normal
            
            # Draw bounding box
            thickness = 3 if person.suspicious_behavior else 2
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"Person {person.person_id}"
            if person.suspicious_behavior:
                label += " [SUSPICIOUS]"
            if person.near_objects:
                label += f" [Near {len(person.near_objects)} objects]"
            if hasattr(person, 'carrying_objects') and person.carrying_objects:
                label += f" [CARRYING: {len(person.carrying_objects)}]"
            if hasattr(person, 'pickup_history') and person.pickup_history:
                label += f" [PICKUPS: {len(person.pickup_history)}]"
            
            cv2.putText(overlay_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw trajectory
            if person.trajectory and len(person.trajectory) > 1:
                points = np.array(person.trajectory, np.int32)
                cv2.polylines(overlay_frame, [points], False, color, 2)
        
        # Draw lost item matches with special highlighting
        for match in lost_item_matches:
            bbox = match.bbox
            x1, y1, x2, y2 = bbox
            
            # Draw highlighted bounding box for matches
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            
            # Draw match label with background
            match_text = f"FOUND: {match.lost_item_id} ({match.confidence:.1%})"
            text_size = cv2.getTextSize(match_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(overlay_frame, (x1, y1-35), (x1+text_size[0]+10, y1), (0, 0, 255), -1)
            cv2.putText(overlay_frame, match_text, (x1+5, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw interaction indicators
        for interaction in interactions:
            if interaction.bbox_person and interaction.bbox_object:
                # Draw line between person and object
                person_center = ((interaction.bbox_person[0] + interaction.bbox_person[2]) // 2,
                               (interaction.bbox_person[1] + interaction.bbox_person[3]) // 2)
                object_center = ((interaction.bbox_object[0] + interaction.bbox_object[2]) // 2,
                               (interaction.bbox_object[1] + interaction.bbox_object[3]) // 2)
                
                color = (0, 0, 255) if interaction.interaction_type == 'pickup_attempt' else (255, 0, 255)
                cv2.line(overlay_frame, person_center, object_center, color, 3)
                
                # Draw interaction label
                mid_point = ((person_center[0] + object_center[0]) // 2,
                           (person_center[1] + object_center[1]) // 2)
                cv2.putText(overlay_frame, interaction.interaction_type.upper(), 
                           mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return overlay_frame
    
    def print_frame_summary(self, objects: List, persons: List, 
                          interactions: List, lost_item_matches: List):
        """Print detailed frame summary."""
        if objects or persons or interactions or lost_item_matches:
            print(f"\n{'='*80}")
            print(f"FRAME {self.stats['frames_processed']} SUMMARY")
            print(f"{'='*80}")
            
            if objects:
                print(f"\n📦 Objects Detected: {len(objects)}")
                for obj in objects:
                    status = f"  • {obj.object_id}: {obj.label} ({obj.state})"
                    if obj.person_nearby:
                        status += " [Person nearby]"
                    if obj.pickup_attempt:
                        status += " [⚠️ PICKUP ATTEMPT]"
                    if hasattr(obj, 'picked_up_by') and obj.picked_up_by:
                        status += f" [🚨 TAKEN BY: {obj.picked_up_by}]"
                    if hasattr(obj, 'tracking_priority') and obj.tracking_priority:
                        status += " [HIGH PRIORITY]"
                    print(status)
            
            if persons:
                print(f"\n👤 Persons Detected: {len(persons)}")
                for person in persons:
                    status = f"  • {person.person_id}"
                    if person.near_objects:
                        status += f" [Near objects: {', '.join(person.near_objects)}]"
                    if person.suspicious_behavior:
                        status += " [🚨 SUSPICIOUS BEHAVIOR]"
                    if hasattr(person, 'carrying_objects') and person.carrying_objects:
                        status += f" [CARRYING: {', '.join(person.carrying_objects)}]"
                    if hasattr(person, 'pickup_history') and person.pickup_history:
                        status += f" [TOTAL PICKUPS: {len(person.pickup_history)}]"
                    print(status)
            
            if interactions:
                print(f"\n🤝 Interactions: {len(interactions)}")
                for interaction in interactions:
                    interaction_status = f"  • {interaction.person_id} → {interaction.object_id}: "
                    interaction_status += f"{interaction.interaction_type} ({interaction.confidence:.1%})"
                    if hasattr(interaction, 'alert_level'):
                        interaction_status += f" [{interaction.alert_level.upper()}]"
                    print(interaction_status)
            
            if lost_item_matches:
                print(f"\n🎯 LOST ITEMS FOUND: {len(lost_item_matches)}")
                for match in lost_item_matches:
                    print(f"  • {match.lost_item_id}: {match.confidence:.1%}")
            
            # Show active alerts
            active_alerts = self.detector.get_active_alerts()
            if active_alerts:
                print(f"\n🚨 ACTIVE ALERTS: {len(active_alerts)}")
                for alert in active_alerts:
                    print(f"  • {alert.alert_message} [{alert.severity.upper()}]")
    
    def get_statistics(self) -> Dict:
        """Get comprehensive pipeline statistics."""
        detector_stats = self.detector.get_statistics()
        
        return {
            **self.stats,
            **detector_stats,
            "alerts_generated": len(self.alerts),
            "pickup_attempts_detected": len([a for a in self.alerts 
                                           if a['type'] == 'pickup_attempt']),
            "lost_items_found": len([a for a in self.alerts 
                                   if a['type'] == 'lost_item_found']),
        }
    
    def print_statistics(self):
        """Print comprehensive statistics."""
        stats = self.get_statistics()
        print(f"\n{'='*80}")
        print(f"ENHANCED PIPELINE STATISTICS")
        print(f"{'='*80}")
        print(f"  Frames Processed:        {stats['frames_processed']:>10d}")
        print(f"  Objects Detected:        {stats['objects_detected']:>10d}")
        print(f"  Persons Detected:        {stats['persons_detected']:>10d}")
        print(f"  Interactions Detected:   {stats['interactions_detected']:>10d}")
        print(f"  Pickup Attempts:         {stats['pickup_attempts']:>10d}")
        print(f"  Lost Items Matched:      {stats['lost_item_matches']:>10d}")
        print(f"  Stationary Objects:      {stats['stationary_objects']:>10d}")
        print(f"  Alerts Generated:        {stats['alerts_generated']:>10d}")
        print(f"{'='*80}\n")
    
    def run(self, max_frames: Optional[int] = None, verbose: bool = True,
            show_video: bool = False):
        """Run the enhanced tracking pipeline."""
        self.logger.info(f"Starting enhanced tracking V2 on {self.video_source}")
        
        try:
            for cam_id, ts, frame in self.loader.frames():
                objects, persons, interactions, lost_item_matches = self.process_frame(frame, ts)
                
                if verbose and (objects or persons or interactions or lost_item_matches):
                    self.print_frame_summary(objects, persons, interactions, lost_item_matches)
                
                if show_video:
                    # Draw enhanced overlay
                    overlay_frame = self.draw_enhanced_overlay(
                        frame, objects, persons, interactions, lost_item_matches
                    )
                    
                    # Show frame
                    cv2.imshow('Enhanced Tracking V2', overlay_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
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
        cv2.destroyAllWindows()
        self.logger.info("Cleanup complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


def main():
    """Main entry point with CLI support."""
    parser = ArgumentParser(description="Enhanced Tracking V2 with Person-Object Interactions")
    parser.add_argument("--video", type=str, default="data/test_clips/streamlit_demo.mp4",
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
    parser.add_argument("--show-video", action="store_true",
                       help="Show video with overlays")
    parser.add_argument("--export", type=str,
                       help="Export results to JSON file")
    
    args = parser.parse_args()
    
    # Validate video
    if args.video != "0" and not Path(args.video).exists():
        print(f"ERROR: Video not found: {args.video}")
        return 1
    
    # Initialize pipeline
    pipeline = EnhancedTrackingPipelineV2(
        video_source=args.video,
        camera_id=args.camera,
        lost_item_threshold=args.confidence,
        stationary_threshold=3.0,
        proximity_threshold=100.0,
        interaction_threshold=50.0
    )
    
    # Add lost item if provided
    if args.lost_item and args.name:
        if not pipeline.add_lost_item(args.lost_item, args.name, args.description):
            print("ERROR: Failed to add lost item")
            return 1
    
    # Run pipeline
    try:
        pipeline.run(
            max_frames=args.max_frames, 
            verbose=not args.quiet,
            show_video=args.show_video
        )
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