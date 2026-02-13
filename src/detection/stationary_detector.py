"""
Industry-Standard Stationary Object Detection System
Detects objects that appear, stop moving, and stay still for a threshold time.
This is how real surveillance systems detect abandoned/dropped objects.
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class StationaryObject:
    """Represents a stationary object in the scene."""
    object_id: str
    bbox: Tuple[int, int, int, int]
    label: str
    confidence: float
    first_seen: datetime
    last_seen: datetime
    stationary_since: Optional[datetime] = None
    velocity_history: List[float] = None
    position_history: List[Tuple[int, int]] = None
    is_stationary: bool = False
    is_dropped: bool = False
    drop_detected_at: Optional[datetime] = None

    def __post_init__(self):
        if self.velocity_history is None:
            self.velocity_history = []
        if self.position_history is None:
            self.position_history = []

class StationaryObjectDetector:
    """
    Industry-standard stationary object detection system.
    Tracks objects and identifies when they become stationary/abandoned.
    """
    
    def __init__(self, 
                 stationary_threshold: float = 3.0,
                 velocity_threshold: float = 5.0,
                 min_stationary_duration: float = 2.0,
                 max_tracking_distance: float = 50.0):
        """
        Initialize stationary object detector.
        
        Args:
            stationary_threshold: Time (seconds) before object is considered stationary
            velocity_threshold: Velocity threshold (pixels/second) for stationary detection
            min_stationary_duration: Minimum duration to confirm stationary status
            max_tracking_distance: Maximum distance for object tracking continuity
        """
        self.stationary_threshold = stationary_threshold
        self.velocity_threshold = velocity_threshold
        self.min_stationary_duration = min_stationary_duration
        self.max_tracking_distance = max_tracking_distance
        
        # Tracking data
        self.tracked_objects: Dict[str, StationaryObject] = {}
        self.next_object_id = 1
        
        # Frame differencing for motion detection
        self.prev_frame = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50
        )
        
        logger.info("Stationary object detector initialized")
    
    def process_frame(self, frame: np.ndarray, detections: List[Dict], 
                     timestamp: float) -> List[StationaryObject]:
        """
        Process frame and update stationary object tracking.
        
        Args:
            frame: Input video frame
            detections: List of object detections from YOLO
            timestamp: Frame timestamp
            
        Returns:
            List of stationary objects detected
        """
        current_time = datetime.fromtimestamp(timestamp)
        
        # Update background model
        fg_mask = self.background_subtractor.apply(frame)
        
        # Match detections to existing tracked objects
        self._update_tracked_objects(detections, current_time)
        
        # Detect motion using frame differencing (backup method)
        motion_areas = self._detect_motion_areas(frame, fg_mask)
        
        # Update stationary status for all objects
        self._update_stationary_status(current_time, motion_areas)
        
        # Clean up old objects
        self._cleanup_old_objects(current_time)
        
        # Return currently stationary objects
        stationary_objects = [obj for obj in self.tracked_objects.values() 
                            if obj.is_stationary]
        
        self.prev_frame = frame.copy()
        return stationary_objects
    
    def _update_tracked_objects(self, detections: List[Dict], current_time: datetime):
        """Update tracked objects with new detections."""
        # Convert detections to centers for easier tracking
        detection_centers = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            detection_centers.append((center_x, center_y, det))
        
        # Match existing objects to detections
        matched_objects = set()
        matched_detections = set()
        
        for obj_id, obj in self.tracked_objects.items():
            if obj_id in matched_objects:
                continue
            
            # Get object center
            x1, y1, x2, y2 = obj.bbox
            obj_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Find closest detection
            min_distance = float('inf')
            best_match_idx = -1
            
            for i, (det_x, det_y, det) in enumerate(detection_centers):
                if i in matched_detections:
                    continue
                
                distance = np.sqrt((obj_center[0] - det_x)**2 + (obj_center[1] - det_y)**2)
                
                if distance < min_distance and distance < self.max_tracking_distance:
                    min_distance = distance
                    best_match_idx = i
            
            # Update matched object
            if best_match_idx >= 0:
                det_x, det_y, detection = detection_centers[best_match_idx]
                
                # Update object properties
                obj.bbox = tuple(detection['bbox'])
                obj.confidence = detection['confidence']
                obj.last_seen = current_time
                
                # Calculate velocity
                if obj.position_history:
                    prev_pos = obj.position_history[-1]
                    time_diff = (current_time - obj.last_seen).total_seconds()
                    if time_diff > 0:
                        velocity = np.sqrt((det_x - prev_pos[0])**2 + (det_y - prev_pos[1])**2) / time_diff
                        obj.velocity_history.append(velocity)
                        
                        # Keep only recent velocity history
                        if len(obj.velocity_history) > 10:
                            obj.velocity_history.pop(0)
                
                # Update position history
                obj.position_history.append((det_x, det_y))
                if len(obj.position_history) > 20:
                    obj.position_history.pop(0)
                
                matched_objects.add(obj_id)
                matched_detections.add(best_match_idx)
        
        # Add new objects for unmatched detections
        for i, (det_x, det_y, detection) in enumerate(detection_centers):
            if i not in matched_detections:
                # Filter for trackable objects
                if detection['label'] in ['backpack', 'handbag', 'suitcase', 'laptop', 
                                        'cell phone', 'bottle', 'cup', 'book']:
                    obj_id = f"obj_{self.next_object_id}"
                    self.next_object_id += 1
                    
                    new_obj = StationaryObject(
                        object_id=obj_id,
                        bbox=tuple(detection['bbox']),
                        label=detection['label'],
                        confidence=detection['confidence'],
                        first_seen=current_time,
                        last_seen=current_time,
                        position_history=[(det_x, det_y)],
                        velocity_history=[0.0]
                    )
                    
                    self.tracked_objects[obj_id] = new_obj
                    logger.debug(f"Started tracking new object: {obj_id} ({detection['label']})")
    
    def _detect_motion_areas(self, frame: np.ndarray, fg_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect motion areas using background subtraction and frame differencing."""
        motion_areas = []
        
        # Clean up foreground mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours in motion areas
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                motion_areas.append((x, y, x + w, y + h))
        
        # Additional frame differencing if previous frame exists
        if self.prev_frame is not None:
            gray1 = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            diff = cv2.absdiff(gray1, gray2)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 300:
                    x, y, w, h = cv2.boundingRect(contour)
                    motion_areas.append((x, y, x + w, y + h))
        
        return motion_areas
    
    def _update_stationary_status(self, current_time: datetime, motion_areas: List[Tuple]):
        """Update stationary status for all tracked objects."""
        for obj_id, obj in self.tracked_objects.items():
            # Calculate average velocity over recent history
            if len(obj.velocity_history) >= 3:
                avg_velocity = np.mean(obj.velocity_history[-5:])
            else:
                avg_velocity = float('inf')  # Not enough data
            
            # Check if object is in motion area
            obj_center = self._get_object_center(obj.bbox)
            in_motion_area = any(self._point_in_bbox(obj_center, motion_area) 
                               for motion_area in motion_areas)
            
            # Determine if object should be considered stationary
            is_low_velocity = avg_velocity < self.velocity_threshold
            time_since_first_seen = (current_time - obj.first_seen).total_seconds()
            
            if is_low_velocity and not in_motion_area and time_since_first_seen > self.min_stationary_duration:
                if not obj.is_stationary:
                    # Object just became stationary
                    obj.stationary_since = current_time
                    obj.is_stationary = True
                    logger.info(f"Object {obj_id} ({obj.label}) became stationary")
                else:
                    # Check if object has been stationary long enough
                    stationary_duration = (current_time - obj.stationary_since).total_seconds()
                    if stationary_duration > self.stationary_threshold and not obj.is_dropped:
                        obj.is_dropped = True
                        obj.drop_detected_at = current_time
                        logger.warning(f"DROPPED OBJECT DETECTED: {obj_id} ({obj.label}) - stationary for {stationary_duration:.1f}s")
            else:
                # Object is moving
                if obj.is_stationary:
                    logger.debug(f"Object {obj_id} is moving again")
                obj.is_stationary = False
                obj.stationary_since = None
    
    def _get_object_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def _point_in_bbox(self, point: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> bool:
        """Check if point is inside bounding box."""
        px, py = point
        x1, y1, x2, y2 = bbox
        return x1 <= px <= x2 and y1 <= py <= y2
    
    def _cleanup_old_objects(self, current_time: datetime, max_age: float = 30.0):
        """Remove objects that haven't been seen for too long."""
        to_remove = []
        
        for obj_id, obj in self.tracked_objects.items():
            age = (current_time - obj.last_seen).total_seconds()
            if age > max_age:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            logger.debug(f"Removing old object: {obj_id}")
            del self.tracked_objects[obj_id]
    
    def get_dropped_objects(self) -> List[StationaryObject]:
        """Get list of objects that have been detected as dropped."""
        return [obj for obj in self.tracked_objects.values() if obj.is_dropped]
    
    def get_stationary_objects(self) -> List[StationaryObject]:
        """Get list of currently stationary objects."""
        return [obj for obj in self.tracked_objects.values() if obj.is_stationary]
    
    def reset(self):
        """Reset the detector state."""
        self.tracked_objects.clear()
        self.next_object_id = 1
        self.prev_frame = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50
        )
        logger.info("Stationary object detector reset")