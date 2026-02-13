"""
Industry-Standard Drop Event Detection System
Detects when objects are dropped by analyzing trajectory patterns:
1. Object near hand/person
2. Sudden downward movement
3. Object becomes stationary
This works even when detection misses 1-2 frames.
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class DropEvent:
    """Represents a detected drop event."""
    event_id: str
    object_id: str
    person_id: Optional[str]
    drop_location: Tuple[int, int]
    drop_time: datetime
    trajectory_data: List[Tuple[int, int, float]]  # (x, y, timestamp)
    confidence: float
    object_label: str
    drop_height: float  # Vertical distance fallen
    drop_velocity: float  # Speed of drop

@dataclass
class TrajectoryPoint:
    """Single point in object trajectory."""
    x: int
    y: int
    timestamp: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    acceleration_y: float = 0.0

class DropEventDetector:
    """
    Industry-standard drop event detection using trajectory analysis.
    Detects objects being dropped by analyzing movement patterns.
    """
    
    def __init__(self,
                 fall_velocity_threshold: float = 50.0,  # pixels/second
                 min_fall_distance: float = 30.0,        # pixels
                 proximity_threshold: float = 100.0,     # pixels from person
                 trajectory_window: int = 10,            # frames to analyze
                 stationary_threshold: float = 2.0):     # seconds
        """
        Initialize drop event detector.
        
        Args:
            fall_velocity_threshold: Minimum downward velocity to detect falling
            min_fall_distance: Minimum vertical distance for drop detection
            proximity_threshold: Distance from person to consider "near hand"
            trajectory_window: Number of frames to analyze for trajectory
            stationary_threshold: Time to confirm object is stationary after drop
        """
        self.fall_velocity_threshold = fall_velocity_threshold
        self.min_fall_distance = min_fall_distance
        self.proximity_threshold = proximity_threshold
        self.trajectory_window = trajectory_window
        self.stationary_threshold = stationary_threshold
        
        # Tracking data
        self.object_trajectories: Dict[str, List[TrajectoryPoint]] = {}
        self.person_positions: Dict[str, Tuple[int, int, float]] = {}  # (x, y, timestamp)
        self.detected_drops: List[DropEvent] = []
        self.next_event_id = 1
        
        logger.info("Drop event detector initialized")
    
    def process_frame(self, objects: List[Dict], persons: List[Dict], 
                     timestamp: float) -> List[DropEvent]:
        """
        Process frame and detect drop events.
        
        Args:
            objects: List of detected objects
            persons: List of detected persons
            timestamp: Frame timestamp
            
        Returns:
            List of newly detected drop events
        """
        current_time = datetime.fromtimestamp(timestamp)
        new_drops = []
        
        # Update person positions
        self._update_person_positions(persons, timestamp)
        
        # Update object trajectories
        self._update_object_trajectories(objects, timestamp)
        
        # Analyze trajectories for drop events
        for obj_id, trajectory in self.object_trajectories.items():
            if len(trajectory) >= 3:  # Need at least 3 points for analysis
                drop_event = self._analyze_trajectory_for_drop(obj_id, trajectory, timestamp)
                if drop_event:
                    new_drops.append(drop_event)
                    self.detected_drops.append(drop_event)
                    logger.warning(f"DROP EVENT DETECTED: {drop_event.object_label} dropped at {drop_event.drop_location}")
        
        # Clean up old trajectories
        self._cleanup_old_trajectories(timestamp)
        
        return new_drops
    
    def _update_person_positions(self, persons: List[Dict], timestamp: float):
        """Update tracked person positions."""
        current_persons = set()
        
        for person in persons:
            person_id = person.get('person_id', f"person_{len(self.person_positions)}")
            current_persons.add(person_id)
            
            # Get person center
            x1, y1, x2, y2 = person['bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            self.person_positions[person_id] = (center_x, center_y, timestamp)
        
        # Remove old persons
        to_remove = []
        for person_id, (_, _, last_seen) in self.person_positions.items():
            if person_id not in current_persons and timestamp - last_seen > 5.0:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.person_positions[person_id]
    
    def _update_object_trajectories(self, objects: List[Dict], timestamp: float):
        """Update object trajectory tracking."""
        current_objects = set()
        
        for obj in objects:
            obj_id = obj.get('object_id', obj.get('detection_id', f"obj_{timestamp}"))
            current_objects.add(obj_id)
            
            # Get object center
            x1, y1, x2, y2 = obj['bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Initialize trajectory if new object
            if obj_id not in self.object_trajectories:
                self.object_trajectories[obj_id] = []
            
            trajectory = self.object_trajectories[obj_id]
            
            # Create trajectory point
            point = TrajectoryPoint(x=center_x, y=center_y, timestamp=timestamp)
            
            # Calculate velocities if we have previous points
            if len(trajectory) > 0:
                prev_point = trajectory[-1]
                dt = timestamp - prev_point.timestamp
                
                if dt > 0:
                    point.velocity_x = (center_x - prev_point.x) / dt
                    point.velocity_y = (center_y - prev_point.y) / dt
                    
                    # Calculate acceleration if we have enough points
                    if len(trajectory) > 1:
                        prev_velocity_y = prev_point.velocity_y
                        point.acceleration_y = (point.velocity_y - prev_velocity_y) / dt
            
            trajectory.append(point)
            
            # Keep only recent trajectory points
            if len(trajectory) > self.trajectory_window:
                trajectory.pop(0)
        
        # Clean up trajectories for objects no longer detected
        to_remove = []
        for obj_id in self.object_trajectories.keys():
            if obj_id not in current_objects:
                # Keep trajectory for a short time in case object reappears
                if len(self.object_trajectories[obj_id]) > 0:
                    last_seen = self.object_trajectories[obj_id][-1].timestamp
                    if timestamp - last_seen > 2.0:  # 2 seconds grace period
                        to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.object_trajectories[obj_id]
    
    def _analyze_trajectory_for_drop(self, obj_id: str, trajectory: List[TrajectoryPoint], 
                                   timestamp: float) -> Optional[DropEvent]:
        """
        Analyze object trajectory to detect drop events.
        
        Args:
            obj_id: Object identifier
            trajectory: List of trajectory points
            timestamp: Current timestamp
            
        Returns:
            DropEvent if drop detected, None otherwise
        """
        if len(trajectory) < 3:
            return None
        
        # Get recent trajectory points
        recent_points = trajectory[-min(len(trajectory), self.trajectory_window):]
        
        # Phase 1: Check if object was near a person
        was_near_person = False
        nearest_person_id = None
        
        for point in recent_points[:len(recent_points)//2]:  # Check earlier points
            for person_id, (px, py, pt) in self.person_positions.items():
                if abs(pt - point.timestamp) < 1.0:  # Within 1 second
                    distance = np.sqrt((point.x - px)**2 + (point.y - py)**2)
                    if distance < self.proximity_threshold:
                        was_near_person = True
                        nearest_person_id = person_id
                        break
            if was_near_person:
                break
        
        # Phase 2: Detect sudden downward movement
        falling_detected = False
        fall_start_idx = -1
        max_fall_velocity = 0.0
        
        for i in range(1, len(recent_points)):
            point = recent_points[i]
            
            # Check for significant downward velocity
            if point.velocity_y > self.fall_velocity_threshold:
                falling_detected = True
                fall_start_idx = i
                max_fall_velocity = max(max_fall_velocity, point.velocity_y)
        
        # Phase 3: Check if object became stationary after falling
        became_stationary = False
        stationary_start = -1
        
        if falling_detected and len(recent_points) > fall_start_idx + 2:
            # Look for stationary period after fall
            for i in range(fall_start_idx + 1, len(recent_points)):
                point = recent_points[i]
                
                # Check if velocity is low (stationary)
                if abs(point.velocity_x) < 10 and abs(point.velocity_y) < 10:
                    if stationary_start == -1:
                        stationary_start = i
                    
                    # Check if stationary for enough time
                    if i == len(recent_points) - 1:  # Last point
                        stationary_duration = point.timestamp - recent_points[stationary_start].timestamp
                        if stationary_duration >= self.stationary_threshold:
                            became_stationary = True
                else:
                    stationary_start = -1  # Reset if movement detected
        
        # Phase 4: Calculate fall distance
        fall_distance = 0.0
        if falling_detected and fall_start_idx > 0:
            start_point = recent_points[fall_start_idx - 1]
            end_point = recent_points[-1]
            fall_distance = end_point.y - start_point.y
        
        # Phase 5: Determine if this is a valid drop event
        is_drop_event = (
            was_near_person and 
            falling_detected and 
            became_stationary and 
            fall_distance >= self.min_fall_distance
        )
        
        if is_drop_event:
            # Calculate confidence based on various factors
            confidence = self._calculate_drop_confidence(
                was_near_person, max_fall_velocity, fall_distance, 
                len(recent_points), became_stationary
            )
            
            # Create drop event
            drop_location = (recent_points[-1].x, recent_points[-1].y)
            
            # Get object label (try to infer from context)
            object_label = self._infer_object_label(obj_id, trajectory)
            
            event = DropEvent(
                event_id=f"drop_{self.next_event_id}",
                object_id=obj_id,
                person_id=nearest_person_id,
                drop_location=drop_location,
                drop_time=datetime.fromtimestamp(timestamp),
                trajectory_data=[(p.x, p.y, p.timestamp) for p in recent_points],
                confidence=confidence,
                object_label=object_label,
                drop_height=fall_distance,
                drop_velocity=max_fall_velocity
            )
            
            self.next_event_id += 1
            return event
        
        return None
    
    def _calculate_drop_confidence(self, was_near_person: bool, max_velocity: float,
                                 fall_distance: float, trajectory_length: int,
                                 became_stationary: bool) -> float:
        """Calculate confidence score for drop event."""
        confidence = 0.0
        
        # Base confidence from trajectory quality
        confidence += min(trajectory_length / self.trajectory_window, 1.0) * 0.2
        
        # Proximity to person
        if was_near_person:
            confidence += 0.3
        
        # Fall velocity (higher velocity = more confident)
        velocity_score = min(max_velocity / (self.fall_velocity_threshold * 2), 1.0)
        confidence += velocity_score * 0.2
        
        # Fall distance (longer fall = more confident)
        distance_score = min(fall_distance / (self.min_fall_distance * 2), 1.0)
        confidence += distance_score * 0.2
        
        # Became stationary after fall
        if became_stationary:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _infer_object_label(self, obj_id: str, trajectory: List[TrajectoryPoint]) -> str:
        """Infer object label from context (placeholder for now)."""
        # This could be enhanced to use object detection labels
        # or other contextual information
        return "dropped_object"
    
    def _cleanup_old_trajectories(self, timestamp: float, max_age: float = 30.0):
        """Remove old trajectory data."""
        to_remove = []
        
        for obj_id, trajectory in self.object_trajectories.items():
            if trajectory and timestamp - trajectory[-1].timestamp > max_age:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.object_trajectories[obj_id]
    
    def get_recent_drops(self, time_window: float = 60.0) -> List[DropEvent]:
        """Get drop events from recent time window."""
        current_time = datetime.now()
        recent_drops = []
        
        for drop in self.detected_drops:
            age = (current_time - drop.drop_time).total_seconds()
            if age <= time_window:
                recent_drops.append(drop)
        
        return recent_drops
    
    def reset(self):
        """Reset detector state."""
        self.object_trajectories.clear()
        self.person_positions.clear()
        self.detected_drops.clear()
        self.next_event_id = 1
        logger.info("Drop event detector reset")