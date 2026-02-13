"""
Enhanced Object Detector with Industry-Standard 3-Layer Detection System

This module provides advanced object detection capabilities including:
- Layer 1: Enhanced small object detection with multi-scale processing
- Layer 2: Stationary object detection logic
- Layer 3: Drop event detection using trajectory analysis
- Fallback: Motion detection when YOLO fails
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .yolo_detector import YOLODetector
from .stationary_detector import StationaryObjectDetector, StationaryObject
from .drop_detector import DropEventDetector, DropEvent
from .motion_fallback import MotionFallbackDetector, MotionRegion

logger = logging.getLogger(__name__)


@dataclass
class ObjectState:
    """Represents the state of a detected object with enhanced tracking."""
    object_id: str
    bbox: Tuple[int, int, int, int]
    label: str
    confidence: float
    state: str  # 'normal', 'stationary', 'dropped', 'picked_up', 'falling'
    last_seen: datetime
    stationary_duration: float = 0.0
    person_nearby: bool = False
    interaction_detected: bool = False
    pickup_attempt: bool = False
    picked_up_by: Optional[str] = None
    pickup_timestamp: Optional[datetime] = None
    original_position: Optional[Tuple[int, int]] = None
    tracking_priority: bool = False
    
    # Enhanced tracking data
    velocity_history: List[float] = None
    position_history: List[Tuple[int, int]] = None
    drop_event: Optional[DropEvent] = None
    motion_detected: bool = False

    def __post_init__(self):
        if self.velocity_history is None:
            self.velocity_history = []
        if self.position_history is None:
            self.position_history = []


@dataclass
class PersonDetection:
    """Enhanced person detection with trajectory tracking."""
    person_id: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    last_seen: datetime
    trajectory: List[Tuple[int, int]] = None
    near_objects: Set[str] = None
    suspicious_behavior: bool = False
    carrying_objects: Set[str] = None
    pickup_history: List[str] = None
    
    # Enhanced tracking
    velocity: float = 0.0
    direction: float = 0.0  # Angle in degrees

    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = []
        if self.near_objects is None:
            self.near_objects = set()
        if self.carrying_objects is None:
            self.carrying_objects = set()
        if self.pickup_history is None:
            self.pickup_history = []


@dataclass
class InteractionEvent:
    """Enhanced interaction event with trajectory data."""
    event_id: str
    person_id: str
    object_id: str
    interaction_type: str
    confidence: float
    timestamp: datetime
    duration: float = 0.0
    bbox_person: Tuple[int, int, int, int] = None
    bbox_object: Tuple[int, int, int, int] = None
    alert_level: str = "info"
    
    # Enhanced data
    trajectory_data: List[Tuple[int, int, float]] = None
    drop_event: Optional[DropEvent] = None

    def __post_init__(self):
        if self.trajectory_data is None:
            self.trajectory_data = []


class IndustryStandardDetector:
    """
    Industry-standard 3-layer detection system:
    Layer 1: Enhanced small object detection
    Layer 2: Stationary object detection
    Layer 3: Drop event detection
    Fallback: Motion detection
    """
    
    def __init__(self,
                 detection_confidence: float = 0.2,
                 stationary_threshold: float = 3.0,
                 proximity_threshold: float = 100.0,
                 interaction_threshold: float = 50.0):
        """
        Initialize the industry-standard detection system.
        
        Args:
            detection_confidence: YOLO detection confidence threshold
            stationary_threshold: Time before object is considered stationary
            proximity_threshold: Distance for person-object proximity
            interaction_threshold: Distance for interaction detection
        """
        # Layer 1: Enhanced YOLO detector
        self.yolo_detector = YOLODetector()
        
        # Layer 2: Stationary object detector
        self.stationary_detector = StationaryObjectDetector(
            stationary_threshold=stationary_threshold,
            velocity_threshold=5.0,
            min_stationary_duration=2.0
        )
        
        # Layer 3: Drop event detector
        self.drop_detector = DropEventDetector(
            fall_velocity_threshold=50.0,
            min_fall_distance=30.0,
            proximity_threshold=proximity_threshold
        )
        
        # Fallback: Motion detector
        self.motion_detector = MotionFallbackDetector(
            motion_threshold=25,
            min_contour_area=500
        )
        
        # Configuration
        self.detection_confidence = detection_confidence
        self.proximity_threshold = proximity_threshold
        self.interaction_threshold = interaction_threshold
        
        # Tracking state
        self.tracked_objects: Dict[str, ObjectState] = {}
        self.tracked_persons: Dict[str, PersonDetection] = {}
        self.interaction_events: List[InteractionEvent] = []
        self.next_object_id = 1
        self.next_person_id = 1
        self.next_event_id = 1
        
        logger.info("Industry-standard detector initialized")
    
    def process_frame(self, frame: np.ndarray, timestamp: float) -> Tuple[List[ObjectState], List[PersonDetection], List[InteractionEvent], List[DropEvent]]:
        """
        Process frame using industry-standard 3-layer detection.
        
        Args:
            frame: Input video frame
            timestamp: Frame timestamp
            
        Returns:
            Tuple of (objects, persons, interactions, drop_events)
        """
        # Layer 1: Enhanced object detection
        yolo_detections = self.yolo_detector.detect(frame, "camera", timestamp)
        
        # Fallback: Motion detection (when YOLO fails)
        motion_regions = self.motion_detector.detect_motion(frame, timestamp)
        
        # Combine YOLO and motion detections
        all_detections = self._combine_detections(yolo_detections, motion_regions, timestamp)
        
        # Separate objects and persons
        object_detections = [d for d in all_detections if d['label'] != 'person']
        person_detections = [d for d in all_detections if d['label'] == 'person']
        
        # Layer 2: Update stationary object tracking
        stationary_objects = self.stationary_detector.process_frame(
            frame, object_detections, timestamp
        )
        
        # Layer 3: Detect drop events
        drop_events = self.drop_detector.process_frame(
            object_detections, person_detections, timestamp
        )
        
        # Update object and person tracking
        tracked_objects = self._update_object_tracking(object_detections, stationary_objects, drop_events, timestamp)
        tracked_persons = self._update_person_tracking(person_detections, timestamp)
        
        # Detect interactions
        interactions = self._detect_interactions(tracked_objects, tracked_persons, timestamp)
        
        return tracked_objects, tracked_persons, interactions, drop_events
    
    def _combine_detections(self, yolo_detections: List[Dict], motion_regions: List[MotionRegion], timestamp: float) -> List[Dict]:
        """Combine YOLO detections with motion fallback detections."""
        combined = yolo_detections.copy()
        
        # Add motion regions as fallback detections where YOLO missed
        for region in motion_regions:
            # Check if this motion region overlaps with any YOLO detection
            overlaps = False
            for yolo_det in yolo_detections:
                if self._bboxes_overlap(region.bbox, tuple(yolo_det['bbox'])):
                    overlaps = True
                    break
            
            # If no overlap, add as fallback detection
            if not overlaps:
                fallback_detection = {
                    'bbox': list(region.bbox),
                    'confidence': min(0.5, region.motion_intensity / 100.0),  # Convert intensity to confidence
                    'label': 'unknown_object',  # Generic label for motion-detected objects
                    'camera_id': 'camera',
                    'timestamp': timestamp,
                    'detection_id': f"motion_{timestamp}_{len(combined)}",
                    'source': 'motion_fallback'
                }
                combined.append(fallback_detection)
        
        return combined
    
    def _bboxes_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> bool:
        """Check if two bounding boxes overlap."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)
    
    def _update_object_tracking(self, detections: List[Dict], stationary_objects: List[StationaryObject], 
                              drop_events: List[DropEvent], timestamp: float) -> List[ObjectState]:
        """Update object tracking with enhanced state management."""
        current_time = datetime.fromtimestamp(timestamp)
        updated_objects = []
        
        # Create mapping of stationary objects
        stationary_map = {obj.object_id: obj for obj in stationary_objects}
        
        # Create mapping of drop events
        drop_map = {}
        for event in drop_events:
            drop_map[event.object_id] = event
        
        # Update existing objects and add new ones
        detection_centers = [(self._get_bbox_center(d['bbox']), d) for d in detections]
        matched_detections = set()
        
        # Match existing objects
        for obj_id, obj in list(self.tracked_objects.items()):
            obj_center = self._get_bbox_center(obj.bbox)
            
            # Find closest detection
            min_distance = float('inf')
            best_match = None
            best_idx = -1
            
            for i, (det_center, detection) in enumerate(detection_centers):
                if i in matched_detections:
                    continue
                
                distance = np.sqrt((obj_center[0] - det_center[0])**2 + (obj_center[1] - det_center[1])**2)
                if distance < min_distance and distance < 100:  # Max tracking distance
                    min_distance = distance
                    best_match = detection
                    best_idx = i
            
            if best_match:
                # Update existing object
                obj.bbox = tuple(best_match['bbox'])
                obj.confidence = best_match['confidence']
                obj.last_seen = current_time
                
                # Update state based on stationary detector
                if obj_id in stationary_map:
                    stationary_obj = stationary_map[obj_id]
                    obj.state = 'stationary' if stationary_obj.is_stationary else 'normal'
                    if stationary_obj.is_dropped:
                        obj.state = 'dropped'
                
                # Update state based on drop events
                if obj_id in drop_map:
                    obj.drop_event = drop_map[obj_id]
                    obj.state = 'dropped'
                
                matched_detections.add(best_idx)
                updated_objects.append(obj)
            else:
                # Object not detected, check if it should be kept
                age = (current_time - obj.last_seen).total_seconds()
                if age < 5.0:  # Keep for 5 seconds
                    updated_objects.append(obj)
        
        # Add new objects
        for i, (det_center, detection) in enumerate(detection_centers):
            if i not in matched_detections:
                obj_id = f"obj_{self.next_object_id}"
                self.next_object_id += 1
                
                new_obj = ObjectState(
                    object_id=obj_id,
                    bbox=tuple(detection['bbox']),
                    label=detection['label'],
                    confidence=detection['confidence'],
                    state='normal',
                    last_seen=current_time,
                    position_history=[det_center]
                )
                
                # Check if this is already stationary or dropped
                if obj_id in stationary_map:
                    stationary_obj = stationary_map[obj_id]
                    new_obj.state = 'stationary' if stationary_obj.is_stationary else 'normal'
                    if stationary_obj.is_dropped:
                        new_obj.state = 'dropped'
                
                if obj_id in drop_map:
                    new_obj.drop_event = drop_map[obj_id]
                    new_obj.state = 'dropped'
                
                updated_objects.append(new_obj)
        
        # Update tracking dictionary
        self.tracked_objects = {obj.object_id: obj for obj in updated_objects}
        
        return updated_objects
    
    def _update_person_tracking(self, detections: List[Dict], timestamp: float) -> List[PersonDetection]:
        """Update person tracking with enhanced features."""
        current_time = datetime.fromtimestamp(timestamp)
        updated_persons = []
        
        detection_centers = [(self._get_bbox_center(d['bbox']), d) for d in detections]
        matched_detections = set()
        
        # Match existing persons
        for person_id, person in list(self.tracked_persons.items()):
            person_center = self._get_bbox_center(person.bbox)
            
            # Find closest detection
            min_distance = float('inf')
            best_match = None
            best_idx = -1
            
            for i, (det_center, detection) in enumerate(detection_centers):
                if i in matched_detections:
                    continue
                
                distance = np.sqrt((person_center[0] - det_center[0])**2 + (person_center[1] - det_center[1])**2)
                if distance < min_distance and distance < 150:  # Max tracking distance for persons
                    min_distance = distance
                    best_match = detection
                    best_idx = i
            
            if best_match:
                # Update existing person
                old_center = person_center
                new_center = self._get_bbox_center(best_match['bbox'])
                
                person.bbox = tuple(best_match['bbox'])
                person.confidence = best_match['confidence']
                person.last_seen = current_time
                
                # Update trajectory
                person.trajectory.append(new_center)
                if len(person.trajectory) > 20:
                    person.trajectory.pop(0)
                
                # Calculate velocity
                if len(person.trajectory) >= 2:
                    prev_pos = person.trajectory[-2]
                    distance = np.sqrt((new_center[0] - prev_pos[0])**2 + (new_center[1] - prev_pos[1])**2)
                    person.velocity = distance  # pixels per frame
                
                matched_detections.add(best_idx)
                updated_persons.append(person)
            else:
                # Person not detected, check if it should be kept
                age = (current_time - person.last_seen).total_seconds()
                if age < 3.0:  # Keep for 3 seconds
                    updated_persons.append(person)
        
        # Add new persons
        for i, (det_center, detection) in enumerate(detection_centers):
            if i not in matched_detections:
                person_id = f"person_{self.next_person_id}"
                self.next_person_id += 1
                
                new_person = PersonDetection(
                    person_id=person_id,
                    bbox=tuple(detection['bbox']),
                    confidence=detection['confidence'],
                    last_seen=current_time,
                    trajectory=[det_center]
                )
                
                updated_persons.append(new_person)
        
        # Update tracking dictionary
        self.tracked_persons = {person.person_id: person for person in updated_persons}
        
        return updated_persons
    
    def _detect_interactions(self, objects: List[ObjectState], persons: List[PersonDetection], 
                           timestamp: float) -> List[InteractionEvent]:
        """Detect person-object interactions with enhanced logic."""
        interactions = []
        current_time = datetime.fromtimestamp(timestamp)
        
        for person in persons:
            person_center = self._get_bbox_center(person.bbox)
            
            for obj in objects:
                obj_center = self._get_bbox_center(obj.bbox)
                distance = np.sqrt((person_center[0] - obj_center[0])**2 + (person_center[1] - obj_center[1])**2)
                
                # Check for proximity
                if distance < self.proximity_threshold:
                    person.near_objects.add(obj.object_id)
                    obj.person_nearby = True
                    
                    # Check for pickup attempt
                    if distance < self.interaction_threshold:
                        interaction_type = "pickup_attempt"
                        alert_level = "warning"
                        
                        # Check if object was dropped by this person
                        if obj.drop_event and obj.drop_event.person_id == person.person_id:
                            interaction_type = "retrieving_own_item"
                            alert_level = "info"
                        
                        # Check if object state changed to picked up
                        if obj.state == 'picked_up':
                            interaction_type = "item_picked_up"
                            alert_level = "critical"
                            obj.picked_up_by = person.person_id
                            obj.pickup_timestamp = current_time
                            person.carrying_objects.add(obj.object_id)
                        
                        event = InteractionEvent(
                            event_id=f"interaction_{self.next_event_id}",
                            person_id=person.person_id,
                            object_id=obj.object_id,
                            interaction_type=interaction_type,
                            confidence=min(person.confidence, obj.confidence),
                            timestamp=current_time,
                            bbox_person=person.bbox,
                            bbox_object=obj.bbox,
                            alert_level=alert_level,
                            drop_event=obj.drop_event
                        )
                        
                        interactions.append(event)
                        self.next_event_id += 1
        
        return interactions
    
    def _get_bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def create_enhanced_overlay(self, frame: np.ndarray, objects: List[ObjectState], 
                              persons: List[PersonDetection], interactions: List[InteractionEvent],
                              drop_events: List[DropEvent]) -> np.ndarray:
        """Create enhanced overlay with all detection information."""
        overlay = frame.copy()
        
        # Draw objects with state-based colors
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox
            
            # Color based on state
            if obj.state == 'dropped':
                color = (0, 0, 255)    # Red for dropped
            elif obj.state == 'stationary':
                color = (0, 255, 255)  # Yellow for stationary
            elif obj.state == 'picked_up':
                color = (255, 0, 255)  # Magenta for picked up
            else:
                color = (0, 255, 0)    # Green for normal
            
            # Draw bounding box
            thickness = 3 if obj.state in ['dropped', 'picked_up'] else 2
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with state
            label_text = f"{obj.label} ({obj.state})"
            if obj.drop_event:
                label_text += f" [DROP: {obj.drop_event.confidence:.2f}]"
            
            cv2.putText(overlay, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw persons
        for person in persons:
            x1, y1, x2, y2 = person.bbox
            color = (255, 0, 0) if person.suspicious_behavior else (255, 255, 0)
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            label = f"Person {person.person_id}"
            if person.carrying_objects:
                label += f" [Carrying: {len(person.carrying_objects)}]"
            
            cv2.putText(overlay, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw trajectory
            if len(person.trajectory) > 1:
                points = np.array(person.trajectory, np.int32)
                cv2.polylines(overlay, [points], False, color, 1)
        
        # Draw interactions
        for interaction in interactions:
            if interaction.bbox_person and interaction.bbox_object:
                person_center = self._get_bbox_center(interaction.bbox_person)
                object_center = self._get_bbox_center(interaction.bbox_object)
                
                color = (0, 0, 255) if interaction.alert_level == 'critical' else (255, 0, 255)
                cv2.line(overlay, person_center, object_center, color, 3)
                
                # Draw interaction label
                mid_point = ((person_center[0] + object_center[0]) // 2,
                           (person_center[1] + object_center[1]) // 2)
                cv2.putText(overlay, interaction.interaction_type.upper(), 
                           mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw drop events
        for drop in drop_events:
            x, y = drop.drop_location
            cv2.circle(overlay, (x, y), 15, (0, 0, 255), 3)
            cv2.putText(overlay, f"DROP: {drop.confidence:.2f}", 
                       (x-30, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return overlay
    
    def reset(self):
        """Reset all detector states."""
        self.tracked_objects.clear()
        self.tracked_persons.clear()
        self.interaction_events.clear()
        self.next_object_id = 1
        self.next_person_id = 1
        self.next_event_id = 1
        
        self.stationary_detector.reset()
        self.drop_detector.reset()
        self.motion_detector.reset()
        
        logger.info("Industry-standard detector reset")
    acknowledged: bool = False


class EnhancedObjectDetector:
    """
    Enhanced detector that tracks objects, people, and their interactions.
    Includes pickup detection and person tracking for stolen items.
    """
    
    def __init__(self, 
                 stationary_threshold: float = 3.0,  # seconds
                 proximity_threshold: float = 100.0,  # pixels
                 interaction_threshold: float = 50.0,  # pixels
                 pickup_confidence_threshold: float = 0.7,
                 tracking_distance_threshold: float = 150.0,  # pixels for tracking
                 detection_confidence: float = 0.15):  # Lower for small objects
        """
        Initialize enhanced detector.
        
        Args:
            stationary_threshold: Time before object is considered stationary
            proximity_threshold: Distance for person-object proximity
            interaction_threshold: Distance for interaction detection
            pickup_confidence_threshold: Confidence for pickup detection
            tracking_distance_threshold: Max distance for person tracking
            detection_confidence: Minimum confidence for object detection
        """
        self.base_detector = YOLODetector()
        
        self.stationary_threshold = stationary_threshold
        self.proximity_threshold = proximity_threshold
        self.interaction_threshold = interaction_threshold
        self.pickup_confidence_threshold = pickup_confidence_threshold
        self.tracking_distance_threshold = tracking_distance_threshold
        self.detection_confidence = detection_confidence
        
        # Tracking state
        self.tracked_objects: Dict[str, ObjectState] = {}
        self.tracked_persons: Dict[str, PersonDetection] = {}
        self.interaction_history: List[InteractionEvent] = []
        self.pickup_alerts: List[PickupAlert] = []
        
        # Object ID counters
        self.next_object_id = 1
        self.next_person_id = 1
        self.next_event_id = 1
        self.next_alert_id = 1
        
        logger.info("Enhanced object detector initialized with improved small object detection")
    
    def detect_and_track(self, frame: np.ndarray, camera_id: str, 
                        timestamp: float) -> Tuple[List[ObjectState], 
                                                  List[PersonDetection], 
                                                  List[InteractionEvent]]:
        """
        Detect objects and people, track their states and interactions.
        
        Args:
            frame: Input video frame
            camera_id: Camera identifier
            timestamp: Frame timestamp
        
        Returns:
            Tuple of (objects, persons, interactions)
        """
        current_time = datetime.fromtimestamp(timestamp)
        
        # Get base detections with improved filtering
        detections = self.base_detector.detect(frame, camera_id, timestamp)
        if not detections:
            return [], [], []
        
        # Filter detections with improved small object handling
        filtered_detections = []
        for det in detections:
            confidence = det.get('confidence', 0)
            area = det.get('area', 0)
            
            # Dynamic confidence threshold based on object size
            if area < 500:  # Very small objects
                min_conf = 0.15
            elif area < 2000:  # Small objects
                min_conf = 0.20
            else:  # Regular objects
                min_conf = 0.25
            
            if confidence >= min_conf:
                filtered_detections.append(det)
        
        logger.debug(f"Filtered {len(filtered_detections)} from {len(detections)} detections")
        
        # Separate persons and objects
        person_detections = []
        object_detections = []
        
        for det in filtered_detections:
            label = det.get('label', 'unknown')
            
            if label == 'person':
                person_detections.append(det)
            else:
                object_detections.append(det)
        
        # Update person tracking
        tracked_persons = self._update_person_tracking(
            person_detections, current_time, frame
        )
        
        # Update object tracking
        tracked_objects = self._update_object_tracking(
            object_detections, current_time, frame
        )
        
        # Detect interactions
        interactions = self._detect_interactions(
            tracked_persons, tracked_objects, current_time, frame
        )
        
        return tracked_objects, tracked_persons, interactions
    
    def _update_person_tracking(self, detections: List, current_time: datetime,
                               frame: np.ndarray) -> List[PersonDetection]:
        """Update person tracking with new detections."""
        updated_persons = []
        
        for det in detections:
            # Extract detection info
            if hasattr(det, 'bbox'):
                bbox = det.bbox
                confidence = getattr(det, 'confidence', 0.8)
            elif isinstance(det, dict):
                bbox = det.get('bbox', (0, 0, 0, 0))
                confidence = det.get('confidence', 0.8)
            else:
                continue
            
            # Find matching existing person or create new
            person_id = self._match_or_create_person(bbox, current_time)
            
            # Update person state
            if person_id in self.tracked_persons:
                person = self.tracked_persons[person_id]
                person.bbox = bbox
                person.confidence = confidence
                person.last_seen = current_time
                
                # Update trajectory
                if person.trajectory is None:
                    person.trajectory = []
                
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                person.trajectory.append((center_x, center_y))
                
                # Keep only recent trajectory points (last 30 frames)
                if len(person.trajectory) > 30:
                    person.trajectory = person.trajectory[-30:]
            else:
                # Create new person
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                
                person = PersonDetection(
                    person_id=person_id,
                    bbox=bbox,
                    confidence=confidence,
                    last_seen=current_time,
                    trajectory=[(center_x, center_y)],
                    near_objects=set(),
                    suspicious_behavior=False,
                    carrying_objects=set(),
                    pickup_history=[]
                )
                self.tracked_persons[person_id] = person
            
            updated_persons.append(person)
        
        # Remove old persons
        self._cleanup_old_persons(current_time)
        
        return updated_persons
    
    def _update_object_tracking(self, detections: List, current_time: datetime,
                               frame: np.ndarray) -> List[ObjectState]:
        """Update object tracking with new detections."""
        updated_objects = []
        
        for det in detections:
            # Extract detection info
            if hasattr(det, 'bbox'):
                bbox = det.bbox
                label = getattr(det, 'label', 'object')
                confidence = getattr(det, 'confidence', 0.8)
            elif isinstance(det, dict):
                bbox = det.get('bbox', (0, 0, 0, 0))
                label = det.get('label', 'object')
                confidence = det.get('confidence', 0.8)
            else:
                continue
            
            # Find matching existing object or create new
            object_id = self._match_or_create_object(bbox, label, current_time)
            
            # Update object state
            if object_id in self.tracked_objects:
                obj = self.tracked_objects[object_id]
                
                # Calculate movement
                old_center = ((obj.bbox[0] + obj.bbox[2]) // 2, 
                             (obj.bbox[1] + obj.bbox[3]) // 2)
                new_center = ((bbox[0] + bbox[2]) // 2, 
                             (bbox[1] + bbox[3]) // 2)
                
                movement = np.sqrt((new_center[0] - old_center[0])**2 + 
                                 (new_center[1] - old_center[1])**2)
                
                # Update state
                obj.bbox = bbox
                obj.confidence = confidence
                obj.last_seen = current_time
                
                # Determine object state
                if movement < 10:  # Minimal movement threshold
                    obj.stationary_duration += 0.033  # Assume ~30fps
                    if obj.stationary_duration > self.stationary_threshold:
                        if obj.state != 'stationary':
                            # Object just became stationary - record position
                            obj.original_position = new_center
                            obj.tracking_priority = True
                        obj.state = 'stationary'
                    else:
                        obj.state = 'dropped' if obj.state != 'stationary' else 'stationary'
                else:
                    # Check if this was a stationary object that's now moving
                    if obj.state in ['stationary', 'dropped'] and movement > 30:
                        # Likely picked up!
                        obj.state = 'picked_up'
                        obj.pickup_timestamp = current_time
                        obj.tracking_priority = True
                        
                        # Find nearby person who likely picked it up
                        nearby_person = self._find_nearest_person(new_center, current_time)
                        if nearby_person:
                            obj.picked_up_by = nearby_person.person_id
                            nearby_person.carrying_objects.add(obj.object_id)
                            nearby_person.pickup_history.append(obj.object_id)
                            nearby_person.suspicious_behavior = True
                            
                            # Create pickup alert
                            self._create_pickup_alert(obj, nearby_person, current_time)
                    else:
                        obj.stationary_duration = 0.0
                        obj.state = 'moving'
            else:
                # Create new object
                obj = ObjectState(
                    object_id=object_id,
                    bbox=bbox,
                    label=label,
                    confidence=confidence,
                    state='moving',
                    last_seen=current_time,
                    stationary_duration=0.0,
                    person_nearby=False,
                    interaction_detected=False,
                    pickup_attempt=False,
                    picked_up_by=None,
                    pickup_timestamp=None,
                    original_position=None,
                    tracking_priority=False
                )
                self.tracked_objects[object_id] = obj
            
            updated_objects.append(obj)
        
        # Remove old objects
        self._cleanup_old_objects(current_time)
        
        return updated_objects
    
    def _detect_interactions(self, persons: List[PersonDetection], 
                           objects: List[ObjectState], 
                           current_time: datetime,
                           frame: np.ndarray) -> List[InteractionEvent]:
        """Detect person-object interactions."""
        interactions = []
        
        for person in persons:
            person_center = ((person.bbox[0] + person.bbox[2]) // 2,
                           (person.bbox[1] + person.bbox[3]) // 2)
            
            for obj in objects:
                obj_center = ((obj.bbox[0] + obj.bbox[2]) // 2,
                             (obj.bbox[1] + obj.bbox[3]) // 2)
                
                # Calculate distance
                distance = np.sqrt((person_center[0] - obj_center[0])**2 + 
                                 (person_center[1] - obj_center[1])**2)
                
                # Check for proximity
                if distance < self.proximity_threshold:
                    obj.person_nearby = True
                    person.near_objects.add(obj.object_id)
                    
                    # Check for interaction
                    if distance < self.interaction_threshold:
                        obj.interaction_detected = True
                        
                        # Detect pickup attempt
                        pickup_detected = self._detect_pickup_attempt(
                            person, obj, frame
                        )
                        
                        if pickup_detected:
                            obj.pickup_attempt = True
                            person.suspicious_behavior = True
                            
                            # Create interaction event
                            event = InteractionEvent(
                                event_id=f"event_{self.next_event_id}",
                                person_id=person.person_id,
                                object_id=obj.object_id,
                                interaction_type='pickup_attempt',
                                confidence=0.8,
                                timestamp=current_time,
                                bbox_person=person.bbox,
                                bbox_object=obj.bbox,
                                alert_level='warning'
                            )
                            
                            interactions.append(event)
                            self.interaction_history.append(event)
                            self.next_event_id += 1
                            
                            logger.warning(
                                f"PICKUP ATTEMPT DETECTED: Person {person.person_id} "
                                f"attempting to take {obj.label} (ID: {obj.object_id})"
                            )
        
        return interactions
    
    def _find_nearest_person(self, position: Tuple[int, int], 
                           current_time: datetime) -> Optional[PersonDetection]:
        """Find the nearest person to a given position."""
        min_distance = float('inf')
        nearest_person = None
        
        for person in self.tracked_persons.values():
            if (current_time - person.last_seen).total_seconds() > 1.0:
                continue  # Person too old
            
            person_center = ((person.bbox[0] + person.bbox[2]) // 2,
                           (person.bbox[1] + person.bbox[3]) // 2)
            
            distance = np.sqrt((position[0] - person_center[0])**2 + 
                             (position[1] - person_center[1])**2)
            
            if distance < min_distance and distance < self.tracking_distance_threshold:
                min_distance = distance
                nearest_person = person
        
        return nearest_person
    
    def _create_pickup_alert(self, obj: ObjectState, person: PersonDetection, 
                           timestamp: datetime):
        """Create an alert for a picked up item."""
        alert = PickupAlert(
            alert_id=f"alert_{self.next_alert_id}",
            object_id=obj.object_id,
            person_id=person.person_id,
            pickup_time=timestamp,
            original_position=obj.original_position or (0, 0),
            alert_message=f"🚨 ITEM PICKED UP: {obj.label} taken by Person {person.person_id}",
            severity='high' if obj.tracking_priority else 'medium',
            acknowledged=False
        )
        
        self.pickup_alerts.append(alert)
        self.next_alert_id += 1
        
        # Create interaction event for pickup
        pickup_event = InteractionEvent(
            event_id=f"event_{self.next_event_id}",
            person_id=person.person_id,
            object_id=obj.object_id,
            interaction_type='item_picked_up',
            confidence=0.9,
            timestamp=timestamp,
            bbox_person=person.bbox,
            bbox_object=obj.bbox,
            alert_level='critical'
        )
        
        self.interaction_history.append(pickup_event)
        self.next_event_id += 1
        
        logger.critical(alert.alert_message)
    
    def get_active_alerts(self) -> List[PickupAlert]:
        """Get all unacknowledged pickup alerts."""
        return [alert for alert in self.pickup_alerts if not alert.acknowledged]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a pickup alert."""
        for alert in self.pickup_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def get_person_tracking_info(self, person_id: str) -> Optional[Dict]:
        """Get detailed tracking information for a person."""
        if person_id not in self.tracked_persons:
            return None
        
        person = self.tracked_persons[person_id]
        return {
            'person_id': person_id,
            'current_bbox': person.bbox,
            'trajectory': person.trajectory,
            'carrying_objects': list(person.carrying_objects),
            'pickup_history': person.pickup_history,
            'suspicious_behavior': person.suspicious_behavior,
            'last_seen': person.last_seen,
            'total_pickups': len(person.pickup_history)
        }
    
    def _detect_pickup_attempt(self, person: PersonDetection, 
                              obj: ObjectState, frame: np.ndarray) -> bool:
        """
        Detect if a person is attempting to pick up an object.
        
        Uses heuristics based on:
        - Person's hand position relative to object
        - Object state (stationary objects are more likely to be picked up)
        - Proximity and overlap
        """
        # Check if object is stationary (more likely to be picked up)
        if obj.state not in ['stationary', 'dropped']:
            return False
        
        # Check overlap between person and object bounding boxes
        person_bbox = person.bbox
        obj_bbox = obj.bbox
        
        # Calculate intersection
        x1 = max(person_bbox[0], obj_bbox[0])
        y1 = max(person_bbox[1], obj_bbox[1])
        x2 = min(person_bbox[2], obj_bbox[2])
        y2 = min(person_bbox[3], obj_bbox[3])
        
        if x2 > x1 and y2 > y1:
            intersection_area = (x2 - x1) * (y2 - y1)
            obj_area = (obj_bbox[2] - obj_bbox[0]) * (obj_bbox[3] - obj_bbox[1])
            
            overlap_ratio = intersection_area / (obj_area + 1e-6)
            
            # If significant overlap, likely pickup attempt
            if overlap_ratio > 0.3:
                return True
        
        # Check if person's lower body is near object (bending down)
        person_bottom = person_bbox[3]
        obj_top = obj_bbox[1]
        
        if abs(person_bottom - obj_top) < 50:  # Person is close to object level
            return True
        
        return False
    
    def _match_or_create_person(self, bbox: Tuple, current_time: datetime) -> str:
        """Match detection to existing person or create new ID."""
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        
        # Find closest existing person
        min_distance = float('inf')
        best_match = None
        
        for person_id, person in self.tracked_persons.items():
            if (current_time - person.last_seen).total_seconds() > 2.0:
                continue  # Too old
            
            person_center = ((person.bbox[0] + person.bbox[2]) // 2,
                           (person.bbox[1] + person.bbox[3]) // 2)
            
            distance = np.sqrt((center[0] - person_center[0])**2 + 
                             (center[1] - person_center[1])**2)
            
            if distance < min_distance and distance < 100:  # Max matching distance
                min_distance = distance
                best_match = person_id
        
        if best_match:
            return best_match
        else:
            # Create new person ID
            new_id = f"person_{self.next_person_id}"
            self.next_person_id += 1
            return new_id
    
    def _match_or_create_object(self, bbox: Tuple, label: str, 
                               current_time: datetime) -> str:
        """Match detection to existing object or create new ID."""
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        
        # Find closest existing object with same label
        min_distance = float('inf')
        best_match = None
        
        for object_id, obj in self.tracked_objects.items():
            if obj.label != label:
                continue
            
            if (current_time - obj.last_seen).total_seconds() > 1.0:
                continue  # Too old
            
            obj_center = ((obj.bbox[0] + obj.bbox[2]) // 2,
                         (obj.bbox[1] + obj.bbox[3]) // 2)
            
            distance = np.sqrt((center[0] - obj_center[0])**2 + 
                             (center[1] - obj_center[1])**2)
            
            if distance < min_distance and distance < 80:  # Max matching distance
                min_distance = distance
                best_match = object_id
        
        if best_match:
            return best_match
        else:
            # Create new object ID
            new_id = f"obj_{self.next_object_id}"
            self.next_object_id += 1
            return new_id
    
    def _cleanup_old_persons(self, current_time: datetime):
        """Remove persons not seen for a while."""
        to_remove = []
        for person_id, person in self.tracked_persons.items():
            if (current_time - person.last_seen).total_seconds() > 5.0:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.tracked_persons[person_id]
    
    def _cleanup_old_objects(self, current_time: datetime):
        """Remove objects not seen for a while."""
        to_remove = []
        for object_id, obj in self.tracked_objects.items():
            if (current_time - obj.last_seen).total_seconds() > 10.0:
                to_remove.append(object_id)
        
        for object_id in to_remove:
            del self.tracked_objects[object_id]
    
    def get_statistics(self) -> Dict:
        """Get detection and tracking statistics."""
        active_alerts = self.get_active_alerts()
        
        return {
            'tracked_objects': len(self.tracked_objects),
            'tracked_persons': len(self.tracked_persons),
            'total_interactions': len(self.interaction_history),
            'pickup_attempts': len([e for e in self.interaction_history 
                                  if e.interaction_type == 'pickup_attempt']),
            'items_picked_up': len([e for e in self.interaction_history 
                                  if e.interaction_type == 'item_picked_up']),
            'active_alerts': len(active_alerts),
            'stationary_objects': len([o for o in self.tracked_objects.values() 
                                     if o.state == 'stationary']),
            'objects_with_nearby_persons': len([o for o in self.tracked_objects.values() 
                                              if o.person_nearby]),
            'persons_carrying_objects': len([p for p in self.tracked_persons.values() 
                                           if p.carrying_objects]),
            'high_priority_objects': len([o for o in self.tracked_objects.values() 
                                        if o.tracking_priority])
        }