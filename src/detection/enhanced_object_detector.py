"""
Enhanced Object Detector with Person and Interaction Detection

This module provides advanced object detection capabilities including:
- Person detection and tracking
- Object state detection (dropped, carried, stationary)
- Person-object interaction detection
- Theft/pickup attempt detection
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .yolo_detector import YOLODetector

logger = logging.getLogger(__name__)


@dataclass
class ObjectState:
    """Represents the state of a detected object."""
    object_id: str
    bbox: Tuple[int, int, int, int]
    label: str
    confidence: float
    state: str  # 'carried', 'dropped', 'stationary', 'moving', 'picked_up'
    last_seen: datetime
    stationary_duration: float = 0.0
    person_nearby: bool = False
    interaction_detected: bool = False
    pickup_attempt: bool = False
    picked_up_by: Optional[str] = None  # Person ID who picked it up
    pickup_timestamp: Optional[datetime] = None
    original_position: Optional[Tuple[int, int]] = None  # Where it was dropped
    tracking_priority: bool = False  # High priority for tracking


@dataclass
class PersonDetection:
    """Represents a detected person."""
    person_id: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    last_seen: datetime
    trajectory: List[Tuple[int, int]] = None  # Center points over time
    near_objects: Set[str] = None  # Object IDs near this person
    suspicious_behavior: bool = False
    carrying_objects: Set[str] = None  # Object IDs this person is carrying
    pickup_history: List[str] = None  # Objects this person has picked up


@dataclass
class InteractionEvent:
    """Represents a person-object interaction."""
    event_id: str
    person_id: str
    object_id: str
    interaction_type: str  # 'pickup_attempt', 'touching', 'near', 'theft_suspected', 'item_picked_up'
    confidence: float
    timestamp: datetime
    duration: float = 0.0
    bbox_person: Tuple[int, int, int, int] = None
    bbox_object: Tuple[int, int, int, int] = None
    alert_level: str = "info"  # 'info', 'warning', 'critical'


@dataclass
class PickupAlert:
    """Represents an alert for a picked up item."""
    alert_id: str
    object_id: str
    person_id: str
    pickup_time: datetime
    original_position: Tuple[int, int]
    alert_message: str
    severity: str  # 'low', 'medium', 'high'
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