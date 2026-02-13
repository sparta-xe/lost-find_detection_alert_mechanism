#!/usr/bin/env python
"""
Test script to demonstrate pickup detection functionality.

Creates a synthetic scenario where objects are dropped and then picked up
to test the enhanced detection system.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.enhanced_object_detector import EnhancedObjectDetector, ObjectState, PersonDetection


def create_test_frame(frame_num: int, scenario: str = "pickup") -> np.ndarray:
    """Create a test frame with objects and people for pickup detection."""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 240  # Light gray background
    
    if scenario == "pickup":
        if frame_num < 30:
            # Stationary object (bag) on the ground
            cv2.rectangle(frame, (300, 350), (380, 420), (50, 50, 200), -1)  # Red bag
            cv2.rectangle(frame, (300, 350), (380, 420), (0, 0, 0), 2)
            
        elif frame_num < 60:
            # Person approaches the bag
            person_x = 200 + (frame_num - 30) * 3  # Person moving towards bag
            cv2.rectangle(frame, (person_x, 200), (person_x + 60, 400), (100, 150, 100), -1)  # Person
            cv2.rectangle(frame, (person_x, 200), (person_x + 60, 400), (0, 0, 0), 2)
            
            # Bag still on ground
            cv2.rectangle(frame, (300, 350), (380, 420), (50, 50, 200), -1)
            cv2.rectangle(frame, (300, 350), (380, 420), (0, 0, 0), 2)
            
        else:
            # Person picks up bag and moves away
            person_x = 320 + (frame_num - 60) * 2  # Person moving away with bag
            cv2.rectangle(frame, (person_x, 200), (person_x + 60, 400), (100, 150, 100), -1)  # Person
            cv2.rectangle(frame, (person_x, 200), (person_x + 60, 400), (0, 0, 0), 2)
            
            # Bag is now carried by person (overlapping)
            bag_x = person_x + 20
            bag_y = 250
            cv2.rectangle(frame, (bag_x, bag_y), (bag_x + 40, bag_y + 35), (50, 50, 200), -1)
            cv2.rectangle(frame, (bag_x, bag_y), (bag_x + 40, bag_y + 35), (0, 0, 0), 2)
    
    # Add frame number for reference
    cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return frame


def simulate_detections(frame: np.ndarray, frame_num: int) -> tuple:
    """Simulate object and person detections for the test frame."""
    detections = []
    
    if frame_num < 30:
        # Only bag detected (stationary)
        detections.append({
            'bbox': (300, 350, 380, 420),
            'label': 'handbag',
            'confidence': 0.9
        })
    
    elif frame_num < 60:
        # Both person and bag detected
        person_x = 200 + (frame_num - 30) * 3
        detections.extend([
            {
                'bbox': (person_x, 200, person_x + 60, 400),
                'label': 'person',
                'confidence': 0.95
            },
            {
                'bbox': (300, 350, 380, 420),
                'label': 'handbag',
                'confidence': 0.9
            }
        ])
    
    else:
        # Person carrying bag (bag position changes)
        person_x = 320 + (frame_num - 60) * 2
        bag_x = person_x + 20
        bag_y = 250
        detections.extend([
            {
                'bbox': (person_x, 200, person_x + 60, 400),
                'label': 'person',
                'confidence': 0.95
            },
            {
                'bbox': (bag_x, bag_y, bag_x + 40, bag_y + 35),
                'label': 'handbag',
                'confidence': 0.85
            }
        ])
    
    return detections


def test_pickup_detection():
    """Test the pickup detection system with synthetic data."""
    print("🧪 Testing Pickup Detection System")
    print("=" * 50)
    
    # Initialize detector
    detector = EnhancedObjectDetector(
        stationary_threshold=1.0,  # 1 second for faster testing
        proximity_threshold=80.0,
        interaction_threshold=40.0
    )
    
    # Simulate video frames
    total_frames = 90
    
    for frame_num in range(1, total_frames + 1):
        # Create test frame
        frame = create_test_frame(frame_num)
        
        # Simulate detections
        detections = simulate_detections(frame, frame_num)
        
        # Mock the detector's detect method
        detector.base_detector.detect = lambda f, c, t: detections
        
        # Process frame
        timestamp = datetime.now().timestamp()
        objects, persons, interactions = detector.detect_and_track(
            frame, "test_cam", timestamp
        )
        
        # Print significant events
        if objects or persons or interactions:
            print(f"\nFrame {frame_num}:")
            
            for obj in objects:
                status = f"  📦 {obj.object_id}: {obj.label} ({obj.state})"
                if hasattr(obj, 'picked_up_by') and obj.picked_up_by:
                    status += f" [TAKEN BY: {obj.picked_up_by}]"
                if hasattr(obj, 'tracking_priority') and obj.tracking_priority:
                    status += " [HIGH PRIORITY]"
                print(status)
            
            for person in persons:
                status = f"  👤 {person.person_id}"
                if hasattr(person, 'carrying_objects') and person.carrying_objects:
                    status += f" [CARRYING: {len(person.carrying_objects)} objects]"
                if person.suspicious_behavior:
                    status += " [SUSPICIOUS]"
                print(status)
            
            for interaction in interactions:
                print(f"  🤝 {interaction.interaction_type}: {interaction.person_id} → {interaction.object_id}")
        
        # Check for pickup alerts
        active_alerts = detector.get_active_alerts()
        if active_alerts:
            for alert in active_alerts:
                if not alert.acknowledged:
                    print(f"  🚨 ALERT: {alert.alert_message}")
                    detector.acknowledge_alert(alert.alert_id)
    
    # Print final statistics
    stats = detector.get_statistics()
    print(f"\n📊 Final Statistics:")
    print(f"  Total Objects Tracked: {stats['tracked_objects']}")
    print(f"  Total Persons Tracked: {stats['tracked_persons']}")
    print(f"  Pickup Attempts: {stats['pickup_attempts']}")
    print(f"  Items Picked Up: {stats['items_picked_up']}")
    print(f"  Active Alerts: {stats['active_alerts']}")
    
    # Show person tracking details
    print(f"\n👤 Person Tracking Details:")
    for person_id in detector.tracked_persons.keys():
        tracking_info = detector.get_person_tracking_info(person_id)
        if tracking_info:
            print(f"  Person {person_id}:")
            print(f"    - Total Pickups: {tracking_info['total_pickups']}")
            print(f"    - Currently Carrying: {len(tracking_info['carrying_objects'])} objects")
            print(f"    - Suspicious: {tracking_info['suspicious_behavior']}")


if __name__ == "__main__":
    test_pickup_detection()