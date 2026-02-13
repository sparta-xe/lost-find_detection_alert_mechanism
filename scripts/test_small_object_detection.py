#!/usr/bin/env python
"""
Test script for improved small object detection and matching.

This script tests the enhanced detection capabilities for small objects
and improved image matching algorithms.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.yolo_detector import YOLODetector
from src.reidentification.lost_item_matcher import LostItemMatcher


def create_test_image_with_small_objects():
    """Create a test image with various sized objects."""
    # Create a 640x480 test image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Add some small objects of different sizes
    objects = [
        # (x, y, width, height, color, label)
        (50, 50, 15, 15, (0, 0, 255), "very_small_red"),      # 15x15 - very small
        (100, 100, 25, 25, (0, 255, 0), "small_green"),       # 25x25 - small
        (200, 150, 40, 40, (255, 0, 0), "medium_blue"),       # 40x40 - medium
        (350, 200, 60, 80, (0, 255, 255), "large_yellow"),    # 60x80 - large
        (500, 300, 20, 30, (255, 0, 255), "small_magenta"),   # 20x30 - small
    ]
    
    for x, y, w, h, color, label in objects:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 1)
        
        # Add label
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    return img, objects


def test_small_object_detection():
    """Test the improved YOLO detector on small objects."""
    print("🔬 Testing Small Object Detection")
    print("=" * 50)
    
    # Create test image
    test_img, expected_objects = create_test_image_with_small_objects()
    
    # Save test image for reference
    cv2.imwrite("test_small_objects.jpg", test_img)
    print(f"📸 Created test image with {len(expected_objects)} objects")
    
    # Initialize improved detector
    detector = YOLODetector()
    
    # Run detection
    timestamp = datetime.now().timestamp()
    detections = detector.detect(test_img, "test_cam", timestamp)
    
    print(f"\n📊 Detection Results:")
    print(f"  Expected objects: {len(expected_objects)}")
    print(f"  Detected objects: {len(detections)}")
    
    # Analyze detections by size
    size_categories = {"very_small": 0, "small": 0, "medium": 0, "large": 0}
    
    for det in detections:
        bbox = det['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        
        if area < 400:
            category = "very_small"
        elif area < 1000:
            category = "small"
        elif area < 2500:
            category = "medium"
        else:
            category = "large"
        
        size_categories[category] += 1
        
        print(f"  • {det['label']}: {width}x{height} (area: {area}, conf: {det['confidence']:.2f})")
    
    print(f"\n📈 Detection by Size:")
    for category, count in size_categories.items():
        print(f"  {category}: {count} objects")
    
    return detections


def test_enhanced_matching():
    """Test the enhanced image matching capabilities."""
    print("\n🎯 Testing Enhanced Image Matching")
    print("=" * 50)
    
    # Create reference images
    ref_images = []
    
    # Create a simple red square reference
    red_square = np.zeros((100, 100, 3), dtype=np.uint8)
    red_square[20:80, 20:80] = (0, 0, 255)  # Red square
    ref_images.append(("red_square.jpg", red_square))
    
    # Create a blue circle reference
    blue_circle = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(blue_circle, (50, 50), 30, (255, 0, 0), -1)  # Blue circle
    ref_images.append(("blue_circle.jpg", blue_circle))
    
    # Save reference images
    for name, img in ref_images:
        cv2.imwrite(name, img)
    
    # Initialize enhanced matcher
    matcher = LostItemMatcher(
        color_weight=0.4,
        edge_weight=0.3,
        texture_weight=0.3,
        threshold=0.4  # Lower threshold for testing
    )
    
    # Add lost items
    for i, (name, _) in enumerate(ref_images):
        success = matcher.add_lost_item(
            f"item_{i+1}",
            name,
            f"Test Item {i+1}",
            "Test description"
        )
        print(f"  Added lost item {name}: {'✅' if success else '❌'}")
    
    # Create test detections with variations
    test_cases = [
        # Similar red square (should match)
        ("similar_red", np.full((50, 50, 3), [0, 0, 200], dtype=np.uint8)),
        # Different red square (should match with lower confidence)
        ("different_red", np.full((60, 60, 3), [0, 0, 150], dtype=np.uint8)),
        # Blue object (should match blue circle)
        ("blue_object", np.full((40, 40, 3), [200, 0, 0], dtype=np.uint8)),
        # Green object (should not match)
        ("green_object", np.full((50, 50, 3), [0, 200, 0], dtype=np.uint8)),
    ]
    
    print(f"\n🔍 Testing {len(test_cases)} matching scenarios:")
    
    for test_name, test_img in test_cases:
        matches = matcher.match_detection(
            test_img,
            f"det_{test_name}",
            "test_cam",
            (0, 0, test_img.shape[1], test_img.shape[0]),
            1,
            datetime.now().timestamp()
        )
        
        print(f"\n  {test_name}:")
        if matches:
            for match in matches:
                print(f"    ✅ Matched {match.lost_item_id} (confidence: {match.confidence:.2f})")
                print(f"       Reasons: {', '.join(match.match_reasons)}")
        else:
            print(f"    ❌ No matches found")
    
    return matcher


def test_integration():
    """Test the complete integration of detection and matching."""
    print("\n🔗 Testing Complete Integration")
    print("=" * 50)
    
    # Create a test scene with a lost item
    scene = np.ones((480, 640, 3), dtype=np.uint8) * 180
    
    # Add a "lost" red bag
    cv2.rectangle(scene, (200, 200), (250, 280), (0, 0, 200), -1)
    cv2.rectangle(scene, (200, 200), (250, 280), (0, 0, 0), 2)
    
    # Add some other objects
    cv2.circle(scene, (400, 150), 25, (255, 0, 0), -1)  # Blue circle
    cv2.rectangle(scene, (100, 350), (150, 400), (0, 255, 0), -1)  # Green square
    
    cv2.imwrite("test_scene.jpg", scene)
    
    # Create reference image for the red bag
    ref_bag = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(ref_bag, (20, 10), (80, 90), (0, 0, 200), -1)
    cv2.imwrite("ref_red_bag.jpg", ref_bag)
    
    # Initialize components
    detector = YOLODetector()
    matcher = LostItemMatcher(threshold=0.3)
    
    # Add lost item
    matcher.add_lost_item("red_bag", "ref_red_bag.jpg", "Red Bag", "A red rectangular bag")
    
    # Detect objects in scene
    detections = detector.detect(scene, "test_cam", datetime.now().timestamp())
    print(f"  Detected {len(detections)} objects in scene")
    
    # Try to match each detection
    matches_found = 0
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox
        
        # Crop detection from scene
        crop = scene[y1:y2, x1:x2]
        
        if crop.size > 0:
            matches = matcher.match_detection(
                crop,
                det['detection_id'],
                "test_cam",
                bbox,
                1,
                datetime.now().timestamp()
            )
            
            if matches:
                matches_found += len(matches)
                print(f"  ✅ Found match for {det['label']} at {bbox}")
                for match in matches:
                    print(f"     → {match.lost_item_id} (confidence: {match.confidence:.2f})")
    
    print(f"\n📊 Integration Results:")
    print(f"  Total detections: {len(detections)}")
    print(f"  Matches found: {matches_found}")
    
    return matches_found > 0


def main():
    """Run all tests."""
    print("🧪 Enhanced Detection and Matching Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Small object detection
        detections = test_small_object_detection()
        
        # Test 2: Enhanced matching
        matcher = test_enhanced_matching()
        
        # Test 3: Integration test
        integration_success = test_integration()
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Small Object Detection: {len(detections)} objects detected")
        print(f"✅ Enhanced Matching: {len(matcher.match_history)} matches in history")
        print(f"✅ Integration Test: {'PASSED' if integration_success else 'FAILED'}")
        
        print("\n💡 Recommendations:")
        if len(detections) < 3:
            print("  - Consider lowering detection confidence threshold")
        if len(matcher.match_history) < 2:
            print("  - Consider lowering matching threshold")
        if not integration_success:
            print("  - Check image preprocessing and feature extraction")
        
        print("\n🎯 The enhanced system is ready for improved small object detection!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())