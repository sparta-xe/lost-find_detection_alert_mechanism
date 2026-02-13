#!/usr/bin/env python
"""
Debug Script for Screenshot Matching Issues
This script helps diagnose why screenshot matching might not be working.
"""
import cv2
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.reidentification.improved_matcher import ImprovedLostItemMatcher

def analyze_image_features(image_path: str):
    """Analyze features of an uploaded image."""
    print(f"🔍 Analyzing image: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"📐 Image dimensions: {image.shape}")
    
    # Create matcher and extract features
    matcher = ImprovedLostItemMatcher()
    features = matcher._extract_comprehensive_features(image)
    
    print(f"\n📊 Extracted Features:")
    for feature_name, feature_data in features.items():
        if isinstance(feature_data, np.ndarray):
            print(f"   • {feature_name}: shape {feature_data.shape}, range [{feature_data.min():.3f}, {feature_data.max():.3f}]")
        elif isinstance(feature_data, list):
            print(f"   • {feature_name}: {len(feature_data)} items")
        else:
            print(f"   • {feature_name}: {type(feature_data)}")
    
    # Analyze color distribution
    print(f"\n🎨 Color Analysis:")
    mean_color = np.mean(image, axis=(0, 1))
    print(f"   • Mean BGR: [{mean_color[0]:.1f}, {mean_color[1]:.1f}, {mean_color[2]:.1f}]")
    
    # Convert to HSV for better analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hsv = np.mean(hsv, axis=(0, 1))
    print(f"   • Mean HSV: [{mean_hsv[0]:.1f}, {mean_hsv[1]:.1f}, {mean_hsv[2]:.1f}]")
    
    # Analyze brightness and contrast
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    print(f"   • Brightness: {brightness:.1f}")
    print(f"   • Contrast: {contrast:.1f}")
    
    # Edge analysis
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    print(f"   • Edge density: {edge_density:.3f}")
    
    return features

def compare_two_images(image1_path: str, image2_path: str):
    """Compare two images and show similarity scores."""
    print(f"\n🔄 Comparing images:")
    print(f"   Image 1: {image1_path}")
    print(f"   Image 2: {image2_path}")
    
    # Check if files exist
    if not Path(image1_path).exists():
        print(f"❌ Image 1 not found: {image1_path}")
        return
    if not Path(image2_path).exists():
        print(f"❌ Image 2 not found: {image2_path}")
        return
    
    # Load images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    if img1 is None or img2 is None:
        print("❌ Failed to load one or both images")
        return
    
    # Create matcher
    matcher = ImprovedLostItemMatcher(threshold=0.1)  # Very low threshold
    
    # Extract features
    features1 = matcher._extract_comprehensive_features(img1)
    features2 = matcher._extract_comprehensive_features(img2)
    
    # Calculate similarity scores
    scores = matcher._calculate_similarity_scores(features1, features2)
    
    # Enhanced template matching
    template_score = matcher._enhanced_template_matching(img2, features1)
    scores['template_match'] = template_score
    
    # Final score
    final_score = matcher._calculate_final_score(scores)
    
    print(f"\n📊 Similarity Scores:")
    for score_name, score_value in scores.items():
        print(f"   • {score_name}: {score_value:.3f}")
    
    print(f"\n🎯 Final Score: {final_score:.3f}")
    print(f"   Would match with threshold 0.25: {'✅ YES' if final_score >= 0.25 else '❌ NO'}")
    print(f"   Would match with threshold 0.20: {'✅ YES' if final_score >= 0.20 else '❌ NO'}")
    print(f"   Would match with threshold 0.15: {'✅ YES' if final_score >= 0.15 else '❌ NO'}")

def test_with_user_images():
    """Interactive test with user-provided images."""
    print("🧪 Screenshot Matching Diagnostic Tool")
    print("=" * 50)
    
    # Get image paths from user
    lost_item_path = input("Enter path to lost item image: ").strip()
    if not lost_item_path:
        print("❌ No image path provided")
        return
    
    detection_path = input("Enter path to detection/screenshot image (or press Enter to skip): ").strip()
    
    # Analyze the lost item image
    analyze_image_features(lost_item_path)
    
    # If detection image provided, compare them
    if detection_path:
        compare_two_images(lost_item_path, detection_path)
    
    # Test with matcher
    print(f"\n🎯 Testing with ImprovedLostItemMatcher:")
    matcher = ImprovedLostItemMatcher(threshold=0.2)
    
    # Add lost item
    success = matcher.add_lost_item("user_item", lost_item_path, "User Item", "Test item")
    print(f"   Added lost item: {'✅ Success' if success else '❌ Failed'}")
    
    if success and detection_path:
        # Load detection image
        det_img = cv2.imread(detection_path)
        if det_img is not None:
            # Test matching
            matches = matcher.match_detection(
                detection_image=det_img,
                detection_id="user_detection",
                camera_id="test_cam",
                bbox=(0, 0, det_img.shape[1], det_img.shape[0]),
                frame_number=1,
                timestamp=1.0
            )
            
            print(f"   Found {len(matches)} match(es)")
            for match in matches:
                print(f"     • Confidence: {match.confidence:.3f}")
                print(f"     • Reasons: {', '.join(match.match_reasons)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line mode
        if len(sys.argv) == 2:
            analyze_image_features(sys.argv[1])
        elif len(sys.argv) == 3:
            compare_two_images(sys.argv[1], sys.argv[2])
    else:
        # Interactive mode
        test_with_user_images()