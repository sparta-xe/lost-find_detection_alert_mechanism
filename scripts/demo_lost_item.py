"""
Quick Demo: Lost Item Identification System

This script demonstrates the lost item identification capabilities with:
1. Creating sample test images
2. Uploading them to the system
3. Running matching against video
4. Generating reports

Run this to see the system in action!
"""

import sys
import logging
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.escalation.lost_item_service import LostItemService, LostItemReporter
from scripts.enhanced_tracking import EnhancedTrackingPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_images():
    """Create sample test images for demonstration."""
    print("\n📸 Creating Sample Test Images...")
    
    # Create output directory
    sample_dir = Path("data/sample_items")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample 1: Red Rectangle (Backpack)
    img1 = np.ones((300, 400, 3), dtype=np.uint8) * 255
    cv2.rectangle(img1, (50, 50), (350, 250), (0, 0, 255), -1)  # Red
    cv2.putText(img1, "BACKPACK", (120, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, (255, 255, 255), 2)
    path1 = sample_dir / "sample_backpack.png"
    cv2.imwrite(str(path1), img1)
    print(f"   ✓ Created: {path1}")
    
    # Sample 2: Blue Square (Phone)
    img2 = np.ones((300, 400, 3), dtype=np.uint8) * 255
    cv2.rectangle(img2, (100, 80), (300, 220), (255, 0, 0), -1)  # Blue
    cv2.putText(img2, "PHONE", (140, 160), cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, (255, 255, 255), 2)
    path2 = sample_dir / "sample_phone.png"
    cv2.imwrite(str(path2), img2)
    print(f"   ✓ Created: {path2}")
    
    # Sample 3: Green Circle (Ball)
    img3 = np.ones((300, 400, 3), dtype=np.uint8) * 255
    cv2.circle(img3, (200, 150), 100, (0, 255, 0), -1)  # Green
    cv2.putText(img3, "BALL", (160, 160), cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, (0, 0, 0), 2)
    path3 = sample_dir / "sample_ball.png"
    cv2.imwrite(str(path3), img3)
    print(f"   ✓ Created: {path3}")
    
    return path1, path2, path3


def demo_upload_and_match():
    """Demonstrate uploading items and matching."""
    print("\n" + "="*70)
    print("🎬 LOST ITEM IDENTIFICATION DEMO")
    print("="*70)
    
    # Create sample images
    path1, path2, path3 = create_sample_images()
    
    # Initialize service
    print("\n📋 Initializing Lost Item Service...")
    service = LostItemService(match_threshold=0.6)
    
    # Upload items
    print("\n📤 Uploading Lost Items...")
    items = [
        (path1, "Red Backpack", "Medium-sized red backpack with zippers"),
        (path2, "Blue Phone", "Blue smartphone with protective case"),
        (path3, "Green Ball", "Green rubber ball for sports"),
    ]
    
    item_ids = []
    for image_path, name, description in items:
        success, item_id = service.upload_lost_item(
            str(image_path), name, description
        )
        if success:
            item_ids.append(item_id)
            print(f"   ✅ {item_id}: {name}")
        else:
            print(f"   ❌ Failed to upload {name}")
    
    # List items
    print("\n📍 Registered Lost Items:")
    registered = service.get_lost_items()
    for item in registered:
        print(f"   • {item['item_id']}: {item['name']}")
        print(f"     {item['description']}")
    
    # Try to match with videos
    print("\n🎥 Attempting to Match Against Available Videos...")
    
    video_dir = Path("data/test_clips")
    if video_dir.exists():
        videos = list(video_dir.glob("*.mp4"))
        
        if videos:
            video_path = videos[0]
            print(f"   Found video: {video_path.name}")
            
            # Initialize enhanced pipeline
            print("\n🔄 Running Enhanced Tracking Pipeline...")
            try:
                pipeline = EnhancedTrackingPipeline(
                    video_source=str(video_path),
                    lost_item_threshold=0.6
                )
                
                # Add lost items to pipeline
                for item_id, name, desc in items:
                    pipeline.add_lost_item(str(item_id), name, desc)
                
                # Run for limited frames
                print("   Processing up to 100 frames (for demo)...")
                pipeline.run(max_frames=100, verbose=False)
                
                # Print statistics
                print("\n📊 Pipeline Statistics:")
                stats = pipeline.get_statistics()
                print(f"   • Frames processed: {stats['frames_processed']}")
                print(f"   • Detections made: {stats['total_detections']}")
                print(f"   • Lost items matched: {stats['lost_item_matches']}")
                
            except Exception as e:
                logger.error(f"Error running pipeline: {e}")
        else:
            print("   ⚠️  No video files found in data/test_clips/")
    else:
        print("   ⚠️  No test videos available")
    
    # Generate report
    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)
    
    reporter = LostItemReporter(service)
    print(reporter.report_matches())
    
    # Export results
    output_file = "demo_results.json"
    if reporter.export_matches(output_file):
        print(f"\n💾 Results exported to {output_file}")
    
    print("\n✅ Demo Complete!")
    print("\nTo use with your own items:")
    print("  1. Take a clear photo of your lost item")
    print("  2. Run: python scripts/lost_item_upload.py --interactive")
    print("  3. Upload the image and monitor available camera feeds")
    print("\nFull documentation: docs/LOST_ITEM_IDENTIFICATION.md")


def demo_interactive():
    """Run interactive demo."""
    print("\n" + "="*70)
    print("🔍 LOST ITEM IDENTIFICATION - INTERACTIVE DEMO")
    print("="*70)
    
    # Initialize service
    service = LostItemService(match_threshold=0.6)
    reporter = LostItemReporter(service)
    
    while True:
        print("\n" + "-"*70)
        print("Options:")
        print("  1. Create sample test items")
        print("  2. View registered items")
        print("  3. View matches")
        print("  4. Generate report")
        print("  5. Export results")
        print("  0. Exit")
        print("-"*70)
        
        choice = input("\nSelect option (0-5): ").strip()
        
        if choice == "1":
            paths = create_sample_images()
            items = [
                (paths[0], "Red Backpack", "Medium-sized red backpack"),
                (paths[1], "Blue Phone", "Blue smartphone"),
                (paths[2], "Green Ball", "Green rubber ball"),
            ]
            
            for path, name, desc in items:
                success, item_id = service.upload_lost_item(str(path), name, desc)
                if success:
                    print(f"✅ Uploaded: {item_id} ({name})")
        
        elif choice == "2":
            items = service.get_lost_items()
            if items:
                print("\n📋 Registered Items:")
                for item in items:
                    print(f"  • {item['item_id']}: {item['name']}")
            else:
                print("\n📭 No items registered")
        
        elif choice == "3":
            matches = service.get_matches()
            if matches:
                print(f"\n🎯 {len(matches)} match(es) found:")
                for match in matches:
                    print(f"  • {match['lost_item_id']}: {match.get('confidence', 'N/A')}")
            else:
                print("\n❌ No matches found yet")
        
        elif choice == "4":
            print("\n" + reporter.report_matches())
        
        elif choice == "5":
            output = input("Enter output filename (default: demo_results.json): ").strip()
            if not output:
                output = "demo_results.json"
            
            if reporter.export_matches(output):
                print(f"\n✅ Exported to {output}")
            else:
                print(f"\n❌ Export failed")
        
        elif choice == "0":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("\n❌ Invalid option")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Lost Item Identification Demo"
    )
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run interactive demo")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick auto demo")
    
    args = parser.parse_args()
    
    if args.interactive:
        demo_interactive()
    else:
        demo_upload_and_match()


if __name__ == "__main__":
    main()
