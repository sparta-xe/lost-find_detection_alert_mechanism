"""
Lost Item Upload & Matching Tool

Interactive CLI for uploading lost item images and matching them against
camera feeds in real-time. Takes a single image of a lost item and
automatically identifies it from video streams.
"""

import sys
import logging
from pathlib import Path
from argparse import ArgumentParser
import cv2
import numpy as np
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.escalation.lost_item_service import LostItemService, LostItemReporter
from src.detection.yolo_detector import YOLODetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LostItemMatcher:
    """Interactive lost item matching interface."""
    
    def __init__(self, config_path: str = "configs/test_config.yaml"):
        """Initialize the matcher."""
        self.service = LostItemService()
        self.reporter = LostItemReporter(self.service)
        
        # Initialize detector
        self.detector = YOLODetector()
        
        logger.info("LostItemMatcher initialized and ready")
    
    def upload_item(self, image_path: str, name: str, description: str = ""):
        """Upload a lost item image."""
        print("\n📸 Uploading Lost Item Image...")
        print(f"   Image: {image_path}")
        print(f"   Item: {name}")
        if description:
            print(f"   Description: {description}")
        
        success, result = self.service.upload_lost_item(
            image_path, name, description
        )
        
        if success:
            print(f"\n✅ SUCCESS: Item uploaded as {result}")
            print(f"   Item ID: {result}")
            print("   This item will now be matched against camera feeds")
        else:
            print(f"\n❌ ERROR: {result}")
        
        return success, result
    
    def match_video(self, video_path: str, confidence_threshold: float = 0.6):
        """Match lost items against a video file."""
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"\n❌ ERROR: Video file not found: {video_path}")
            return False
        
        lost_items = self.service.get_lost_items()
        if not lost_items:
            print("\n⚠️  No lost items registered. Please upload an image first.")
            return False
        
        print(f"\n🎬 Processing Video: {video_path}")
        print(f"   Lost items to match: {len(lost_items)}")
        print("   This will take a few minutes...\n")
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"❌ ERROR: Cannot open video: {video_path}")
                return False
            # Prepare output video writer for overlay visualization
            fps_real = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = video_path.with_name(video_path.stem + "_matches.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(out_path), fourcc, fps_real, (width, height))
            
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            matches_found = 0
            
            print(f"📊 Total frames: {total_frames}")
            print("   Processing frames...\n")
            
            # Derive camera_id from filename (e.g., cam1.mp4 -> cam1)
            camera_id = video_path.stem

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Show progress every 30 frames
                if frame_count % 30 == 0:
                    print(f"   ⏳ Processed {frame_count}/{total_frames} frames "
                          f"({frame_count*100//total_frames}%)")
                
                # Detect objects in frame
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                timestamp = frame_count / fps
                detections = self.detector.detect(frame, camera_id, timestamp)
                
                # Try to match each detection against lost items
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    
                    # Crop detection
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if crop.size == 0:
                        continue
                    
                    # Match against lost items
                    matches = self.service.matcher.match_detection(
                        crop, 
                        detection_id=det.get('detection_id', f"det_{frame_count}_{det.get('class_id','unknown')}"),
                        camera_id=camera_id,
                        bbox=det['bbox'],
                        frame_number=frame_count,
                        timestamp=timestamp
                    )
                    
                    # Annotate frame for visualization
                    color = (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = det.get('label', '')
                    cv2.putText(frame, f"{label} {det.get('confidence',0):.2f}", (x1, max(y1-6,0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                    if matches:
                        matches_found += len(matches)
                        for i, match in enumerate(matches):
                            print(f"\n🎯 MATCH FOUND!")
                            print(f"   Lost Item: {match.lost_item_id}")
                            print(f"   Confidence: {match.confidence:.1%}")
                            print(f"   Frame: {frame_count}/{total_frames}")
                            print(f"   Reasons: {', '.join(match.match_reasons)}")
                            # Draw match overlay (green)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            text = f"{match.lost_item_id} ({match.confidence:.2f})"
                            cv2.putText(frame, text, (x1, y2 + 15 + 15*i),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                # Write annotated frame to output
                if 'writer' in locals() and writer.isOpened():
                    writer.write(frame)
            
            cap.release()
            if 'writer' in locals() and writer.isOpened():
                writer.release()
                print(f"\n✅ Overlay video saved to: {out_path}")
                try:
                    # Open the video file with the default system player (Windows)
                    if os.name == 'nt':
                        os.startfile(str(out_path))
                except Exception:
                    pass
            
            print(f"\n{'='*50}")
            print(f"✅ Video Processing Complete")
            print(f"   Frames processed: {frame_count}")
            print(f"   Matches found: {matches_found}")
            print(f"{'='*50}")
            
            return True
        
        except Exception as e:
            print(f"\n❌ ERROR processing video: {e}")
            logger.exception(e)
            return False
    
    def list_items(self):
        """List all registered lost items."""
        items = self.service.get_lost_items()
        
        if not items:
            print("\n📭 No lost items registered yet")
            return
        
        print("\n" + "="*50)
        print("REGISTERED LOST ITEMS")
        print("="*50)
        
        for i, item in enumerate(items, 1):
            print(f"\n{i}. {item['item_id']}: {item['name']}")
            print(f"   Description: {item['description']}")
            print(f"   Uploaded: {item['upload_time']}")
        
        print(f"\n📊 Total: {len(items)} item(s)")
    
    def show_report(self):
        """Display the match report."""
        print("\n" + self.reporter.report_matches())
    
    def interactive_menu(self):
        """Run interactive menu."""
        while True:
            print("\n" + "="*50)
            print("🔍 LOST ITEM IDENTIFICATION SYSTEM")
            print("="*50)
            print("\n1. Upload Lost Item Image")
            print("2. Match Against Video")
            print("3. List Registered Items")
            print("4. View Match Report")
            print("5. Mark Item as Found")
            print("6. Export Results")
            print("0. Exit")
            print()
            
            choice = input("Select option (0-6): ").strip()
            
            if choice == "1":
                self._upload_menu()
            elif choice == "2":
                self._match_menu()
            elif choice == "3":
                self.list_items()
            elif choice == "4":
                self.show_report()
            elif choice == "5":
                self._mark_found_menu()
            elif choice == "6":
                self._export_menu()
            elif choice == "0":
                print("\n👋 Goodbye!")
                break
            else:
                print("\n❌ Invalid option. Please try again.")
    
    def _upload_menu(self):
        """Handle item upload."""
        print("\n📸 UPLOAD LOST ITEM")
        print("-" * 50)
        
        image_path = input("Enter image path: ").strip()
        if not image_path:
            print("❌ Cancelled")
            return
        
        name = input("Item name/category: ").strip()
        if not name:
            print("❌ Cancelled")
            return
        
        description = input("Description (optional): ").strip()
        
        self.upload_item(image_path, name, description)
    
    def _match_menu(self):
        """Handle video matching."""
        print("\n🎬 MATCH AGAINST VIDEO")
        print("-" * 50)
        
        video_path = input("Enter video path: ").strip()
        if not video_path:
            print("❌ Cancelled")
            return
        
        confidence = 0.6
        try:
            conf_input = input("Confidence threshold (0.0-1.0, default 0.6): ").strip()
            if conf_input:
                confidence = float(conf_input)
                if not 0.0 <= confidence <= 1.0:
                    print("❌ Invalid confidence. Using default 0.6")
                    confidence = 0.6
        except ValueError:
            print("⚠️  Invalid input. Using default 0.6")
        
        self.match_video(video_path, confidence)
    
    def _mark_found_menu(self):
        """Handle marking item as found."""
        items = self.service.get_lost_items()
        if not items:
            print("\n📭 No lost items to mark as found")
            return
        
        print("\n✅ MARK ITEM AS FOUND")
        print("-" * 50)
        
        for i, item in enumerate(items, 1):
            print(f"{i}. {item['item_id']}: {item['name']}")
        
        try:
            choice = int(input("\nSelect item number (or 0 to cancel): "))
            if 1 <= choice <= len(items):
                item_id = items[choice-1]['item_id']
                if self.service.mark_found(item_id):
                    print(f"\n✅ Marked {item_id} as found!")
                else:
                    print(f"\n❌ Error marking item as found")
            elif choice == 0:
                print("Cancelled")
        except ValueError:
            print("❌ Invalid selection")
    
    def _export_menu(self):
        """Handle results export."""
        print("\n💾 EXPORT RESULTS")
        print("-" * 50)
        
        output_path = input("Enter output file path (default: lost_items_results.json): ").strip()
        if not output_path:
            output_path = "lost_items_results.json"
        
        if self.reporter.export_matches(output_path):
            print(f"\n✅ Results exported to {output_path}")
        else:
            print(f"\n❌ Error exporting results")


def main():
    """Main entry point."""
    parser = ArgumentParser(description="Lost Item Identification System")
    parser.add_argument("--upload", type=str, help="Upload lost item image")
    parser.add_argument("--name", type=str, help="Lost item name")
    parser.add_argument("--description", type=str, default="", help="Lost item description")
    parser.add_argument("--match", type=str, help="Video file to match against")
    parser.add_argument("--list", action="store_true", help="List registered items")
    parser.add_argument("--report", action="store_true", help="Show match report")
    parser.add_argument("--export", type=str, help="Export results to file")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Initialize matcher
    matcher = LostItemMatcher()
    
    # If no arguments, use interactive mode
    if not any(vars(args).values()):
        args.interactive = True
    
    # Handle command-line arguments
    if args.upload and args.name:
        matcher.upload_item(args.upload, args.name, args.description)
    
    if args.match:
        matcher.match_video(args.match)
    
    if args.list:
        matcher.list_items()
    
    if args.report:
        matcher.show_report()
    
    if args.export:
        matcher.reporter.export_matches(args.export)
    
    # Interactive mode
    if args.interactive:
        matcher.interactive_menu()


if __name__ == "__main__":
    main()
