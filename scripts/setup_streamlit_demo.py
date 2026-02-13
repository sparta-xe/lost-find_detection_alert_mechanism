#!/usr/bin/env python
"""
Setup script for Streamlit demo

Creates sample lost item images and ensures proper directory structure
for the Streamlit application demo.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def create_sample_lost_item_images():
    """Create sample lost item images for demo purposes."""
    
    # Ensure directories exist
    lost_items_dir = PROJECT_ROOT / "data" / "lost_items"
    lost_items_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample items to create
    sample_items = [
        {
            "name": "red_backpack.jpg",
            "color": (200, 50, 50),  # Red
            "shape": "backpack",
            "size": (150, 200)
        },
        {
            "name": "blue_phone.jpg", 
            "color": (50, 50, 200),  # Blue
            "shape": "phone",
            "size": (80, 160)
        },
        {
            "name": "black_wallet.jpg",
            "color": (50, 50, 50),  # Black
            "shape": "wallet", 
            "size": (120, 80)
        },
        {
            "name": "green_bottle.jpg",
            "color": (50, 200, 50),  # Green
            "shape": "bottle",
            "size": (60, 180)
        }
    ]
    
    created_items = []
    
    for item in sample_items:
        item_path = lost_items_dir / item["name"]
        
        # Skip if already exists
        if item_path.exists():
            print(f"✅ Sample item already exists: {item['name']}")
            created_items.append(str(item_path))
            continue
        
        # Create image
        img = Image.new('RGB', (300, 300), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Calculate position to center the item
        item_width, item_height = item["size"]
        x = (300 - item_width) // 2
        y = (300 - item_height) // 2
        
        # Draw different shapes based on item type
        if item["shape"] == "backpack":
            # Draw backpack shape
            draw.rectangle([x, y, x + item_width, y + item_height], 
                          fill=item["color"], outline=(0, 0, 0), width=3)
            # Add straps
            draw.rectangle([x + 20, y - 10, x + 40, y + 20], 
                          fill=(100, 50, 0), outline=(0, 0, 0), width=2)
            draw.rectangle([x + item_width - 40, y - 10, x + item_width - 20, y + 20], 
                          fill=(100, 50, 0), outline=(0, 0, 0), width=2)
            
        elif item["shape"] == "phone":
            # Draw phone shape
            draw.rounded_rectangle([x, y, x + item_width, y + item_height], 
                                 radius=10, fill=item["color"], outline=(0, 0, 0), width=3)
            # Add screen
            draw.rounded_rectangle([x + 10, y + 20, x + item_width - 10, y + item_height - 30], 
                                 radius=5, fill=(20, 20, 20))
            
        elif item["shape"] == "wallet":
            # Draw wallet shape
            draw.rectangle([x, y, x + item_width, y + item_height], 
                          fill=item["color"], outline=(0, 0, 0), width=3)
            # Add fold line
            draw.line([x, y + item_height//2, x + item_width, y + item_height//2], 
                     fill=(0, 0, 0), width=2)
            
        elif item["shape"] == "bottle":
            # Draw bottle shape
            # Body
            draw.rectangle([x + 15, y + 30, x + item_width - 15, y + item_height], 
                          fill=item["color"], outline=(0, 0, 0), width=3)
            # Neck
            draw.rectangle([x + 25, y, x + item_width - 25, y + 40], 
                          fill=item["color"], outline=(0, 0, 0), width=3)
            # Cap
            draw.rectangle([x + 20, y - 10, x + item_width - 20, y + 10], 
                          fill=(100, 100, 100), outline=(0, 0, 0), width=2)
        
        # Add some texture/noise for realism
        pixels = np.array(img)
        noise = np.random.randint(-20, 20, pixels.shape, dtype=np.int16)
        pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(pixels)
        
        # Save image
        img.save(item_path, quality=85)
        created_items.append(str(item_path))
        print(f"✅ Created sample item: {item['name']}")
    
    return created_items


def create_sample_video():
    """Create a simple sample video with moving objects."""
    
    test_clips_dir = PROJECT_ROOT / "data" / "test_clips"
    test_clips_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = test_clips_dir / "streamlit_demo.mp4"
    
    # Skip if already exists
    if video_path.exists():
        print(f"✅ Sample video already exists: {video_path.name}")
        return str(video_path)
    
    # Video parameters
    width, height = 640, 480
    fps = 30
    duration = 10  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    print(f"🎬 Creating sample video: {video_path.name}")
    
    for frame_num in range(total_frames):
        # Create frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Add moving objects that could match our sample items
        
        # Moving red rectangle (backpack-like)
        x = int(50 + (frame_num * 2) % (width - 100))
        y = 100
        cv2.rectangle(frame, (x, y), (x + 80, y + 120), (50, 50, 200), -1)  # Red in BGR
        cv2.rectangle(frame, (x, y), (x + 80, y + 120), (0, 0, 0), 2)
        
        # Moving blue rectangle (phone-like)
        x2 = int(width - 100 - (frame_num * 1.5) % (width - 100))
        y2 = 200
        cv2.rectangle(frame, (x2, y2), (x2 + 40, y2 + 80), (200, 50, 50), -1)  # Blue in BGR
        cv2.rectangle(frame, (x2, y2), (x2 + 40, y2 + 80), (0, 0, 0), 2)
        
        # Stationary black rectangle (wallet-like)
        x3, y3 = 300, 350
        cv2.rectangle(frame, (x3, y3), (x3 + 60, y3 + 40), (50, 50, 50), -1)  # Black
        cv2.rectangle(frame, (x3, y3), (x3 + 60, y3 + 40), (0, 0, 0), 2)
        
        # Moving green circle (bottle-like)
        x4 = int(100 + (frame_num * 3) % (width - 200))
        y4 = 300
        cv2.circle(frame, (x4, y4), 25, (50, 200, 50), -1)  # Green in BGR
        cv2.circle(frame, (x4, y4), 25, (0, 0, 0), 2)
        
        # Add frame number for reference
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Created sample video: {video_path.name} ({duration}s, {total_frames} frames)")
    
    return str(video_path)


def setup_demo_environment():
    """Set up complete demo environment."""
    print("🚀 Setting up Streamlit demo environment...")
    
    # Create sample lost item images
    print("\n📦 Creating sample lost item images...")
    created_items = create_sample_lost_item_images()
    
    # Create sample video
    print("\n🎬 Creating sample video...")
    video_path = create_sample_video()
    
    # Print summary
    print(f"\n✅ Demo setup complete!")
    print(f"\n📁 Created files:")
    print(f"   📦 Lost items ({len(created_items)}):")
    for item in created_items:
        print(f"      • {Path(item).name}")
    print(f"   🎬 Sample video: {Path(video_path).name}")
    
    print(f"\n🚀 Ready to run Streamlit app!")
    print(f"   Run: python run_streamlit_app.py")
    print(f"   Or:  streamlit run streamlit_app.py")
    
    return {
        'lost_items': created_items,
        'sample_video': video_path
    }


def main():
    """Main entry point."""
    try:
        setup_demo_environment()
        return 0
    except Exception as e:
        print(f"❌ Error setting up demo: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())