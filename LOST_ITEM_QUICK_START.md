# Lost Item Re-Identification Feature - Quick Start Guide

## 🚀 Installation & Setup

### Step 1: Dependencies Already Installed ✅

The required packages are already in your environment:
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Python 3.13+

No additional installations needed!

### Step 2: Verify Installation

```bash
# Test that modules can be imported
python -c "from src.reidentification.lost_item_matcher import LostItemMatcher; print('✅ Import successful!')"
```

---

## 📸 Quick Start (5 minutes)

### Option 1: Interactive Demo (Easiest)

```bash
# Run the demo with sample images
python scripts/demo_lost_item.py
```

This will:
1. Create sample test images
2. Upload them to the system
3. Attempt to match against available video
4. Generate a report

### Option 2: With Your Own Image

```bash
# Run interactive tool
python scripts/lost_item_upload.py --interactive
```

Follow the prompts:
1. Select "Upload Lost Item Image"
2. Provide image path
3. Enter item name
4. Select "Match Against Video"
5. View results

### Option 3: Command-Line (Fastest)

```bash
# Upload item
python scripts/lost_item_upload.py \
  --upload "path/to/your/item.jpg" \
  --name "My Lost Item" \
  --description "Optional description"

# View registered items
python scripts/lost_item_upload.py --list

# View matches
python scripts/lost_item_upload.py --report

# Export results
python scripts/lost_item_upload.py --export "results.json"
```

---

## 🎥 Full Tracking with Lost Item Matching

### Basic Usage

```bash
python scripts/enhanced_tracking.py \
  --video "data/test_clips/cam1.mp4" \
  --lost-item "path/to/item.jpg" \
  --name "My Backpack"
```

### With Options

```bash
python scripts/enhanced_tracking.py \
  --video "data/test_clips/cam1.mp4" \
  --lost-item "path/to/item.jpg" \
  --name "My Backpack" \
  --confidence 0.65 \
  --max-frames 500 \
  --export "findings.json"
```

**Available Options:**
- `--video`: Path to video file (default: data/test_clips/cam1.mp4)
- `--camera`: Camera identifier (default: cam_1)
- `--lost-item`: Image path of lost item
- `--name`: Name of the item
- `--description`: Description (optional)
- `--confidence`: Match threshold 0.0-1.0 (default: 0.6)
- `--max-frames`: Limit frames to process (optional)
- `--quiet`: Suppress output
- `--export`: Save results to JSON

---

## 🐍 Python API Usage

### Simple Example

```python
from src.escalation.lost_item_service import LostItemService, LostItemReporter

# Initialize
service = LostItemService(match_threshold=0.6)

# Upload a lost item
success, item_id = service.upload_lost_item(
    image_path="path/to/lost_item.jpg",
    name="Lost Phone",
    description="Black iPhone with red case"
)

if success:
    print(f"✅ Item uploaded: {item_id}")

# Get registered items
items = service.get_lost_items()
print(f"Registered items: {len(items)}")

# Get matches
matches = service.get_matches()
print(f"Matches found: {len(matches)}")

# Generate report
reporter = LostItemReporter(service)
print(reporter.report_matches())

# Export results
reporter.export_matches("results.json")
```

### With Enhanced Tracking

```python
from scripts.enhanced_tracking import EnhancedTrackingPipeline

# Create pipeline
pipeline = EnhancedTrackingPipeline(
    video_source="camera_feed.mp4",
    lost_item_threshold=0.6
)

# Add lost item
pipeline.add_lost_item(
    "item_photo.jpg",
    "Lost Backpack",
    "Red with leather straps"
)

# Run tracking with lost item matching
pipeline.run(max_frames=None, verbose=True)

# Get statistics
stats = pipeline.get_statistics()
print(f"Lost items matched: {stats['lost_item_matches']}")

# Print final report
pipeline.print_statistics()
```

---

## 📊 Understanding Results

### Match Confidence Scores

- **0.9-1.0**: Excellent match (very likely the same item)
- **0.8-0.9**: Strong match (probably the same item)
- **0.7-0.8**: Good match (reasonably confident)
- **0.6-0.7**: Possible match (consider reviewing)
- **<0.6**: Not a match (filtered out by default)

### Adjusting Sensitivity

```bash
# More sensitive (find more, more false positives)
--confidence 0.5

# More strict (find less, fewer false positives)
--confidence 0.75

# Very strict (only high-confidence matches)
--confidence 0.85
```

---

## 📁 Directory Structure

```
Lost_and_found_Temp-main/
├── data/
│   ├── lost_items/          ← Uploaded item images
│   └── test_clips/          ← Video files to search
├── src/
│   ├── reidentification/
│   │   └── lost_item_matcher.py    ← Core matching engine
│   └── escalation/
│       └── lost_item_service.py    ← Service layer
├── scripts/
│   ├── lost_item_upload.py         ← Interactive CLI
│   ├── enhanced_tracking.py        ← Tracking + matching
│   └── demo_lost_item.py           ← Demo script
├── tests/
│   └── test_lost_item_matching.py  ← Unit tests
└── docs/
    ├── LOST_ITEM_IDENTIFICATION.md          ← Full documentation
    └── LOST_ITEM_FEATURE_SUMMARY.md         ← This feature summary
```

---

## 🧪 Testing

### Run Unit Tests

```bash
# All tests
pytest tests/test_lost_item_matching.py -v

# Specific test class
pytest tests/test_lost_item_matching.py::TestLostItemMatcher -v

# With coverage report
pytest tests/test_lost_item_matching.py --cov=src
```

### Quick Test Without Resources

```bash
# Just test imports
python -c "
from src.reidentification.lost_item_matcher import LostItemMatcher
from src.escalation.lost_item_service import LostItemService
print('✅ All modules imported successfully!')
"
```

---

## 🔧 Configuration

Edit `configs/test_config.yaml` to customize:

```yaml
lost_item_matching:
  color_weight: 0.4        # Color importance (0-1)
  edge_weight: 0.3         # Shape importance (0-1)
  texture_weight: 0.3      # Texture importance (0-1)
  threshold: 0.6           # Minimum confidence (0-1)
  upload_dir: data/lost_items
```

---

## 📋 Workflow Examples

### Scenario 1: Lost Item in Shopping Mall

```bash
# Step 1: Upload photo of lost item
python scripts/lost_item_upload.py \
  --upload "backpack.jpg" \
  --name "Red Backpack" \
  --description "Medium size, leather straps"

# Step 2: Search security camera footage
python scripts/enhanced_tracking.py \
  --video "mall_security_cam.mp4" \
  --lost-item "backpack.jpg" \
  --name "Red Backpack" \
  --export "mall_findings.json"

# Step 3: Review findings
cat mall_findings.json
```

### Scenario 2: Multiple Lost Items

```bash
# Initialize service
python scripts/lost_item_upload.py --interactive

# Upload multiple items (use menu option 1)
# Then search video (option 2)
# View all matches (option 4)
```

### Scenario 3: Continuous Monitoring

```bash
# Create a script to monitor multiple videos
for video in cafe_*.mp4; do
  python scripts/enhanced_tracking.py \
    --video "$video" \
    --lost-item "missing_item.jpg" \
    --name "Missing Item" \
    --export "results_${video}.json"
done
```

---

## ⚡ Performance Tips

### For Large Videos

```bash
# Process in chunks to save memory
--max-frames 500

# Then process next chunk
# (system remembers previously found items)
```

### For Faster Processing

```bash
# Lower resolution matching
--confidence 0.65  # Faster but less precise

# Limit detections per frame
# (handled automatically)
```

### For Better Results

```bash
# Higher confidence for strict matching
--confidence 0.75

# Use multiple angles in upload image
# Ensure good lighting in test images
```

---

## 🐛 Troubleshooting

### "No matches found"
1. Lower confidence threshold: `--confidence 0.55`
2. Check image quality (clear, well-lit)
3. Ensure item is visible in video

### "Too many false positives"
1. Raise confidence threshold: `--confidence 0.75`
2. Use more specific upload image
3. Check video relevance

### "Video not found"
1. Check file path is correct
2. Ensure video format is MP4/MOV/AVI
3. Verify file is readable

### "Out of memory"
1. Use `--max-frames` to process in chunks
2. Reduce video resolution
3. Clear match history in service

---

## 📚 Full Documentation

For detailed information, see:
- **Full Guide**: `docs/LOST_ITEM_IDENTIFICATION.md`
- **Feature Summary**: `docs/LOST_ITEM_FEATURE_SUMMARY.md`
- **API Reference**: `docs/LOST_ITEM_IDENTIFICATION.md#api-reference`

---

## 🎯 Common Commands

```bash
# Interactive mode (recommended for beginners)
python scripts/lost_item_upload.py --interactive

# Quick demo with sample images
python scripts/demo_lost_item.py

# Search video with your image
python scripts/enhanced_tracking.py --video "video.mp4" --lost-item "item.jpg" --name "Item Name"

# View results
python scripts/lost_item_upload.py --report

# Export findings
python scripts/lost_item_upload.py --export "results.json"

# Mark item as found
python scripts/lost_item_upload.py --interactive  # Then select option 5

# Run tests
pytest tests/test_lost_item_matching.py -v
```

---

## ✨ Features Summary

✅ **Easy Upload**: Just provide an image of your lost item
✅ **Smart Matching**: Multi-modal features for robust matching
✅ **Low-Res Ready**: Works with compressed and low-quality images
✅ **Real-Time**: Process at video frame rates
✅ **Detailed Reports**: Know exactly where and when items are found
✅ **Export Results**: JSON export for integration with other systems
✅ **Production Ready**: Error handling, logging, and testing included

---

## 🚀 Next Steps

1. **Try the demo**: `python scripts/demo_lost_item.py`
2. **Read docs**: `docs/LOST_ITEM_IDENTIFICATION.md`
3. **Test with your own image**: Use `--interactive` mode
4. **Integrate with your system**: Use the Python API
5. **Process videos**: Run against camera feeds

---

**Questions?** See the full documentation or check the troubleshooting section!
