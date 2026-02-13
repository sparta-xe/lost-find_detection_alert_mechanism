# 🔍 Lost Item Detection System

An advanced AI-powered system for detecting and tracking lost items in video surveillance footage using computer vision and machine learning.

## ✨ Features

- **Real-time Object Detection**: Uses YOLOv8 for accurate object detection
- **Lost Item Matching**: Advanced screenshot-to-video matching with multiple feature extraction methods
- **Person-Object Interaction Detection**: Detects pickup attempts and suspicious behavior
- **Interactive Web Interface**: Streamlit-based UI for easy operation
- **Enhanced Small Object Detection**: Optimized for detecting small items like phones, keys, etc.
- **Multi-scale Template Matching**: Robust matching across different lighting and compression conditions

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Interactive Web App

```bash
streamlit run streamlit_app.py
```

### Command Line Usage

```bash
# Basic detection
python scripts/enhanced_tracking_v2.py --video path/to/video.mp4

# With lost item tracking
python scripts/enhanced_tracking_v2.py --video path/to/video.mp4 --lost-item path/to/lost_item.jpg --name "My Lost Phone"
```

## 📁 Project Structure

```
├── src/
│   ├── detection/          # Object detection modules
│   │   ├── yolo_detector.py       # Core YOLO detection
│   │   ├── enhanced_object_detector.py  # Advanced tracking
│   │   └── object_filter.py       # Object filtering
│   ├── reidentification/   # Lost item matching
│   │   ├── improved_matcher.py    # Enhanced matching system
│   │   └── lost_item_matcher.py   # Basic matcher
│   ├── tracking/           # Object tracking
│   ├── escalation/         # Alert and notification system
│   └── ingestion/          # Video input handling
├── scripts/                # Utility scripts
│   ├── enhanced_tracking_v2.py    # Main pipeline
│   └── setup_streamlit_demo.py    # Demo setup
├── data/                   # Data storage
│   ├── lost_items/        # Uploaded lost item images
│   └── test_clips/         # Sample videos
└── streamlit_app.py        # Web interface
```

## 🎯 Core Components

### 1. Object Detection (`src/detection/`)
- **YOLODetector**: Core object detection using YOLOv8
- **EnhancedObjectDetector**: Advanced tracking with state management
- **ObjectFilter**: Filters relevant objects for tracking

### 2. Lost Item Matching (`src/reidentification/`)
- **ImprovedLostItemMatcher**: Multi-feature matching system optimized for screenshots
- Uses color histograms, texture analysis, shape matching, and template matching
- Specifically designed for screenshot-to-video matching scenarios

### 3. Interactive Interface (`streamlit_app.py`)
- Real-time video processing with overlay
- Lost item upload and management
- Live statistics and alerts
- Configurable detection parameters

## 🔧 Configuration

### Detection Settings
- **Detection Confidence**: 0.10-0.25 (lower for small objects)
- **Match Threshold**: 0.20-0.30 (lower for screenshot matching)
- **Stationary Threshold**: 3.0 seconds
- **Proximity Threshold**: 100 pixels

### Screenshot Matching
For best results with screenshot-to-video matching:
- Set match threshold to **0.20-0.25**
- Use clear, well-lit screenshots
- Crop images to focus on the object
- See `SCREENSHOT_MATCHING_GUIDE.md` for detailed troubleshooting

## 📊 Performance Features

- **Multi-scale Detection**: Processes images at multiple resolutions for small objects
- **Enhanced Preprocessing**: CLAHE contrast enhancement for better detection
- **Duplicate Removal**: Intelligent filtering of overlapping detections
- **Real-time Processing**: Optimized for live video streams

## 🧪 Testing and Debugging

### Test Screenshot Matching
```bash
python debug_screenshot_matching.py path/to/lost_item.jpg path/to/detection.jpg
```

### Test Small Object Detection
```bash
python scripts/test_small_object_detection.py
```

### Test Pickup Detection
```bash
python scripts/test_pickup_detection.py
```

## 📚 Documentation

- `SCREENSHOT_MATCHING_GUIDE.md` - Troubleshooting screenshot matching issues
- `DETECTION_IMPROVEMENT_GUIDE.md` - Improving detection accuracy
- `STREAMLIT_GUIDE.md` - Using the web interface
- `README_STREAMLIT_INTERACTIVE.md` - Interactive features guide

## 🎮 Usage Examples

### Web Interface
1. Start the Streamlit app: `streamlit run streamlit_app.py`
2. Upload a lost item image in the sidebar
3. Select video source (upload, sample, or webcam)
4. Adjust detection settings
5. Click "Start Processing"

### Command Line
```bash
# Process video with lost item
python scripts/enhanced_tracking_v2.py \
    --video data/test_clips/sample.mp4 \
    --lost-item data/lost_items/phone.jpg \
    --name "iPhone 12" \
    --description "Black iPhone with blue case" \
    --show-video \
    --export results.json
```

## 🚨 Alert System

The system generates alerts for:
- **Lost Item Found**: When a registered lost item is detected
- **Pickup Attempt**: When someone approaches a stationary object
- **Item Picked Up**: When an object is taken by a person
- **Suspicious Behavior**: Based on person-object interaction patterns

## 🔍 Advanced Features

### Person-Object Interaction Detection
- Tracks people and objects simultaneously
- Detects when people approach objects
- Identifies pickup attempts and theft scenarios
- Maintains interaction history

### Object State Management
- **Stationary**: Objects that haven't moved for a threshold time
- **Dropped**: Objects that were previously carried
- **Picked Up**: Objects taken by people
- **Moving**: Objects in motion

### Multi-Feature Matching
The improved matcher uses:
- **Template Matching** (50% weight): Direct image comparison
- **Color Similarity** (25% weight): Color histogram matching
- **Keypoint Matching** (10% weight): SIFT/ORB feature points
- **Shape Analysis** (8% weight): Hu moments and contours
- **Texture Analysis** (4% weight): Local Binary Patterns
- **Edge Detection** (3% weight): Canny edge patterns

## 🛠️ Development

### Adding New Object Types
Edit `src/detection/object_filter.py`:
```python
TRACKABLE_OBJECTS = {
    "backpack",
    "handbag", 
    "suitcase",
    "laptop",
    "cell phone",
    "your_new_object"  # Add here
}
```

### Customizing Matching Algorithm
Modify weights in `src/reidentification/improved_matcher.py`:
```python
weights = {
    'template_match': 0.5,      # Adjust these weights
    'color_similarity': 0.25,   # based on your needs
    'keypoint_similarity': 0.1,
    # ...
}
```

## 📈 Performance Tips

1. **For Small Objects**: Lower detection confidence to 0.10-0.15
2. **For Screenshot Matching**: Use threshold 0.20-0.25
3. **For Real-time Processing**: Enable frame skipping
4. **For High Accuracy**: Increase detection confidence to 0.30+

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics for object detection
- OpenCV for computer vision operations
- Streamlit for the web interface
- NumPy and scikit-learn for numerical operations

## 📞 Support

For issues and questions:
1. Check the troubleshooting guides in the docs/ folder
2. Run the debug scripts for specific issues
3. Open an issue on GitHub with detailed information

---

**Built with ❤️ for finding lost items using AI**