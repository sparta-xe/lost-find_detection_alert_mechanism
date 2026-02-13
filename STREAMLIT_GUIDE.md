# 🔍 Interactive Lost Item Detection with Streamlit

This guide shows you how to use the interactive Streamlit web application for real-time lost item detection and tracking.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Launch the Application

```bash
# Option 1: Using the launcher script (recommended)
python run_streamlit_app.py

# Option 2: Direct Streamlit command
streamlit run streamlit_app.py
```

### 3. Open Your Browser

The app will automatically open at: **http://localhost:8501**

---

## 🎯 Features

### 📹 Video Processing
- **Upload Videos**: Drag and drop your own video files
- **Sample Videos**: Use pre-loaded test clips
- **Live Webcam**: Real-time processing from your camera
- **Interactive Controls**: Start, stop, and configure processing

### 📦 Lost Item Management
- **Upload Images**: Add photos of lost items
- **Item Details**: Name and describe your items
- **Real-time Matching**: Automatic detection in video streams
- **Match Confidence**: Adjustable sensitivity settings

### 📊 Live Statistics
- **Frame Processing**: Real-time frame count and detection stats
- **Match History**: Timeline of all found items
- **Confidence Scores**: Visual confidence indicators
- **Export Results**: Download findings as JSON

### 🎨 Visual Overlays
- **Detection Boxes**: Green boxes around detected objects
- **Match Highlights**: Red boxes for lost item matches
- **Confidence Labels**: Real-time confidence scores
- **Status Indicators**: Processing status and alerts

---

## 📱 User Interface Guide

### Sidebar Controls

#### 📹 Video Source
- **Upload Video**: Select MP4, AVI, MOV, or MKV files
- **Use Sample Video**: Choose from pre-loaded test clips
- **Webcam (Live)**: Use your computer's camera

#### ⚙️ Detection Settings
- **Detection Confidence**: How confident the system should be (0.1-1.0)
- **Lost Item Match Threshold**: Sensitivity for lost item matching (0.1-1.0)
- **Max Frames**: Limit processing (0 = unlimited)

#### 📦 Lost Items
- **Upload Image**: Add photos of items you're looking for
- **Item Name**: Give your item a recognizable name
- **Description**: Optional details about the item
- **Registered Items**: View all uploaded lost items

### Main Content Area

#### 📺 Video Processing
- **Live Video Feed**: Real-time video with detection overlays
- **Control Buttons**: Start, stop, and clear processing
- **Status Messages**: Success/error notifications

#### 📊 Live Statistics
- **Metrics**: Frames processed, detections, matches
- **Recent Matches**: Last 10 found items
- **Item Gallery**: Visual list of registered items

### Results Tab

#### 📋 Reports & Export
- **Generate Report**: Detailed text summary
- **Export Results**: Download JSON with all data
- **Match History**: Complete timeline of findings

---

## 🎮 How to Use

### Step 1: Add Lost Items

1. **Upload Image**: Click "Upload lost item image" in sidebar
2. **Enter Details**: Provide item name and optional description
3. **Add Item**: Click "Add Lost Item" button
4. **Verify**: Check that item appears in "Registered Items"

### Step 2: Select Video Source

**For Uploaded Videos:**
1. Select "Upload Video" from dropdown
2. Drag and drop your video file
3. Wait for upload to complete

**For Sample Videos:**
1. Select "Use Sample Video"
2. Choose from available test clips

**For Live Camera:**
1. Select "Webcam (Live)"
2. Allow camera permissions if prompted

### Step 3: Configure Settings

1. **Detection Confidence**: Start with 0.25 (default)
   - Lower = more detections (may include false positives)
   - Higher = fewer detections (may miss some objects)

2. **Match Threshold**: Start with 0.6 (default)
   - Lower = more sensitive matching (more false positives)
   - Higher = stricter matching (may miss matches)

3. **Max Frames**: Leave at 0 for unlimited processing

### Step 4: Start Processing

1. Click **"▶️ Start Processing"**
2. Watch the live video feed with overlays
3. Monitor statistics in the right panel
4. Look for red-highlighted matches

### Step 5: Review Results

1. **Live Alerts**: Green success messages for matches
2. **Statistics Panel**: Real-time metrics and recent matches
3. **Results Tab**: Complete history and export options

---

## 🎨 Visual Indicators

### Detection Overlays

- **Green Boxes**: Regular object detections
- **Red Boxes**: Lost item matches (highlighted)
- **Labels**: Object type and confidence score
- **Match Labels**: "FOUND: [Item Name] (XX%)"

### Status Messages

- **🎯 LOST ITEM FOUND!**: Success alert for matches
- **✅ Processing completed**: Finished processing
- **❌ Processing error**: Error occurred
- **⚠️ Warning messages**: Configuration issues

### Confidence Colors

- **90-100%**: Excellent match (very confident)
- **80-89%**: Strong match (confident)
- **70-79%**: Good match (reasonably confident)
- **60-69%**: Possible match (review recommended)

---

## ⚙️ Configuration Tips

### For Better Detection

```
Detection Confidence: 0.15-0.3
- Lower values catch more objects
- Good for crowded scenes
- May increase false positives
```

### For Accurate Matching

```
Match Threshold: 0.7-0.8
- Higher values reduce false matches
- Better for similar-looking items
- May miss some valid matches
```

### For Fast Processing

```
Max Frames: 100-500
- Process video in chunks
- Good for long videos
- Restart processing to continue
```

### For Live Camera

```
Detection Confidence: 0.3-0.4
Match Threshold: 0.6-0.7
- Balanced settings for real-time
- Good responsiveness
- Reasonable accuracy
```

---

## 📊 Understanding Results

### Statistics Explained

- **Frames Processed**: Total video frames analyzed
- **Total Detections**: All objects found (people, bags, etc.)
- **Lost Item Matches**: Confirmed matches with uploaded items
- **Registered Items**: Number of lost items uploaded

### Match Confidence

The system uses multiple features to match items:
- **Color similarity**: Dominant colors and patterns
- **Shape analysis**: Object edges and contours  
- **Texture features**: Surface patterns and details

### Export Data Format

```json
{
  "statistics": {
    "frames_processed": 150,
    "detections": 45,
    "matches": 3,
    "lost_items": 2
  },
  "matches": [
    {
      "timestamp": 1640995200.0,
      "item_name": "Red Backpack",
      "confidence": "85.2%",
      "camera": "streamlit_cam"
    }
  ],
  "timestamp": "2024-01-01T12:00:00"
}
```

---

## 🐛 Troubleshooting

### Video Issues

**"Video not loading"**
- Check file format (MP4, AVI, MOV, MKV)
- Ensure file isn't corrupted
- Try a smaller file size

**"Webcam not working"**
- Allow camera permissions in browser
- Close other apps using camera
- Try refreshing the page

### Detection Issues

**"No objects detected"**
- Lower detection confidence to 0.15
- Check video quality and lighting
- Ensure objects are clearly visible

**"Too many false detections"**
- Raise detection confidence to 0.4+
- Check for camera shake or blur
- Use better quality video

### Matching Issues

**"No matches found"**
- Lower match threshold to 0.5
- Check uploaded image quality
- Ensure item is visible in video
- Try different angles/lighting

**"Too many false matches"**
- Raise match threshold to 0.75+
- Use clearer reference images
- Add more specific descriptions

### Performance Issues

**"App running slowly"**
- Set max frames limit (e.g., 200)
- Use smaller video files
- Close other browser tabs
- Restart the application

**"Browser freezing"**
- Reduce video resolution
- Lower detection confidence
- Process in smaller chunks

---

## 🔧 Advanced Usage

### Batch Processing

1. Upload multiple lost items first
2. Process video with all items
3. Export results for analysis
4. Clear and repeat with new video

### Custom Workflows

**Security Camera Analysis:**
```
1. Upload photos of missing items
2. Load security footage
3. Set high match threshold (0.8)
4. Process and export findings
```

**Event Monitoring:**
```
1. Use live webcam mode
2. Add items before event starts
3. Monitor in real-time
4. Get instant alerts for matches
```

### Integration Tips

- Export JSON data for external systems
- Use consistent naming for items
- Document match timestamps
- Save reference images separately

---

## 📚 API Integration

The Streamlit app uses the same core modules as the command-line tools:

```python
# Access the underlying service
from src.escalation.lost_item_service import LostItemService

# Use in your own code
service = LostItemService()
success, item_id = service.upload_lost_item("image.jpg", "My Item")
matches = service.get_matches()
```

---

## 🎯 Best Practices

### Image Upload Tips

1. **Clear Photos**: Well-lit, focused images work best
2. **Multiple Angles**: Upload different views if possible
3. **Distinctive Features**: Highlight unique characteristics
4. **Good Resolution**: At least 200x200 pixels recommended

### Processing Tips

1. **Start Conservative**: Begin with default settings
2. **Adjust Gradually**: Make small threshold changes
3. **Monitor Results**: Watch for false positives/negatives
4. **Save Settings**: Note what works for your use case

### Performance Tips

1. **Chunk Large Videos**: Use max frames for long videos
2. **Close Unused Tabs**: Free up browser memory
3. **Use Local Files**: Avoid network-stored videos
4. **Regular Exports**: Save results periodically

---

## 🚀 Next Steps

1. **Try the Demo**: Start with sample videos and test images
2. **Upload Your Items**: Add real lost item photos
3. **Test Different Settings**: Find optimal thresholds
4. **Process Your Videos**: Analyze your own footage
5. **Export Results**: Save findings for records

---

## 📞 Support

- **Documentation**: See `docs/` folder for detailed guides
- **Command Line**: Use `scripts/` for batch processing
- **API Reference**: Check `src/` modules for integration
- **Troubleshooting**: Review error messages and logs

---

**Ready to find your lost items? Launch the app and start detecting!** 🔍✨