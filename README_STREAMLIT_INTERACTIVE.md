# 🔍 Interactive Lost Item Detection System

An interactive web-based application for real-time lost item detection and tracking using Streamlit.

## 🚀 Quick Start (30 seconds)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the interactive app (includes demo setup)
python quick_start_streamlit.py
```

**That's it!** Your browser will open to `http://localhost:8501` with the interactive interface.

---

## 🎯 What You Get

### 🖥️ Interactive Web Interface
- **Drag & Drop**: Upload videos and lost item images
- **Real-time Processing**: Live video analysis with overlays
- **Visual Feedback**: Instant alerts when items are found
- **Statistics Dashboard**: Live metrics and match history

### 📹 Video Input Options
- **Upload Videos**: MP4, AVI, MOV, MKV files
- **Sample Videos**: Pre-loaded test clips
- **Live Webcam**: Real-time camera feed processing

### 📦 Lost Item Management
- **Image Upload**: Add photos of lost items
- **Item Details**: Name and describe items
- **Match Confidence**: Adjustable sensitivity
- **Visual Gallery**: See all registered items

### 🎨 Visual Overlays
- **Green Boxes**: Regular object detections
- **Red Highlights**: Lost item matches
- **Confidence Scores**: Real-time match percentages
- **Status Alerts**: Success/warning notifications

---

## 📱 User Interface

### Main Features

1. **Video Processing Area**
   - Live video feed with detection overlays
   - Start/Stop/Clear controls
   - Real-time status messages

2. **Sidebar Controls**
   - Video source selection
   - Detection sensitivity settings
   - Lost item upload and management

3. **Statistics Panel**
   - Live processing metrics
   - Recent match history
   - Registered items gallery

4. **Results & Export**
   - Detailed match reports
   - JSON export functionality
   - Complete match timeline

---

## 🎮 How to Use

### Step 1: Launch the App

```bash
# Quick start with demo data
python quick_start_streamlit.py

# Or launch manually
python run_streamlit_app.py
```

### Step 2: Add Lost Items

1. **Upload Image**: Use the sidebar file uploader
2. **Enter Details**: Provide item name and description
3. **Add Item**: Click "Add Lost Item"
4. **Verify**: Check item appears in registered list

### Step 3: Select Video Source

**Option A: Upload Your Video**
- Select "Upload Video" from dropdown
- Drag and drop your video file
- Wait for upload completion

**Option B: Use Sample Video**
- Select "Use Sample Video"
- Choose from available test clips

**Option C: Live Camera**
- Select "Webcam (Live)"
- Allow camera permissions

### Step 4: Configure Settings

- **Detection Confidence**: Start with 0.25
- **Match Threshold**: Start with 0.6
- **Max Frames**: 0 for unlimited

### Step 5: Start Processing

1. Click **"▶️ Start Processing"**
2. Watch live video with overlays
3. Monitor statistics panel
4. Look for red match highlights

### Step 6: Review Results

- **Live Alerts**: Instant match notifications
- **Statistics**: Real-time processing metrics
- **Export**: Download results as JSON
- **Reports**: Generate detailed summaries

---

## ⚙️ Configuration Guide

### Detection Settings

```
Detection Confidence: 0.1 - 1.0
├── 0.1-0.3: High sensitivity (more detections, more false positives)
├── 0.3-0.5: Balanced (recommended for most cases)
└── 0.5-1.0: Low sensitivity (fewer detections, fewer false positives)
```

### Match Threshold

```
Lost Item Match Threshold: 0.1 - 1.0
├── 0.4-0.6: High sensitivity (more matches, more false matches)
├── 0.6-0.8: Balanced (recommended for most cases)
└── 0.8-1.0: High precision (fewer matches, fewer false matches)
```

### Recommended Settings

**For Security Cameras:**
- Detection Confidence: 0.3
- Match Threshold: 0.7

**For Mobile Videos:**
- Detection Confidence: 0.25
- Match Threshold: 0.6

**For Live Webcam:**
- Detection Confidence: 0.35
- Match Threshold: 0.65

---

## 🎨 Visual Indicators

### Detection Overlays

| Color | Meaning | Description |
|-------|---------|-------------|
| 🟢 Green | Regular Detection | Normal objects found |
| 🔴 Red | Lost Item Match | Your lost item detected! |
| 📝 Labels | Confidence Score | Match percentage |

### Status Messages

| Icon | Message | Meaning |
|------|---------|---------|
| 🎯 | LOST ITEM FOUND! | Match detected |
| ✅ | Processing completed | Video finished |
| ❌ | Processing error | Something went wrong |
| ⚠️ | Warning | Configuration issue |

### Confidence Levels

| Range | Color | Interpretation |
|-------|-------|----------------|
| 90-100% | 🟢 Excellent | Very confident match |
| 80-89% | 🟡 Strong | Confident match |
| 70-79% | 🟠 Good | Reasonably confident |
| 60-69% | 🔴 Possible | Review recommended |

---

## 📊 Understanding Results

### Statistics Explained

- **Frames Processed**: Total video frames analyzed
- **Total Detections**: All objects found (people, bags, etc.)
- **Lost Item Matches**: Confirmed matches with your items
- **Registered Items**: Number of lost items uploaded

### Match Data

Each match includes:
- **Timestamp**: When the item was found
- **Item Name**: Which lost item was matched
- **Confidence**: How certain the system is
- **Location**: Bounding box coordinates

### Export Format

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
  ]
}
```

---

## 🛠️ Advanced Features

### Batch Processing

1. Upload multiple lost items
2. Process long videos in chunks
3. Export comprehensive results
4. Analyze patterns over time

### Real-time Monitoring

1. Use webcam for live monitoring
2. Set up alerts for specific items
3. Monitor events in real-time
4. Export live session data

### Custom Workflows

**Lost & Found Office:**
```
1. Upload photos of reported items
2. Process security camera footage
3. Generate match reports
4. Export findings for staff
```

**Event Security:**
```
1. Pre-upload common lost items
2. Monitor live camera feeds
3. Get instant match alerts
4. Track items throughout event
```

---

## 🐛 Troubleshooting

### Common Issues

**Video not loading:**
- Check file format (MP4, AVI, MOV, MKV)
- Ensure file size < 200MB
- Try refreshing the page

**No detections:**
- Lower detection confidence to 0.15
- Check video quality and lighting
- Ensure objects are clearly visible

**No matches found:**
- Lower match threshold to 0.5
- Check uploaded image quality
- Ensure item appears in video

**App running slowly:**
- Set max frames limit (e.g., 200)
- Use smaller video files
- Close other browser tabs

### Performance Tips

1. **Use smaller videos** for faster processing
2. **Set frame limits** for long videos
3. **Close unused tabs** to free memory
4. **Use local files** instead of network storage

---

## 📚 File Structure

```
Lost_and_found_Temp-main/
├── streamlit_app.py              ← Main Streamlit application
├── run_streamlit_app.py          ← App launcher script
├── quick_start_streamlit.py      ← Quick start with demo
├── STREAMLIT_GUIDE.md            ← Detailed user guide
├── .streamlit/
│   └── config.toml               ← Streamlit configuration
├── data/
│   ├── lost_items/               ← Uploaded lost item images
│   └── test_clips/               ← Sample videos
└── scripts/
    └── setup_streamlit_demo.py   ← Demo data generator
```

---

## 🔗 Integration

### Command Line Tools

The Streamlit app uses the same core system as the command-line tools:

```bash
# Command line processing
python scripts/enhanced_tracking.py --video video.mp4 --lost-item item.jpg

# Streamlit interface
python run_streamlit_app.py
```

### API Access

```python
# Access the underlying service
from src.escalation.lost_item_service import LostItemService

service = LostItemService()
success, item_id = service.upload_lost_item("image.jpg", "My Item")
matches = service.get_matches()
```

---

## 🎯 Best Practices

### Image Upload

1. **Use clear, well-lit photos**
2. **Include distinctive features**
3. **Multiple angles help accuracy**
4. **Minimum 200x200 pixels**

### Video Processing

1. **Start with default settings**
2. **Adjust thresholds gradually**
3. **Monitor for false positives**
4. **Process in chunks for long videos**

### Results Management

1. **Export results regularly**
2. **Document successful settings**
3. **Keep reference images**
4. **Review match history**

---

## 🚀 What's Next?

1. **Try the Demo**: Run `python quick_start_streamlit.py`
2. **Upload Your Items**: Add real lost item photos
3. **Test Your Videos**: Process your own footage
4. **Optimize Settings**: Find the best configuration
5. **Export Results**: Save your findings

---

## 📞 Support & Documentation

- **Quick Start**: `python quick_start_streamlit.py`
- **User Guide**: `STREAMLIT_GUIDE.md`
- **API Docs**: `docs/` folder
- **Command Line**: `LOST_ITEM_QUICK_START.md`

---

**Ready to start finding lost items interactively?** 🔍✨

```bash
python quick_start_streamlit.py
```