# Screenshot Matching Troubleshooting Guide

If you're having trouble with screenshot-to-video matching, this guide will help you get better results.

## 🎯 Quick Fix for Screenshot Matching

**The most common issue is the matching threshold being too high.**

### Recommended Settings:
- **Match Threshold**: Set to `0.20` - `0.30` (instead of default 0.4)
- **Detection Confidence**: Set to `0.10` - `0.20` for small objects

## 🔧 Step-by-Step Troubleshooting

### 1. Check Your Threshold Settings
In the Streamlit app sidebar:
- Lower the "Lost Item Match Threshold" to **0.25** or **0.20**
- Lower the "Detection Confidence" to **0.15** if objects are small

### 2. Image Quality Tips
For better screenshot matching:
- ✅ Use clear, well-lit screenshots
- ✅ Crop the screenshot to focus on the object
- ✅ Avoid blurry or low-resolution images
- ✅ Make sure the object is clearly visible

### 3. Screenshot Best Practices
- Take screenshots from the same video you're analyzing
- Ensure the object is in a similar lighting condition
- Avoid screenshots with heavy compression artifacts
- Use PNG format if possible (less compression than JPG)

### 4. Advanced Debugging

If you're still having issues, use the debug script:

```bash
python debug_screenshot_matching.py path/to/your/lost_item.jpg path/to/detection.jpg
```

This will show you:
- Feature analysis of your images
- Similarity scores between images
- Recommended threshold settings

## 🧪 Test the System

Run this quick test to verify screenshot matching is working:

```bash
python -c "
import sys
sys.path.append('.')
from src.reidentification.improved_matcher import ImprovedLostItemMatcher
import cv2
import numpy as np

# Create test image
img = np.ones((100, 100, 3), dtype=np.uint8) * 128
cv2.rectangle(img, (20, 20), (80, 80), (0, 0, 255), -1)
cv2.imwrite('test_item.jpg', img)

# Test matcher
matcher = ImprovedLostItemMatcher(threshold=0.25)
success = matcher.add_lost_item('test', 'test_item.jpg', 'Test Item')
print(f'Matcher working: {success}')

# Test matching
matches = matcher.match_detection(img, 'det1', 'cam1', (0,0,100,100), 1, 1.0)
print(f'Found {len(matches)} matches')
if matches:
    print(f'Confidence: {matches[0].confidence:.3f}')
"
```

## 📊 Understanding Match Scores

The system uses multiple features for matching:

- **Template Match** (50% weight): Direct image comparison - highest for screenshots
- **Color Similarity** (25% weight): Color histogram matching
- **Keypoint Similarity** (10% weight): Feature point matching
- **Shape Similarity** (8% weight): Object shape comparison
- **Texture Similarity** (4% weight): Surface texture patterns
- **Edge Similarity** (3% weight): Edge pattern matching

For screenshot matching, template matching should score very high (>0.8).

## 🚨 Common Issues and Solutions

### Issue: "No matches found even with identical screenshots"
**Solution**: Lower the threshold to 0.15-0.20

### Issue: "Too many false matches"
**Solution**: Increase the threshold to 0.35-0.40

### Issue: "Object detected but not matched"
**Solution**: 
1. Check if the lost item was properly uploaded
2. Verify the object is being detected (green boxes)
3. Lower the match threshold

### Issue: "System is too slow"
**Solution**: 
1. Reduce video resolution
2. Skip frames (process every 2nd or 3rd frame)
3. Limit max frames in settings

## 🎛️ Optimal Settings for Different Scenarios

### Screenshot from Same Video
- Match Threshold: **0.20**
- Detection Confidence: **0.15**

### Similar but Different Objects
- Match Threshold: **0.35**
- Detection Confidence: **0.20**

### High Precision (Avoid False Positives)
- Match Threshold: **0.50**
- Detection Confidence: **0.25**

### High Recall (Catch Everything)
- Match Threshold: **0.15**
- Detection Confidence: **0.10**

## 🔍 Manual Verification

If automatic matching isn't working:

1. **Check Detection**: Verify objects are being detected (green boxes)
2. **Check Upload**: Ensure lost item appears in the sidebar list
3. **Check Logs**: Look for error messages in the console
4. **Try Debug Script**: Use `debug_screenshot_matching.py` for detailed analysis

## 📞 Still Having Issues?

If you're still experiencing problems:

1. Run the debug script with your specific images
2. Check the console for error messages
3. Try with a simple test image first
4. Verify all dependencies are installed correctly

The improved matching system is specifically designed for screenshot-to-video matching and should work well with the recommended settings above.