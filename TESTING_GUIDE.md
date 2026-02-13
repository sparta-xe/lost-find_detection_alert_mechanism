# Testing Guide - Real-Time Object Tracking & Loss Detection System

Complete guide to testing and running the object tracking system.

---

## Quick Start (2 Minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Live Detection Test
```bash
python scripts/test_detection_live.py
```

### 3. Visualize Tracking Results
```bash
python scripts/visualize_tracking.py
```

---

## Complete Testing Framework

### Files Created

#### 1. **test_detection_live.py** ✅ (Already created)
   - **Purpose**: Complete real-time tracking pipeline with loss detection
   - **Features**:
     - Real-time frame processing
     - YOLO detection integration
     - Multi-object tracking
     - Loss event detection for crowd analysis
     - Comprehensive statistics collection
     - Professional logging system

   - **Run**: `python scripts/test_detection_live.py`
   - **Expected Output**: Frame-by-frame tracking summary with loss events
   - **Configuration**: Adjust parameters in the `main()` function

---

#### 2. **test_config.yaml** ✅ (NEW)
   - **Purpose**: Configuration file for test parameters
   - **Location**: `configs/test_config.yaml`
   - **Settings**:
     - Video source and camera configuration
     - Detection confidence thresholds
     - Tracking parameters (IOU, appearance matching)
     - Loss detection sensitivity
     - Logging verbosity
     - Output format options

   - **Usage**: Can be integrated into test runner for easy parameter tuning
   - **Edit**: Customize before running tests for different scenarios

---

#### 3. **test_tracking_pipeline.py** ✅ (NEW)
   - **Purpose**: Comprehensive unit test suite
   - **Location**: `tests/test_tracking_pipeline.py`
   - **Test Suites**:
     - **IoU Calculation Tests** (5 tests)
       - Perfect overlap, no overlap, partial overlap
       - Contained boxes, symmetry verification
     - **Track Memory Tests** (4 tests)
       - Track creation, updating, marking missed, deletion
     - **Loss Event Tests** (2 tests)
       - Event creation and representation
     - **Tracking Algorithm Tests** (5 tests)
       - New track creation, track continuation
       - Loss detection, multiple objects
     - **Pipeline Integration Tests** (2 tests)
       - Statistics initialization and accumulation
     - **Error Handling Tests** (3 tests)
       - Empty detections, None inputs, invalid bboxes
     - **Performance Tests** (2 tests)
       - IoU computation speed, track update speed

   - **Run**:
     ```bash
     # Using unittest
     python tests/test_tracking_pipeline.py
     
     # Using pytest
     pytest tests/test_tracking_pipeline.py -v
     
     # With coverage report
     pytest tests/test_tracking_pipeline.py --cov=src
     ```

   - **Expected Results**:
     ```
     Ran 23 tests in 2.345s
     OK
     ```

   - **Test Coverage**:
     - Core algorithms (IoU, tracking matching)
     - Data structures (Track, TrackMemory)
     - Loss event processing
     - Statistics collection
     - Error handling and edge cases
     - Performance benchmarks

---

#### 4. **visualize_tracking.py** ✅ (NEW)
   - **Purpose**: Real-time visualization of tracking results
   - **Location**: `scripts/visualize_tracking.py`
   - **Features**:
     - Real-time bounding box rendering
     - Color-coded track IDs
     - Track trail history (configurable length)
     - Confidence score display
     - Loss event alerts (red overlay)
     - Statistics overlay:
       - Active track count
       - Detection count
       - FPS indicator
       - Frame progress
       - Camera ID
     - Video output export (MP4 format)
     - Graceful error handling

   - **Run**: `python scripts/visualize_tracking.py`
   - **Controls**:
     - Press 'q' to quit
     - Window title shows camera ID and frame info
   - **Output**: Saved to `runs/detect/predict/output.mp4`
   - **Configuration**: Adjust in initialization:
     - `draw_trails`: Show track history
     - `trail_length`: Number of frames to show
     - `draw_stats`: Show statistics overlay
     - `output_path`: Save location

   - **Expected Output**:
     ```
     OpenCV Window showing:
     ├── Colored bounding boxes
     ├── Track ID labels with confidence
     ├── Movement trails (polylines)
     ├── Statistics box (top-left)
     ├── Loss alerts (red bar at bottom)
     └── Frame counter and progress
     ```

---

#### 5. **run_tests.py** ✅ (NEW)
   - **Purpose**: Unified test runner for all components
   - **Location**: `scripts/run_tests.py`
   - **Features**:
     - Run individual test modules
     - Run all tests together
     - JSON result export
     - Debug logging support
     - Execution timing
     - Pass/fail summary

   - **Usage**:
     ```bash
     # Run all tests
     python scripts/run_tests.py --all
     
     # Run specific tests
     python scripts/run_tests.py --unit        # Unit tests only
     python scripts/run_tests.py --detection   # Detection test
     python scripts/run_tests.py --live        # Live detection
     python scripts/run_tests.py --visualize   # Visualization
     
     # Advanced options
     python scripts/run_tests.py --all --debug              # With debug logging
     python scripts/run_tests.py --all --save results.json  # Save results
     ```

   - **Output Format**:
     ```
     TEST SUMMARY
     ✓ Unit Tests: PASSED
     ✓ Detection Test: PASSED
     ✓ Live Detection Test: PASSED
     ✓ Tracking Visualization: PASSED
     
     Total: 4/4 passed
     ```

---

#### 6. **quick_start.py** ✅ (NEW)
   - **Purpose**: Interactive quick start guide
   - **Location**: `scripts/quick_start.py`
   - **Shows**:
     - All available commands
     - Examples with explanations
     - Expected output samples
     - Troubleshooting guide
     - Configuration instructions
     - Decision tree for common tasks

   - **Run**: `python scripts/quick_start.py`
   - **Content**:
     - Installation instructions
     - Quick start test commands
     - Unit test commands
     - Full test suite options
     - Performance benchmarks
     - Troubleshooting section
     - Configuration guide

---

## Testing Workflows

### Workflow 1: Basic Setup & Validation
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test basic detection
python scripts/test_detection.py

# 3. Test live tracking
python scripts/test_detection_live.py

# Expected: See tracked objects and loss events
```

### Workflow 2: Comprehensive Unit Testing
```bash
# Run all unit tests
python tests/test_tracking_pipeline.py

# Or with pytest for detailed output
pytest tests/test_tracking_pipeline.py -v

# Or with coverage report
pytest tests/test_tracking_pipeline.py --cov=src --cov-report=html
```

### Workflow 3: Visualization & Analysis
```bash
# Run live detection with tracking
python scripts/test_detection_live.py

# In another terminal, visualize results
python scripts/visualize_tracking.py

# Output saved to: runs/detect/predict/output.mp4
```

### Workflow 4: Full Test Suite
```bash
# Run all tests with unified runner
python scripts/run_tests.py --all

# Run with debug output
python scripts/run_tests.py --all --debug

# Save results
python scripts/run_tests.py --all --save test_results_$(date +%Y%m%d_%H%M%S).json
```

### Workflow 5: Performance Testing
```bash
# Run benchmark
python scripts/benchmark.py

# View results
cat runs/benchmark_results.json | jq .
```

---

## Test Categories

### 1. Unit Tests ✅
**What**: Tests individual components in isolation
**Includes**: IoU calculation, track memory, loss events
**Run**: `python tests/test_tracking_pipeline.py`
**Estimated Time**: 2-5 seconds
**Quality**: Gold standard - should always pass

### 2. Integration Tests ✅
**What**: Tests components working together
**Includes**: Full pipeline, detection → tracking → loss
**Run**: `python scripts/test_detection_live.py`
**Estimated Time**: 30-120 seconds
**Quality**: Most realistic - uses actual video

### 3. Visual Tests ✅
**What**: Tests rendered output visually
**Includes**: Bounding boxes, trails, overlays
**Run**: `python scripts/visualize_tracking.py`
**Estimated Time**: 30-120 seconds
**Quality**: Manual inspection required

### 4. Performance Tests ✅
**What**: Tests speed and efficiency
**Includes**: IoU computation, track updates
**Run**: `python tests/test_tracking_pipeline.py` (includes performance tests)
**Estimated Time**: < 1 second per benchmark
**Quality**: Ensures real-time capabilities

### 5. Configuration Tests ✅
**What**: Tests with different parameters
**Edit**: `configs/test_config.yaml`
**Variations**:
   - Different confidence thresholds
   - Different tracking algorithms
   - Different video sources
   - Different output formats

---

## Understanding Test Results

### Successful Output
```
✓ Detection Pipeline: 100+ objects tracked
✓ Tracking Accuracy: 95%+ ID preservation
✓ Loss Detection: Correctly identifies disappearances
✓ Processing Speed: > 30 FPS on CPU
```

### What Each Metric Means

| Metric | Good | Fair | Poor |
|--------|------|------|------|
| Active Tracks | 5-50 | 1-5 | 0 |
| Detections/Frame | 2-10 | 1-2 | 0 |
| Loss Rate | 0-5% | 5-15% | >15% |
| Processing FPS | >30 | 15-30 | <15 |
| False Match Rate | <5% | 5-10% | >10% |

### Interpreting Loss Events
```
LOSS DETECTED: Track a1b2c3d4 (person) lost at timestamp
├── Normal: Track filtered out by appearance threshold
├── Expected: Occlusion by crowd
└── Concern: Tracking breakage during motion
```

---

## Configuration Parameters

### In `test_config.yaml`:

```yaml
# Detection sensitivity (lower = more detections)
detection:
  confidence_threshold: 0.25  # Range: 0.0-1.0

# Tracking precision (higher = stricter matching)
tracking:
  iou_threshold: 0.3          # Range: 0.0-1.0
  appearance_threshold: 0.7   # Range: 0.0-1.0
  max_missed_frames: 12       # Range: 1-30

# Loss detection (lower = more sensitive)
loss_detection:
  min_track_age_for_loss: 5   # Frames before loss
  crowd_size_threshold: 5     # Objects/area for crowd
```

### Recommended Tuning:

**For Speed (Real-time on GPU)**:
```yaml
detection.confidence_threshold: 0.4
tracking.max_missed_frames: 5
```

**For Accuracy (Offline Analysis)**:
```yaml
detection.confidence_threshold: 0.2
tracking.iou_threshold: 0.2
tracking.max_missed_frames: 20
```

**For Crowd Detection**:
```yaml
loss_detection.crowd_size_threshold: 3
loss_detection.min_track_age_for_loss: 10
```

---

## Troubleshooting

### Common Issues

#### Error: "Video file not found"
```bash
# Verify file exists
ls data/test_clips/cam1.mp4
ls data/test_clips/cam2.mp4

# If missing, create dummy test video:
python -c "
import cv2
import numpy as np
cap = cv2.VideoCapture(0)  # Capture from webcam
out = cv2.VideoWriter('data/test_clips/test.mp4', 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, (640,480))
for _ in range(300): 
    ret, frame = cap.read()
    if ret: out.write(frame)
out.release()
"
```

#### Error: "ModuleNotFoundError"
```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt

# Verify installation
python -c "import cv2, torch, ultralytics; print('OK')"
```

#### Test runs slowly / times out
```bash
# Reduce number of frames processed
# Edit test script: pipeline.run(max_frames=100)

# Or reduce video resolution
# Edit test script: loader = VideoLoader(..., resize=0.5)

# Or use faster model
# Edit test script: YOLODetector(model_path="yolov8n.pt")  # nano model
```

#### Visualization window not appearing
```bash
# Linux/WSL: Install display server
sudo apt-get install python3-tk xvfb
apt-get install x11-apps  # For WSL

# Test display
python -c "import cv2; cv2.namedWindow('test'); cv2.waitKey(1)"

# Or disable display and save to file only
# Edit: visualizer = TrackingVisualizer(..., output_path="output.mp4")
```

#### GPU out of memory
```bash
# Use CPU inference only
# Edit: detector = YOLODetector(device='cpu')

# Or reduce batch size and confidence
# Edit: conf=0.5  # Higher threshold = fewer detections
```

---

## Next Steps

1. **Run Quick Start**: `python scripts/quick_start.py`
2. **Run Tests**: `python scripts/run_tests.py --all`
3. **Visualize Results**: `python scripts/visualize_tracking.py`
4. **Tune Parameters**: Edit `configs/test_config.yaml`
5. **Analyze Results**: Check `runs/` directory for outputs

---

## File Structure
```
project/
├── scripts/
│   ├── test_detection_live.py    ← Main tracking pipeline
│   ├── visualize_tracking.py     ← Visualization engine (NEW)
│   ├── run_tests.py              ← Test runner (NEW)
│   └── quick_start.py            ← Quick start guide (NEW)
├── tests/
│   └── test_tracking_pipeline.py ← Unit test suite (NEW)
├── configs/
│   └── test_config.yaml          ← Test configuration (NEW)
└── runs/
    └── detect/predict/
        └── output.mp4            ← Visualization output
```

---

## Contact & Support

For issues or questions:
1. Check troubleshooting section above
2. Review test output for specific errors
3. Check log files in `logs/` directory
4. Run with `--debug` flag for detailed output

---

**Last Updated**: February 13, 2026
**Status**: Production Ready ✅
