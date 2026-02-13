"""
Quick Start Guide - Commands for Testing the Object Tracking System

This file provides convenient commands for common testing tasks.
Copy and paste any command into your terminal to run it.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def print_commands():
    """Print all available test commands."""
    
    commands = {
        "INSTALLATION & SETUP": {
            "Install Dependencies": f"pip install -r requirements.txt",
            "Install PyTest": f"pip install pytest pytest-cov",
            "Verify Installation": f"{sys.executable} -c \"import cv2, torch, ultralytics; print('All dependencies OK')\"",
        },
        
        "QUICK START TESTS": {
            "Test Detection (Basic)": f"{sys.executable} {PROJECT_ROOT}/scripts/test_detection.py",
            "Test Detection Live": f"{sys.executable} {PROJECT_ROOT}/scripts/test_detection_live.py",
            "Visualize Tracking": f"{sys.executable} {PROJECT_ROOT}/scripts/visualize_tracking.py",
        },
        
        "UNIT TESTS": {
            "Run All Unit Tests": f"{sys.executable} {PROJECT_ROOT}/tests/test_tracking_pipeline.py",
            "Run Unit Tests (PyTest)": f"{sys.executable} -m pytest {PROJECT_ROOT}/tests/test_tracking_pipeline.py -v",
            "Run with Coverage": f"{sys.executable} -m pytest {PROJECT_ROOT}/tests/ --cov=src --cov-report=html",
        },
        
        "COMPREHENSIVE TEST SUITE": {
            "Run All Tests": f"{sys.executable} {PROJECT_ROOT}/scripts/run_tests.py --all",
            "Run Tests (Verbose)": f"{sys.executable} {PROJECT_ROOT}/scripts/run_tests.py --all --debug",
            "Run & Save Results": f"{sys.executable} {PROJECT_ROOT}/scripts/run_tests.py --all --save test_results.json",
        },
        
        "INDIVIDUAL TEST MODULES": {
            "Test Tracking Only": f"{sys.executable} {PROJECT_ROOT}/scripts/test_tracking_live.py",
            "Test ReID": f"{sys.executable} {PROJECT_ROOT}/scripts/test_reid.py",
            "Test Ingestion": f"{sys.executable} {PROJECT_ROOT}/scripts/test_ingestion.py",
        },
        
        "PERFORMANCE & BENCHMARKS": {
            "Benchmark Detection": f"{sys.executable} {PROJECT_ROOT}/scripts/benchmark.py",
            "List Installed Packages": f"pip list | grep -E 'torch|opencv|ultralytics'",
        },
    }
    
    print("\n" + "="*80)
    print("OBJECT TRACKING SYSTEM - QUICK START GUIDE")
    print("="*80 + "\n")
    
    for category, cmds in commands.items():
        print(f"\n{category}")
        print("-" * 80)
        
        for description, command in cmds.items():
            print(f"\n  {description}:")
            print(f"    {command}\n")
    
    print("\n" + "="*80)
    print("QUICK DECISION TREE")
    print("="*80 + "\n")
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║ What do you want to do?                                                    ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  1. First Time Setup?                                                      ║
║     → pip install -r requirements.txt                                      ║
║     → python scripts/test_detection.py                                     ║
║                                                                            ║
║  2. Test Real-Time Tracking?                                               ║
║     → python scripts/test_detection_live.py                                ║
║                                                                            ║
║  3. See Tracking Visualization?                                            ║
║     → python scripts/visualize_tracking.py                                 ║
║                                                                            ║
║  4. Run Complete Test Suite?                                               ║
║     → python scripts/run_tests.py --all                                    ║
║                                                                            ║
║  5. Run Unit Tests Only?                                                   ║
║     → python tests/test_tracking_pipeline.py                               ║
║                                                                            ║
║  6. Benchmark Performance?                                                 ║
║     → python scripts/benchmark.py                                          ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n" + "="*80)
    print("EXPECTED OUTPUT SAMPLES")
    print("="*80 + "\n")
    
    print("""
WHEN RUNNING test_detection_live.py:
────────────────────────────────────────────────────────────────────────────
2026-02-13 14:23:45 - ObjectTracking - [INFO] - Initializing tracking pipeline
2026-02-13 14:23:48 - ObjectTracking - [INFO] - Starting real-time tracking...

======================================================================
FRAME 30 SUMMARY
======================================================================
Active Tracks: 5

TRACKABLE OBJECTS (5 active):
  1. ID: a1b2c3d4 | Label: person           | BBox: [120, 150, 200, 350] | Missed: 0
  2. ID: e5f6g7h8 | Label: person           | BBox: [400, 200, 480, 420] | Missed: 1
  ...

PIPELINE STATISTICS (Updated: 2026-02-13 14:23:52)
  Frames Processed:              1200
  Total Detections:              4800
  Active Tracks:                   12
  Lost Objects:                    45
  Loss Rate:                        1.25%


WHEN RUNNING visualize_tracking.py:
────────────────────────────────────────────────────────────────────────────
[OpenCV window appears showing video with colored bounding boxes and track IDs]

Real-time overlays:
  - Color-coded bounding boxes (one color per track ID)
  - Track trails showing movement history
  - Statistics box (FPS, active tracks, detections)
  - Red alerts for loss events
  - Frame progress indicator


WHEN RUNNING test_tracking_pipeline.py:
────────────────────────────────────────────────────────────────────────────
test_iou_perfect_overlap (test_IoUCalculation) ... ok
test_iou_no_overlap (test_IoUCalculation) ... ok
test_iou_partial_overlap (test_IoUCalculation) ... ok
...
Ran 45 tests in 2.345s
OK
    """)
    
    print("\n" + "="*80)
    print("TROUBLESHOOTING")
    print("="*80 + "\n")
    
    print("""
Issue: "ModuleNotFoundError: No module named 'cv2'"
→ Solution: pip install opencv-python

Issue: "ModuleNotFoundError: No module named 'ultralytics'"
→ Solution: pip install ultralytics

Issue: "Video file not found"
→ Solution: Verify video exists at: data/test_clips/cam1.mp4

Issue: "CUDA out of memory" (if using GPU)
→ Solution: Run detection.py --cpu or reload your Python:
    import torch
    torch.cuda.empty_cache()

Issue: Tests hang or timeout
→ Solution: 
  1. Reduce max_frames: pipeline.run(max_frames=10)
  2. Lower detection confidence: YOLODetector(conf=0.5)
  3. Check system resources: top or Task Manager

Issue: Visualization window not showing
→ Solution: 
  1. On Linux: Install: sudo apt-get install python3-tk
  2. On WSL: Use X11 forwarding or XLaunch
  3. Add --no-display flag (if available)
    """)
    
    print("\n" + "="*80)
    print("CONFIGURATION & CUSTOMIZATION")
    print("="*80 + "\n")
    
    print("""
Edit configs/test_config.yaml to customize:
  - Video source and camera IDs
  - Detection confidence threshold
  - Tracking parameters (IOU, appearance thresholds)
  - Loss detection sensitivity
  - Output format and visualization options
  - Logging verbosity
    """)
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print_commands()
