#!/usr/bin/env python
"""
Quick Start Script for Streamlit Lost Item Detection

This script sets up the demo environment and launches the Streamlit app.
"""

import sys
import subprocess
from pathlib import Path

# Set up project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """Quick start the Streamlit application with demo setup."""
    
    print("🔍 Lost Item Detection - Streamlit Quick Start")
    print("=" * 50)
    
    # Step 1: Setup demo environment
    print("\n📋 Step 1: Setting up demo environment...")
    try:
        from scripts.setup_streamlit_demo import setup_demo_environment
        setup_demo_environment()
    except Exception as e:
        print(f"⚠️  Demo setup failed: {e}")
        print("   Continuing anyway - you can add items manually")
    
    # Step 2: Launch Streamlit
    print("\n🚀 Step 2: Launching Streamlit application...")
    try:
        from run_streamlit_app import main as launch_streamlit
        return launch_streamlit()
    except Exception as e:
        print(f"❌ Failed to launch Streamlit: {e}")
        print("\n💡 Try running manually:")
        print("   python run_streamlit_app.py")
        print("   or")
        print("   streamlit run streamlit_app.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())