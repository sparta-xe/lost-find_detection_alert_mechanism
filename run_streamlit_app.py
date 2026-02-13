#!/usr/bin/env python
"""
Launcher script for the Streamlit Lost Item Detection Application

This script sets up the environment and launches the Streamlit app.
"""

import sys
import subprocess
from pathlib import Path

# Set up project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """Launch the Streamlit application."""
    print("🚀 Starting Lost Item Detection Streamlit App...")
    print(f"📁 Project root: {PROJECT_ROOT}")
    
    # Ensure data directories exist
    (PROJECT_ROOT / "data" / "lost_items").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "test_clips").mkdir(parents=True, exist_ok=True)
    
    # Launch Streamlit
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(PROJECT_ROOT / "streamlit_app.py"),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]
        
        print("🌐 Starting Streamlit server...")
        print("📱 Open your browser to: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the server")
        
        subprocess.run(cmd, cwd=PROJECT_ROOT)
        
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())