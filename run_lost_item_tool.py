#!/usr/bin/env python
"""
Simple wrapper to run the lost item upload tool.
Ensures proper path setup before importing.
"""

import sys
from pathlib import Path

# Set up path BEFORE any imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now import and run
if __name__ == "__main__":
    from scripts.lost_item_upload import main
    sys.exit(main())
