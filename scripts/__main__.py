"""
Lost Item Identification System - Entry Point

Run the interactive lost item upload tool.
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Set up path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    
    from scripts.lost_item_upload import main
    sys.exit(main())
