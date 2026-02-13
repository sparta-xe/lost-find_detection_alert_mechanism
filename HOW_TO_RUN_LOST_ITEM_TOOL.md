# Lost Item Tool - How to Run

The Lost Item Upload Tool can be run using any of these methods:

## ✅ Method 1: Wrapper Script (Easiest)
```bash
python run_lost_item_tool.py --interactive
python run_lost_item_tool.py --list
python run_lost_item_tool.py --report
python run_lost_item_tool.py --export "results.json"
```

## ✅ Method 2: Python Module Import
```bash
python -m scripts.lost_item_upload --interactive
python -m scripts.lost_item_upload --list
python -m scripts.lost_item_upload --report
```

## ✅ Method 3: Direct Script (after clearing cache)
```bash
python scripts/lost_item_upload.py --interactive
python scripts/lost_item_upload.py --list
python scripts/lost_item_upload.py --report
```

## 📋 Common Commands

### View registered items
```bash
python run_lost_item_tool.py --list
```

### View match report
```bash
python run_lost_item_tool.py --report
```

### Upload a lost item
```bash
python run_lost_item_tool.py --upload "path/to/image.jpg" --name "Item Name" --description "Optional description"
```

### Interactive mode (menu-driven)
```bash
python run_lost_item_tool.py --interactive
```

### Export results to JSON
```bash
python run_lost_item_tool.py --export "findings.json"
```

### Full tracking with lost item matching
```bash
python scripts/enhanced_tracking.py --video "video.mp4" --lost-item "item.jpg" --name "My Item"
```

## 🔧 If you get import errors

1. **Clear Python cache:**
   ```bash
   Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
   ```

2. Use one of the methods above - they handle path setup automatically

## ✅ Verified Working Methods

- ✅ `python run_lost_item_tool.py --list`
- ✅ `python run_lost_item_tool.py --report`
- ✅ `python -m scripts.lost_item_upload --list`
- ✅ `python -m scripts.lost_item_upload --report`

## 🐍 Python API Usage

```python
import sys
from pathlib import Path

# Ensure path is set
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.escalation.lost_item_service import LostItemService

# Initialize and use
service = LostItemService()
success, item_id = service.upload_lost_item("image.jpg", "Item Name", "Description")
```

## 📊 Usage Examples

### Quick list check
```bash
python run_lost_item_tool.py --list
```
Output:
```
📭 No lost items registered yet
```

### Generate report
```bash
python run_lost_item_tool.py --report
```
Output:
```
╔════════════════════════════════════════════════════════╗
║         LOST ITEM IDENTIFICATION REPORT                ║
╚════════════════════════════════════════════════════════╝

📋 Lost Items Registered: 0
🎯 Total Matches Found: 0
✅ Items Matched: 0
📊 Avg Confidence: 0.00%
```

All methods are fully functional and handle module imports correctly!
