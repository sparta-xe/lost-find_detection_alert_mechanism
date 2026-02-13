"""
Lost Item Identification Service

Integrates image upload capability with real-time tracking to identify
lost items in camera feeds. Handles image uploads, manages lost items,
and reports matches in real-time.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
from src.reidentification.improved_matcher import ImprovedLostItemMatcher, MatchResult

logger = logging.getLogger(__name__)


class LostItemService:
    """
    Service for managing lost item uploads and tracking.
    Provides REST-like interface for uploading items and retrieving matches.
    """
    
    def __init__(self, upload_dir: str = "data/lost_items", 
                 match_threshold: float = 0.4):
        """
        Initialize the lost item service.
        
        Args:
            upload_dir: Directory for storing uploaded images
            match_threshold: Confidence threshold for matches
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        self.matcher = ImprovedLostItemMatcher(threshold=match_threshold)
        self.next_item_id = 1
        
        logger.info(f"LostItemService initialized. Upload dir: {self.upload_dir}")
    
    def upload_lost_item(self, image_path: str, name: str, 
                        description: str = "") -> Tuple[bool, str]:
        """
        Upload an image of a lost item.
        
        Args:
            image_path: Path to the image file
            name: Name/category of the lost item
            description: Additional description
        
        Returns:
            (success, item_id or error_message)
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            msg = f"Image file not found: {image_path}"
            logger.error(msg)
            return False, msg
        
        try:
            # Generate item ID
            item_id = f"item_{self.next_item_id:04d}"
            self.next_item_id += 1
            
            # Copy image to upload directory
            filename = f"{item_id}_{image_path.name}"
            dest_path = self.upload_dir / filename
            
            import shutil
            shutil.copy2(image_path, dest_path)
            
            # Add to matcher
            if self.matcher.add_lost_item(item_id, str(dest_path), name, description):
                msg = f"Successfully uploaded lost item: {item_id} ({name})"
                logger.info(msg)
                return True, item_id
            else:
                msg = f"Failed to process image for lost item"
                logger.error(msg)
                return False, msg
        
        except Exception as e:
            msg = f"Error uploading lost item: {e}"
            logger.error(msg)
            return False, msg
    
    def mark_found(self, item_id: str) -> bool:
        """Mark a lost item as found."""
        if self.matcher.remove_lost_item(item_id):
            logger.info(f"Marked item {item_id} as found")
            return True
        return False
    
    def get_lost_items(self) -> List[Dict]:
        """Get list of all lost items."""
        items = []
        for lost_item in self.matcher.get_lost_items():
            items.append({
                "item_id": lost_item.item_id,
                "name": lost_item.name,
                "description": lost_item.description,
                "image_path": lost_item.image_path,
                "upload_time": lost_item.upload_time.isoformat(),
                "timestamp": lost_item.upload_time.isoformat()  # For compatibility
            })
        return items
    
    def get_matches(self, item_id: Optional[str] = None) -> List[Dict]:
        """Get matches for a lost item or all items."""
        matches = []
        for match in self.matcher.get_match_history(item_id):
            matches.append({
                "lost_item_id": match.lost_item_id,
                "detection_id": match.detection_id,
                "camera_id": match.camera_id,
                "confidence": f"{match.confidence:.3f}",
                "frame_number": match.frame_number,
                "timestamp": match.timestamp,
                "bbox": match.bbox,
                "reasons": match.match_reasons
            })
        return matches
    
    def get_statistics(self) -> Dict:
        """Get statistics about lost items and matches."""
        stats = self.matcher.get_statistics()
        stats['upload_directory'] = str(self.upload_dir)
        stats['match_threshold'] = self.matcher.threshold
        return stats


class LostItemReporter:
    """Generates reports and alerts for lost item matches."""
    
    def __init__(self, service: LostItemService):
        self.service = service
    
    def report_matches(self, camera_id: str = None) -> str:
        """Generate a formatted report of all matches."""
        stats = self.service.get_statistics()
        
        report = []
        report.append("╔════════════════════════════════════════════════════════╗")
        report.append("║         LOST ITEM IDENTIFICATION REPORT                ║")
        report.append("╚════════════════════════════════════════════════════════╝")
        report.append("")
        
        # Lost items summary
        report.append(f"📋 Lost Items Registered: {stats['lost_items_registered']}")
        report.append(f"🎯 Total Matches Found: {stats['total_matches_found']}")
        report.append(f"✅ Items Matched: {stats['items_matched']}")
        report.append(f"📊 Avg Confidence: {stats['avg_confidence']:.2%}")
        report.append("")
        
        # Matches by item
        items = self.service.get_lost_items()
        if items:
            report.append("═" * 54)
            report.append("LOST ITEMS")
            report.append("═" * 54)
            for item in items:
                report.append(f"\n🔍 {item['item_id']}: {item['name']}")
                report.append(f"   Description: {item['description']}")
                report.append(f"   Uploaded: {item['upload_time']}")
                
                # Matches for this item
                matches = self.service.get_matches(item['item_id'])
                if matches:
                    report.append(f"   📍 {len(matches)} match(es) found:")
                    for match in sorted(matches, 
                                      key=lambda m: float(m['confidence']), 
                                      reverse=True):
                        report.append(
                            f"      • {match['detection_id']} "
                            f"({match['confidence']} confidence) "
                            f"on camera {match['camera_id']}, "
                            f"frame {match['frame_number']}"
                        )
                        if match['reasons']:
                            reasons_str = ", ".join(match['reasons'])
                            report.append(f"        Reason: {reasons_str}")
                else:
                    report.append(f"   ❌ No matches yet")
        else:
            report.append("\n📭 No lost items registered yet")
        
        return "\n".join(report)
    
    def export_matches(self, output_path: str) -> bool:
        """Export matches to JSON file."""
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "statistics": self.service.get_statistics(),
                "lost_items": self.service.get_lost_items(),
                "matches": self.service.get_matches()
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported matches to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting matches: {e}")
            return False
