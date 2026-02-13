"""
Professional overlay system for video frames with timestamp and camera information.
"""
import cv2
import numpy as np
from datetime import datetime
from typing import Tuple, Optional

class VideoOverlay:
    """Professional video overlay system with customizable styling."""
    
    def __init__(self, 
                 font=cv2.FONT_HERSHEY_SIMPLEX,
                 font_scale=0.7,
                 color=(0, 255, 0),
                 thickness=2,
                 background_alpha=0.7):
        """
        Initialize overlay system.
        
        Args:
            font: OpenCV font type
            font_scale: Font size multiplier
            color: Text color (B, G, R)
            thickness: Text thickness
            background_alpha: Background transparency (0-1)
        """
        self.font = font
        self.font_scale = font_scale
        self.color = color
        self.thickness = thickness
        self.background_alpha = background_alpha
    
    def add_overlay(self, frame: np.ndarray, camera_id: str, 
                   custom_timestamp: Optional[str] = None,
                   additional_info: Optional[dict] = None) -> np.ndarray:
        """
        Add professional overlay to video frame.
        
        Args:
            frame: Input video frame
            camera_id: Camera identifier
            custom_timestamp: Custom timestamp (if None, uses current time)
            additional_info: Dictionary of additional info to display
            
        Returns:
            Frame with overlay applied
        """
        overlay_frame = frame.copy()
        
        # Generate timestamp
        if custom_timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            timestamp = custom_timestamp
        
        # Prepare overlay text
        overlay_texts = [
            f"Camera: {camera_id}",
            f"Time: {timestamp}"
        ]
        
        # Add additional information
        if additional_info:
            for key, value in additional_info.items():
                overlay_texts.append(f"{key}: {value}")
        
        # Calculate overlay dimensions
        text_sizes = []
        for text in overlay_texts:
            (text_width, text_height), baseline = cv2.getTextSize(
                text, self.font, self.font_scale, self.thickness
            )
            text_sizes.append((text_width, text_height + baseline))
        
        # Calculate background rectangle
        max_width = max([size[0] for size in text_sizes])
        total_height = sum([size[1] for size in text_sizes]) + (len(overlay_texts) - 1) * 10
        
        # Add semi-transparent background
        background_rect = np.zeros((total_height + 20, max_width + 20, 3), dtype=np.uint8)
        background_rect[:] = (0, 0, 0)  # Black background
        
        # Blend background with frame
        y_end = min(total_height + 20, overlay_frame.shape[0])
        x_end = min(max_width + 20, overlay_frame.shape[1])
        
        overlay_region = overlay_frame[0:y_end, 0:x_end]
        blended = cv2.addWeighted(
            overlay_region, 
            1 - self.background_alpha, 
            background_rect[0:y_end, 0:x_end], 
            self.background_alpha, 
            0
        )
        overlay_frame[0:y_end, 0:x_end] = blended
        
        # Add text overlay
        y_offset = 25
        for i, text in enumerate(overlay_texts):
            cv2.putText(
                overlay_frame, 
                text,
                (10, y_offset),
                self.font,
                self.font_scale, 
                self.color, 
                self.thickness
            )
            y_offset += text_sizes[i][1] + 10
        
        return overlay_frame
    
    def add_detection_overlay(self, frame: np.ndarray, detections: list,
                            show_confidence: bool = True,
                            show_labels: bool = True) -> np.ndarray:
        """
        Add detection boxes and labels to frame.
        
        Args:
            frame: Input video frame
            detections: List of detection objects
            show_confidence: Whether to show confidence scores
            show_labels: Whether to show object labels
            
        Returns:
            Frame with detection overlay
        """
        overlay_frame = frame.copy()
        
        for detection in detections:
            # Extract detection info
            if hasattr(detection, 'bbox'):
                bbox = detection.bbox
                label = getattr(detection, 'label', 'object')
                confidence = getattr(detection, 'confidence', 0.0)
            else:
                bbox = detection.get('bbox')
                label = detection.get('label', 'object')
                confidence = detection.get('confidence', 0.0)
            
            if bbox is None:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Choose color based on confidence
            if confidence > 0.7:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.5:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 165, 255)  # Orange for low confidence
            
            # Draw bounding box
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            if show_labels and show_confidence:
                text = f"{label}: {confidence:.2f}"
            elif show_labels:
                text = label
            elif show_confidence:
                text = f"{confidence:.2f}"
            else:
                continue
            
            # Add label background
            (text_width, text_height), baseline = cv2.getTextSize(
                text, self.font, self.font_scale * 0.8, self.thickness
            )
            
            cv2.rectangle(
                overlay_frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width + 5, y1),
                color,
                -1
            )
            
            # Add label text
            cv2.putText(
                overlay_frame,
                text,
                (x1 + 2, y1 - baseline - 2),
                self.font,
                self.font_scale * 0.8,
                (0, 0, 0),  # Black text
                self.thickness
            )
        
        return overlay_frame
    
    def add_status_overlay(self, frame: np.ndarray, status_info: dict,
                          position: str = "bottom_right") -> np.ndarray:
        """
        Add system status overlay to frame.
        
        Args:
            frame: Input video frame
            status_info: Dictionary with status information
            position: Position for status overlay ("bottom_right", "bottom_left", etc.)
            
        Returns:
            Frame with status overlay
        """
        overlay_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Prepare status text
        status_texts = []
        for key, value in status_info.items():
            status_texts.append(f"{key}: {value}")
        
        # Calculate text dimensions
        text_sizes = []
        for text in status_texts:
            (text_width, text_height), baseline = cv2.getTextSize(
                text, self.font, self.font_scale * 0.6, 1
            )
            text_sizes.append((text_width, text_height + baseline))
        
        max_width = max([size[0] for size in text_sizes]) if text_sizes else 0
        total_height = sum([size[1] for size in text_sizes]) + (len(status_texts) - 1) * 5
        
        # Calculate position
        if position == "bottom_right":
            x_start = w - max_width - 20
            y_start = h - total_height - 20
        elif position == "bottom_left":
            x_start = 10
            y_start = h - total_height - 20
        elif position == "top_right":
            x_start = w - max_width - 20
            y_start = 20
        else:  # top_left
            x_start = 10
            y_start = 20
        
        # Add semi-transparent background
        if max_width > 0 and total_height > 0:
            background_rect = np.zeros((total_height + 10, max_width + 10, 3), dtype=np.uint8)
            background_rect[:] = (0, 0, 0)
            
            y_end = min(y_start + total_height + 10, h)
            x_end = min(x_start + max_width + 10, w)
            
            if y_end > y_start and x_end > x_start:
                overlay_region = overlay_frame[y_start:y_end, x_start:x_end]
                bg_region = background_rect[0:y_end-y_start, 0:x_end-x_start]
                
                blended = cv2.addWeighted(overlay_region, 0.3, bg_region, 0.7, 0)
                overlay_frame[y_start:y_end, x_start:x_end] = blended
        
        # Add status text
        y_offset = y_start + 20
        for i, text in enumerate(status_texts):
            cv2.putText(
                overlay_frame,
                text,
                (x_start + 5, y_offset),
                self.font,
                self.font_scale * 0.6,
                (255, 255, 255),  # White text
                1
            )
            y_offset += text_sizes[i][1] + 5
        
        return overlay_frame

# Convenience function for backward compatibility
def add_overlay(frame: np.ndarray, camera_id: str) -> np.ndarray:
    """Simple overlay function for basic timestamp and camera ID."""
    overlay = VideoOverlay()
    return overlay.add_overlay(frame, camera_id)