"""
Motion Detection Fallback System
Provides backup detection when YOLO fails using frame differencing.
This ensures we never miss motion even if object detection fails.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MotionRegion:
    """Represents a detected motion region."""
    bbox: Tuple[int, int, int, int]
    area: float
    centroid: Tuple[int, int]
    motion_intensity: float
    timestamp: float

class MotionFallbackDetector:
    """
    Fallback motion detection system using frame differencing.
    Detects motion even when YOLO object detection fails.
    """
    
    def __init__(self,
                 motion_threshold: int = 25,
                 min_contour_area: int = 500,
                 max_contour_area: int = 50000,
                 gaussian_blur_size: int = 5,
                 morphology_kernel_size: int = 5):
        """
        Initialize motion fallback detector.
        
        Args:
            motion_threshold: Threshold for frame difference
            min_contour_area: Minimum area for motion detection
            max_contour_area: Maximum area to filter out large motions
            gaussian_blur_size: Gaussian blur kernel size
            morphology_kernel_size: Morphological operations kernel size
        """
        self.motion_threshold = motion_threshold
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        self.gaussian_blur_size = gaussian_blur_size
        self.morphology_kernel_size = morphology_kernel_size
        
        # Frame storage
        self.prev_frame = None
        self.prev_gray = None
        
        # Background subtraction (additional method)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=16,
            history=500
        )
        
        # Morphological kernel
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.morphology_kernel_size, self.morphology_kernel_size)
        )
        
        logger.info("Motion fallback detector initialized")
    
    def detect_motion(self, frame: np.ndarray, timestamp: float) -> List[MotionRegion]:
        """
        Detect motion regions in the current frame.
        
        Args:
            frame: Current video frame
            timestamp: Frame timestamp
            
        Returns:
            List of detected motion regions
        """
        motion_regions = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.gaussian_blur_size, self.gaussian_blur_size), 0)
        
        # Method 1: Frame differencing
        if self.prev_gray is not None:
            diff_regions = self._frame_differencing(gray, timestamp)
            motion_regions.extend(diff_regions)
        
        # Method 2: Background subtraction
        bg_regions = self._background_subtraction(frame, timestamp)
        motion_regions.extend(bg_regions)
        
        # Method 3: Three-frame differencing (if we have enough frames)
        if len(motion_regions) == 0 and self.prev_gray is not None:
            # More sensitive detection when other methods fail
            sensitive_regions = self._sensitive_frame_diff(gray, timestamp)
            motion_regions.extend(sensitive_regions)
        
        # Update frame history
        self.prev_frame = frame.copy()
        self.prev_gray = gray.copy()
        
        # Remove overlapping regions
        motion_regions = self._merge_overlapping_regions(motion_regions)
        
        if motion_regions:
            logger.debug(f"Motion fallback detected {len(motion_regions)} regions")
        
        return motion_regions
    
    def _frame_differencing(self, gray: np.ndarray, timestamp: float) -> List[MotionRegion]:
        """Standard frame differencing method."""
        regions = []
        
        # Calculate absolute difference
        diff = cv2.absdiff(self.prev_gray, gray)
        
        # Apply threshold
        _, thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up noise
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_contour_area <= area <= self.max_contour_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                
                # Calculate motion intensity (average pixel difference in region)
                roi_diff = diff[y:y+h, x:x+w]
                motion_intensity = np.mean(roi_diff)
                
                region = MotionRegion(
                    bbox=(x, y, x + w, y + h),
                    area=area,
                    centroid=(cx, cy),
                    motion_intensity=motion_intensity,
                    timestamp=timestamp
                )
                regions.append(region)
        
        return regions
    
    def _background_subtraction(self, frame: np.ndarray, timestamp: float) -> List[MotionRegion]:
        """Background subtraction method."""
        regions = []
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up the mask
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_contour_area <= area <= self.max_contour_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                
                # Motion intensity from foreground mask
                roi_mask = fg_mask[y:y+h, x:x+w]
                motion_intensity = np.mean(roi_mask)
                
                region = MotionRegion(
                    bbox=(x, y, x + w, y + h),
                    area=area,
                    centroid=(cx, cy),
                    motion_intensity=motion_intensity,
                    timestamp=timestamp
                )
                regions.append(region)
        
        return regions
    
    def _sensitive_frame_diff(self, gray: np.ndarray, timestamp: float) -> List[MotionRegion]:
        """More sensitive frame differencing for when other methods fail."""
        regions = []
        
        # Use lower threshold for more sensitivity
        sensitive_threshold = max(15, self.motion_threshold - 10)
        
        # Calculate absolute difference
        diff = cv2.absdiff(self.prev_gray, gray)
        
        # Apply sensitive threshold
        _, thresh = cv2.threshold(diff, sensitive_threshold, 255, cv2.THRESH_BINARY)
        
        # Less aggressive morphological operations
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, small_kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Lower minimum area for sensitive detection
            min_sensitive_area = max(200, self.min_contour_area // 2)
            
            if min_sensitive_area <= area <= self.max_contour_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                
                # Calculate motion intensity
                roi_diff = diff[y:y+h, x:x+w]
                motion_intensity = np.mean(roi_diff)
                
                region = MotionRegion(
                    bbox=(x, y, x + w, y + h),
                    area=area,
                    centroid=(cx, cy),
                    motion_intensity=motion_intensity,
                    timestamp=timestamp
                )
                regions.append(region)
        
        return regions
    
    def _merge_overlapping_regions(self, regions: List[MotionRegion]) -> List[MotionRegion]:
        """Merge overlapping motion regions."""
        if len(regions) <= 1:
            return regions
        
        merged = []
        used = set()
        
        for i, region1 in enumerate(regions):
            if i in used:
                continue
            
            # Start with current region
            merged_bbox = list(region1.bbox)
            merged_area = region1.area
            merged_intensity = region1.motion_intensity
            count = 1
            used.add(i)
            
            # Check for overlaps with remaining regions
            for j, region2 in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue
                
                if self._regions_overlap(region1.bbox, region2.bbox):
                    # Merge bounding boxes
                    merged_bbox[0] = min(merged_bbox[0], region2.bbox[0])  # x1
                    merged_bbox[1] = min(merged_bbox[1], region2.bbox[1])  # y1
                    merged_bbox[2] = max(merged_bbox[2], region2.bbox[2])  # x2
                    merged_bbox[3] = max(merged_bbox[3], region2.bbox[3])  # y2
                    
                    # Average other properties
                    merged_area += region2.area
                    merged_intensity += region2.motion_intensity
                    count += 1
                    used.add(j)
            
            # Create merged region
            merged_centroid = (
                (merged_bbox[0] + merged_bbox[2]) // 2,
                (merged_bbox[1] + merged_bbox[3]) // 2
            )
            
            merged_region = MotionRegion(
                bbox=tuple(merged_bbox),
                area=merged_area / count,
                centroid=merged_centroid,
                motion_intensity=merged_intensity / count,
                timestamp=region1.timestamp
            )
            merged.append(merged_region)
        
        return merged
    
    def _regions_overlap(self, bbox1: Tuple[int, int, int, int], 
                        bbox2: Tuple[int, int, int, int]) -> bool:
        """Check if two bounding boxes overlap."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)
    
    def create_motion_overlay(self, frame: np.ndarray, regions: List[MotionRegion]) -> np.ndarray:
        """Create overlay showing detected motion regions."""
        overlay = frame.copy()
        
        for region in regions:
            x1, y1, x2, y2 = region.bbox
            
            # Color based on motion intensity
            if region.motion_intensity > 100:
                color = (0, 0, 255)  # Red for high intensity
            elif region.motion_intensity > 50:
                color = (0, 165, 255)  # Orange for medium intensity
            else:
                color = (0, 255, 255)  # Yellow for low intensity
            
            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw centroid
            cv2.circle(overlay, region.centroid, 5, color, -1)
            
            # Add text
            text = f"Motion: {region.motion_intensity:.0f}"
            cv2.putText(overlay, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return overlay
    
    def reset(self):
        """Reset detector state."""
        self.prev_frame = None
        self.prev_gray = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=16,
            history=500
        )
        logger.info("Motion fallback detector reset")