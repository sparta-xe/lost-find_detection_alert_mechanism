"""
Lost Item Matcher - Re-Identification System

Matches uploaded lost item images against detected objects in video streams.
Uses multi-modal features (color, shape, texture) for robust matching even
with low resolution, occlusions, and angle changes.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cv2
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class LostItem:
    """Represents a lost item to be found."""
    item_id: str
    image_path: str
    name: str
    description: str
    upload_time: datetime
    features: Dict = None  # Multi-modal features
    
    def __repr__(self):
        return f"<LostItem id={self.item_id} name='{self.name}' uploaded={self.upload_time.strftime('%Y-%m-%d %H:%M:%S')}>"


@dataclass
class MatchResult:
    """Result of a lost item matching attempt."""
    lost_item_id: str
    detection_id: str
    camera_id: str
    confidence: float
    frame_number: int
    timestamp: float
    bbox: Tuple[int, int, int, int]
    match_reasons: List[str]  # Why it matched
    
    def __repr__(self):
        return (
            f"<Match lost_item={self.lost_item_id} detection={self.detection_id} "
            f"conf={self.confidence:.2f} camera={self.camera_id}>"
        )


# ============================================================================
# FEATURE EXTRACTORS
# ============================================================================

class ColorHistogramExtractor:
    """Extract color histogram features for robust color matching."""
    
    def __init__(self, bins: int = 32):
        self.bins = bins
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HSV color histogram.
        
        Args:
            image: BGR image
        
        Returns:
            Normalized histogram features
        """
        if image is None or image.size == 0:
            return np.zeros(self.bins * 3)
        
        # Convert to HSV for better color matching
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract histograms for each channel
        histograms = []
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [self.bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)
        
        return np.concatenate(histograms)
    
    def compare(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compare two histograms (0-1, higher is better)."""
        if hist1.size == 0 or hist2.size == 0:
            return 0.0
        
        # Bhattacharyya distance
        distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        return 1.0 - distance  # Convert distance to similarity


class EdgeFeatureExtractor:
    """Extract edge/shape features for robust shape matching."""
    
    def __init__(self, threshold1: int = 100, threshold2: int = 200):
        self.threshold1 = threshold1
        self.threshold2 = threshold2
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract edge features using Canny edge detection.
        
        Args:
            image: BGR image
        
        Returns:
            Edge features (flattened and normalized)
        """
        if image is None or image.size == 0:
            return np.zeros(256)  # Fixed size
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.threshold1, self.threshold2)
        
        # Resize to fixed size for comparison
        edges_resized = cv2.resize(edges, (16, 16))
        
        # Normalize
        features = edges_resized.astype(np.float32) / 255.0
        return features.flatten()
    
    def compare(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compare edge features using correlation."""
        if feat1.size == 0 or feat2.size == 0:
            return 0.0
        
        correlation = np.corrcoef(feat1, feat2)[0, 1]
        if np.isnan(correlation):
            return 0.0
        return max(0.0, correlation)  # Clamp to [0, 1]


class TextureFeatureExtractor:
    """Extract texture features using Local Binary Patterns."""
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract texture features.
        
        Args:
            image: BGR image
        
        Returns:
            Texture features
        """
        if image is None or image.size == 0:
            return np.zeros(59)  # LBP histogram size
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize for consistency
        gray = cv2.resize(gray, (64, 64))
        
        # Simple LBP-like feature: gradient magnitudes
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Histogram of magnitudes
        hist, _ = np.histogram(magnitude.flatten(), bins=59, range=(0, 256))
        return hist.astype(np.float32) / (hist.sum() + 1e-6)
    
    def compare(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compare texture features using EMD-like distance."""
        if feat1.size == 0 or feat2.size == 0:
            return 0.0
        
        # Chi-square distance
        chi_sq = np.sum(((feat1 - feat2) ** 2) / (feat1 + feat2 + 1e-6))
        return 1.0 / (1.0 + chi_sq)


# ============================================================================
# LOST ITEM MATCHER
# ============================================================================

class LostItemMatcher:
    """
    Matches lost items against detected objects using multi-modal features.
    Robust to low resolution, lighting changes, and partial occlusions.
    Enhanced for better small object matching.
    """
    
    def __init__(self, color_weight: float = 0.4, edge_weight: float = 0.3, 
                 texture_weight: float = 0.3, threshold: float = 0.5):
        """
        Initialize matcher with feature extractors.
        
        Args:
            color_weight: Weight for color histogram matching
            edge_weight: Weight for edge feature matching
            texture_weight: Weight for texture feature matching
            threshold: Confidence threshold for matches (lowered for better matching)
        """
        self.color_extractor = ColorHistogramExtractor()
        self.edge_extractor = EdgeFeatureExtractor()
        self.texture_extractor = TextureFeatureExtractor()
        
        self.color_weight = color_weight
        self.edge_weight = edge_weight
        self.texture_weight = texture_weight
        self.threshold = threshold
        
        self.lost_items: Dict[str, LostItem] = {}
        self.match_history: List[MatchResult] = []
        
        logger.info(
            f"LostItemMatcher initialized: "
            f"color={color_weight}, edge={edge_weight}, texture={texture_weight}, "
            f"threshold={threshold}"
        )
    
    def add_lost_item(self, item_id: str, image_path: str, name: str, 
                      description: str) -> bool:
        """
        Register a lost item by uploading its image.
        Enhanced preprocessing for better matching.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return False
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return False
            
            # Enhanced preprocessing for better matching
            processed_images = self._preprocess_lost_item_image(image)
            
            # Extract features from multiple processed versions
            all_features = []
            for proc_img in processed_images:
                features = {
                    'color': self.color_extractor.extract(proc_img),
                    'edge': self.edge_extractor.extract(proc_img),
                    'texture': self.texture_extractor.extract(proc_img),
                }
                all_features.append(features)
            
            # Combine features (average of all versions)
            combined_features = {
                'color': np.mean([f['color'] for f in all_features], axis=0),
                'edge': np.mean([f['edge'] for f in all_features], axis=0),
                'texture': np.mean([f['texture'] for f in all_features], axis=0),
                'image_shape': image.shape,
                'original_path': str(image_path),
                'processed_versions': len(processed_images)
            }
            
            # Create lost item
            lost_item = LostItem(
                item_id=item_id,
                image_path=str(image_path),
                name=name,
                description=description,
                upload_time=datetime.now(),
                features=combined_features
            )
            
            self.lost_items[item_id] = lost_item
            logger.info(f"Added lost item with enhanced features: {lost_item}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding lost item {item_id}: {e}")
            return False
    
    def _preprocess_lost_item_image(self, image):
        """
        Preprocess lost item image in multiple ways for robust matching.
        """
        processed_images = []
        
        # Original resized image
        original = cv2.resize(image, (224, 224))
        processed_images.append(original)
        
        # Brightness adjusted versions
        for gamma in [0.7, 1.0, 1.3]:
            gamma_corrected = self._adjust_gamma(original, gamma)
            processed_images.append(gamma_corrected)
        
        # Contrast enhanced version
        lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        processed_images.append(enhanced)
        
        # Slightly blurred version (for noisy detections)
        blurred = cv2.GaussianBlur(original, (3, 3), 0)
        processed_images.append(blurred)
        
        return processed_images
    
    def _adjust_gamma(self, image, gamma=1.0):
        """Adjust image gamma for brightness correction."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def match_detection(self, detection_image: np.ndarray, detection_id: str,
                       camera_id: str, bbox: Tuple, frame_number: int,
                       timestamp: float) -> List[MatchResult]:
        """
        Match a detected object against all registered lost items.
        Enhanced with multi-scale and multi-processing matching.
        """
        matches = []
        
        if len(self.lost_items) == 0:
            return matches
        
        if detection_image is None or detection_image.size == 0:
            return matches
        
        try:
            # Preprocess detection image in multiple ways
            processed_detections = self._preprocess_detection_image(detection_image)
            
            # Compare against all lost items
            for item_id, lost_item in self.lost_items.items():
                item_features = lost_item.features
                
                best_confidence = 0.0
                best_reasons = []
                
                # Try matching with each processed version
                for det_img in processed_detections:
                    # Extract features from detection
                    det_features = {
                        'color': self.color_extractor.extract(det_img),
                        'edge': self.edge_extractor.extract(det_img),
                        'texture': self.texture_extractor.extract(det_img)
                    }
                    
                    # Calculate similarity scores
                    color_score = self.color_extractor.compare(
                        det_features['color'], item_features['color']
                    )
                    edge_score = self.edge_extractor.compare(
                        det_features['edge'], item_features['edge']
                    )
                    texture_score = self.texture_extractor.compare(
                        det_features['texture'], item_features['texture']
                    )
                    
                    # Weighted combination
                    confidence = (
                        self.color_weight * color_score +
                        self.edge_weight * edge_score +
                        self.texture_weight * texture_score
                    )
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_reasons = []
                        if color_score > 0.6:
                            best_reasons.append(f"color match ({color_score:.2f})")
                        if edge_score > 0.6:
                            best_reasons.append(f"shape match ({edge_score:.2f})")
                        if texture_score > 0.6:
                            best_reasons.append(f"texture match ({texture_score:.2f})")
                
                # Create match if confidence exceeds threshold
                if best_confidence >= self.threshold:
                    match = MatchResult(
                        lost_item_id=item_id,
                        detection_id=detection_id,
                        camera_id=camera_id,
                        confidence=best_confidence,
                        frame_number=frame_number,
                        timestamp=timestamp,
                        bbox=bbox,
                        match_reasons=best_reasons
                    )
                    
                    matches.append(match)
                    self.match_history.append(match)
                    
                    logger.info(f"Enhanced match found: {item_id} with confidence {best_confidence:.2f}")
        
        except Exception as e:
            logger.error(f"Error matching detection {detection_id}: {e}")
        
        return sorted(matches, key=lambda m: m.confidence, reverse=True)
    
    def _preprocess_detection_image(self, detection_image):
        """
        Preprocess detection image in multiple ways for robust matching.
        """
        processed_images = []
        
        # Resize to standard size
        resized = cv2.resize(detection_image, (224, 224))
        processed_images.append(resized)
        
        # Brightness adjusted versions
        for gamma in [0.8, 1.0, 1.2]:
            gamma_corrected = self._adjust_gamma(resized, gamma)
            processed_images.append(gamma_corrected)
        
        # Contrast enhanced version
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        processed_images.append(enhanced)
        
        # Denoised version
        denoised = cv2.fastNlMeansDenoisingColored(resized, None, 10, 10, 7, 21)
        processed_images.append(denoised)
        
        return processed_images
    
    def get_lost_items(self) -> List[LostItem]:
        """Get all registered lost items."""
        return list(self.lost_items.values())
    
    def remove_lost_item(self, item_id: str) -> bool:
        """Remove a lost item (when found)."""
        if item_id in self.lost_items:
            del self.lost_items[item_id]
            logger.info(f"Removed lost item: {item_id}")
            return True
        return False
    
    def get_match_history(self, item_id: Optional[str] = None) -> List[MatchResult]:
        """Get matching history, optionally filtered by lost item ID."""
        if item_id:
            return [m for m in self.match_history if m.lost_item_id == item_id]
        return self.match_history
    
    def get_statistics(self) -> Dict:
        """Get matching statistics."""
        return {
            "lost_items_registered": len(self.lost_items),
            "total_matches_found": len(self.match_history),
            "items_matched": len(set(m.lost_item_id for m in self.match_history)),
            "avg_confidence": np.mean([m.confidence for m in self.match_history])
            if self.match_history else 0.0
        }
