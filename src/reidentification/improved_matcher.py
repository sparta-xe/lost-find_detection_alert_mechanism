"""
Improved Lost Item Matcher with Screenshot-to-Video Matching
This enhanced matcher is specifically designed to handle cases where:
- Users upload screenshots from the same video
- Identical objects need to be matched across different frames
- Lighting and compression differences exist
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MatchResult:
    """Enhanced match result with detailed scoring."""
    lost_item_id: str
    detection_id: str
    camera_id: str
    confidence: float
    frame_number: int
    timestamp: float
    bbox: Tuple[int, int, int, int]
    match_reasons: List[str]
    feature_scores: Dict[str, float]
    preprocessing_method: str

@dataclass
class LostItem:
    """Lost item data structure."""
    item_id: str
    name: str
    description: str
    image_path: str
    upload_time: datetime
    features: Dict

class ImprovedLostItemMatcher:
    """
    Enhanced matcher specifically designed for screenshot-to-video matching.
    Uses multiple feature extraction methods and robust preprocessing.
    """
    def __init__(self, threshold: float = 0.25):
        """
        Initialize with very low threshold for screenshot matching.
        Args:
            threshold: Base matching threshold (lowered for screenshot matching)
        """
        self.threshold = threshold
        self.lost_items: Dict[str, LostItem] = {}
        self.match_history: List[MatchResult] = []
        
        # Initialize feature extractors
        try:
            self.sift = cv2.SIFT_create(nfeatures=500)
            self.orb = cv2.ORB_create(nfeatures=500)
        except Exception as e:
            logger.warning(f"Could not initialize SIFT/ORB: {e}")
            self.sift = None
            self.orb = None
        
        logger.info(f"Improved matcher initialized with threshold {threshold}")
    
    def add_lost_item(self, item_id: str, image_path: str, name: str, 
                      description: str = "") -> bool:
        """
        Add lost item with comprehensive feature extraction.
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                return False
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return False
            
            # Extract comprehensive features
            features = self._extract_comprehensive_features(image)
            
            lost_item = LostItem(
                item_id=item_id,
                name=name,
                description=description,
                image_path=str(image_path),
                upload_time=datetime.now(),
                features=features
            )
            
            self.lost_items[item_id] = lost_item
            logger.info(f"Added lost item: {name} with {len(features)} feature sets")
            return True
            
        except Exception as e:
            logger.error(f"Error adding lost item {item_id}: {e}")
            return False
    
    def _extract_comprehensive_features(self, image: np.ndarray) -> Dict:
        """
        Extract multiple types of features for robust matching.
        """
        features = {}
        
        # Resize to standard size for consistent comparison
        std_image = cv2.resize(image, (224, 224))
        gray = cv2.cvtColor(std_image, cv2.COLOR_BGR2GRAY)
        
        # 1. Color histogram (multiple color spaces)
        features['color_bgr'] = self._extract_color_histogram(std_image, None)  # No conversion for BGR
        features['color_hsv'] = self._extract_color_histogram(std_image, cv2.COLOR_BGR2HSV)
        features['color_lab'] = self._extract_color_histogram(std_image, cv2.COLOR_BGR2LAB)
        
        # 2. Texture features
        features['lbp'] = self._extract_lbp_features(gray)
        
        # 3. Shape features
        features['hu_moments'] = self._extract_hu_moments(gray)
        
        # 4. Keypoint features (if available)
        if self.sift:
            features['sift_desc'] = self._extract_sift_features(gray)
        if self.orb:
            features['orb_desc'] = self._extract_orb_features(gray)
        
        # 5. Edge features
        features['edge_histogram'] = self._extract_edge_features(gray)
        
        # 6. Template matching features (for exact matching)
        features['template'] = std_image.copy()
        features['template_gray'] = gray.copy()
        
        # 7. Multi-scale templates for better matching
        features['templates_multiscale'] = self._create_multiscale_templates(std_image)
        
        return features
    
    def _extract_color_histogram(self, image: np.ndarray, color_space: Optional[int]) -> np.ndarray:
        """Extract color histogram in specified color space."""
        if color_space is not None:
            image = cv2.cvtColor(image, color_space)
        
        # Calculate histogram for each channel
        hist_0 = cv2.calcHist([image], [0], None, [32], [0, 256])
        hist_1 = cv2.calcHist([image], [1], None, [32], [0, 256])
        hist_2 = cv2.calcHist([image], [2], None, [32], [0, 256])
        
        # Normalize and concatenate
        hist = np.concatenate([hist_0.flatten(), hist_1.flatten(), hist_2.flatten()])
        return hist / (np.sum(hist) + 1e-7)
    
    def _extract_lbp_features(self, gray: np.ndarray) -> np.ndarray:
        """Extract Local Binary Pattern features."""
        # Simple LBP implementation
        lbp = np.zeros_like(gray)
        for i in range(1, gray.shape[0]-1):
            for j in range(1, gray.shape[1]-1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] >= center) << 7
                code |= (gray[i-1, j] >= center) << 6
                code |= (gray[i-1, j+1] >= center) << 5
                code |= (gray[i, j+1] >= center) << 4
                code |= (gray[i+1, j+1] >= center) << 3
                code |= (gray[i+1, j] >= center) << 2
                code |= (gray[i+1, j-1] >= center) << 1
                code |= (gray[i, j-1] >= center) << 0
                lbp[i, j] = code
        
        # Calculate histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        return hist / (np.sum(hist) + 1e-7)
    
    def _extract_hu_moments(self, gray: np.ndarray) -> np.ndarray:
        """Extract Hu moments for shape description."""
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        # Log transform to make them scale invariant
        return -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-7)
    
    def _extract_sift_features(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Extract SIFT keypoint descriptors."""
        if not self.sift:
            return None
        try:
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            if descriptors is not None and len(descriptors) > 0:
                # Return mean descriptor as feature vector
                return np.mean(descriptors, axis=0)
        except Exception as e:
            logger.debug(f"SIFT extraction failed: {e}")
        return None
    
    def _extract_orb_features(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Extract ORB keypoint descriptors."""
        if not self.orb:
            return None
        try:
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            if descriptors is not None and len(descriptors) > 0:
                # Return mean descriptor as feature vector
                return np.mean(descriptors.astype(np.float32), axis=0)
        except Exception as e:
            logger.debug(f"ORB extraction failed: {e}")
        return None
    
    def _extract_edge_features(self, gray: np.ndarray) -> np.ndarray:
        """Extract edge-based features."""
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        # Edge histogram
        hist, _ = np.histogram(edges.ravel(), bins=32, range=(0, 256))
        return hist / (np.sum(hist) + 1e-7)
    
    def _create_multiscale_templates(self, image: np.ndarray) -> List[np.ndarray]:
        """Create templates at multiple scales for better matching."""
        templates = []
        scales = [0.8, 1.0, 1.2]  # Different scales
        
        for scale in scales:
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            if new_h > 0 and new_w > 0:
                scaled = cv2.resize(image, (new_w, new_h))
                templates.append(scaled)
        
        return templates
    
    def match_detection(self, detection_image: np.ndarray, detection_id: str,
                       camera_id: str, bbox: Tuple, frame_number: int,
                       timestamp: float) -> List[MatchResult]:
        """
        Match detection against lost items using comprehensive feature matching.
        """
        matches = []
        if len(self.lost_items) == 0 or detection_image is None or detection_image.size == 0:
            return matches
        
        try:
            # Extract features from detection
            det_features = self._extract_comprehensive_features(detection_image)
            
            # Compare against all lost items
            for item_id, lost_item in self.lost_items.items():
                item_features = lost_item.features
                
                # Calculate multiple similarity scores
                scores = self._calculate_similarity_scores(det_features, item_features)
                
                # Try template matching for exact matches (screenshot case)
                template_score = self._enhanced_template_matching(
                    detection_image, item_features
                )
                scores['template_match'] = template_score
                
                # Calculate weighted final score
                final_score = self._calculate_final_score(scores)
                
                # Create match if above threshold
                if final_score >= self.threshold:
                    # Determine match reasons
                    reasons = []
                    if scores.get('color_similarity', 0) > 0.6:
                        reasons.append(f"color match ({scores['color_similarity']:.2f})")
                    if scores.get('shape_similarity', 0) > 0.6:
                        reasons.append(f"shape match ({scores['shape_similarity']:.2f})")
                    if scores.get('texture_similarity', 0) > 0.6:
                        reasons.append(f"texture match ({scores['texture_similarity']:.2f})")
                    if scores.get('template_match', 0) > 0.5:
                        reasons.append(f"template match ({scores['template_match']:.2f})")
                    if scores.get('keypoint_similarity', 0) > 0.5:
                        reasons.append(f"keypoint match ({scores['keypoint_similarity']:.2f})")
                    
                    match = MatchResult(
                        lost_item_id=item_id,
                        detection_id=detection_id,
                        camera_id=camera_id,
                        confidence=final_score,
                        frame_number=frame_number,
                        timestamp=timestamp,
                        bbox=bbox,
                        match_reasons=reasons,
                        feature_scores=scores,
                        preprocessing_method="comprehensive"
                    )
                    matches.append(match)
                    self.match_history.append(match)
                    logger.info(f"Match found: {lost_item.name} with confidence {final_score:.2f}")
        
        except Exception as e:
            logger.error(f"Error matching detection {detection_id}: {e}")
        
        return sorted(matches, key=lambda m: m.confidence, reverse=True)
    
    def _enhanced_template_matching(self, detection_image: np.ndarray, 
                                  item_features: Dict) -> float:
        """Enhanced template matching with multiple scales and methods."""
        max_score = 0.0
        
        try:
            # Resize detection to standard size
            det_resized = cv2.resize(detection_image, (224, 224))
            det_gray = cv2.cvtColor(det_resized, cv2.COLOR_BGR2GRAY)
            
            # Try matching with main template
            if 'template_gray' in item_features:
                template = item_features['template_gray']
                result = cv2.matchTemplate(det_gray, template, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result)
                max_score = max(max_score, score)
            
            # Try matching with multiscale templates
            if 'templates_multiscale' in item_features:
                for template in item_features['templates_multiscale']:
                    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    
                    # Resize template to match detection size
                    template_resized = cv2.resize(template_gray, (224, 224))
                    
                    # Multiple template matching methods
                    methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
                    
                    for method in methods:
                        result = cv2.matchTemplate(det_gray, template_resized, method)
                        _, score, _, _ = cv2.minMaxLoc(result)
                        max_score = max(max_score, score)
            
            # Try normalized cross-correlation
            if 'template' in item_features:
                template_color = cv2.resize(item_features['template'], (224, 224))
                
                # Normalize both images
                det_norm = cv2.normalize(det_resized.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
                template_norm = cv2.normalize(template_color.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
                
                # Calculate normalized cross-correlation
                correlation = cv2.matchTemplate(det_norm, template_norm, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(correlation)
                max_score = max(max_score, score)
            
        except Exception as e:
            logger.debug(f"Enhanced template matching failed: {e}")
        
        return max_score
    
    def _calculate_similarity_scores(self, det_features: Dict, item_features: Dict) -> Dict[str, float]:
        """Calculate similarity scores for all feature types."""
        scores = {}
        
        # Color similarity (average across color spaces)
        color_scores = []
        for color_space in ['color_bgr', 'color_hsv', 'color_lab']:
            if color_space in det_features and color_space in item_features:
                score = self._histogram_similarity(det_features[color_space], item_features[color_space])
                color_scores.append(score)
        scores['color_similarity'] = np.mean(color_scores) if color_scores else 0.0
        
        # Texture similarity
        if 'lbp' in det_features and 'lbp' in item_features:
            scores['texture_similarity'] = self._histogram_similarity(det_features['lbp'], item_features['lbp'])
        else:
            scores['texture_similarity'] = 0.0
        
        # Shape similarity
        if 'hu_moments' in det_features and 'hu_moments' in item_features:
            scores['shape_similarity'] = self._vector_similarity(det_features['hu_moments'], item_features['hu_moments'])
        else:
            scores['shape_similarity'] = 0.0
        
        # Keypoint similarity
        keypoint_scores = []
        for kp_type in ['sift_desc', 'orb_desc']:
            if (kp_type in det_features and kp_type in item_features and 
                det_features[kp_type] is not None and item_features[kp_type] is not None):
                score = self._vector_similarity(det_features[kp_type], item_features[kp_type])
                keypoint_scores.append(score)
        scores['keypoint_similarity'] = np.mean(keypoint_scores) if keypoint_scores else 0.0
        
        # Edge similarity
        if 'edge_histogram' in det_features and 'edge_histogram' in item_features:
            scores['edge_similarity'] = self._histogram_similarity(
                det_features['edge_histogram'], item_features['edge_histogram']
            )
        else:
            scores['edge_similarity'] = 0.0
        
        return scores
    
    def _histogram_similarity(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Calculate histogram similarity using multiple methods."""
        try:
            # Correlation
            corr = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
            
            # Chi-square (inverted and normalized)
            chi_sq = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CHISQR)
            chi_sq_norm = 1.0 / (1.0 + chi_sq)
            
            # Intersection
            intersection = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_INTERSECT)
            
            # Bhattacharyya (inverted)
            bhatta = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
            bhatta_norm = 1.0 - bhatta
            
            # Average of all methods
            return np.mean([corr, chi_sq_norm, intersection, bhatta_norm])
            
        except Exception as e:
            logger.debug(f"Histogram similarity failed: {e}")
            return 0.0
    
    def _vector_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate vector similarity using cosine similarity."""
        try:
            # Normalize vectors
            vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-7)
            vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-7)
            
            # Cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            
            # Convert to 0-1 range
            return (similarity + 1) / 2
            
        except Exception as e:
            logger.debug(f"Vector similarity failed: {e}")
            return 0.0
    
    def _calculate_final_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted final similarity score optimized for screenshot matching."""
        # Weights optimized for screenshot matching
        weights = {
            'template_match': 0.5,      # Very high weight for exact matching
            'color_similarity': 0.25,   # Important for visual similarity
            'keypoint_similarity': 0.1, # Good for distinctive features
            'shape_similarity': 0.08,   # Shape consistency
            'texture_similarity': 0.04, # Less important for screenshots
            'edge_similarity': 0.03     # Edge consistency
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for feature, weight in weights.items():
            if feature in scores:
                weighted_score += scores[feature] * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def remove_lost_item(self, item_id: str) -> bool:
        """Remove a lost item (mark as found)."""
        if item_id in self.lost_items:
            del self.lost_items[item_id]
            logger.info(f"Removed lost item: {item_id}")
            return True
        return False
    
    def get_lost_items(self) -> List[LostItem]:
        """Get list of registered lost items."""
        return list(self.lost_items.values())
    
    def get_match_history(self, item_id: Optional[str] = None) -> List[MatchResult]:
        """Get match history for specific item or all items."""
        if item_id:
            return [m for m in self.match_history if m.lost_item_id == item_id]
        return self.match_history
    
    def get_statistics(self) -> Dict:
        """Get matching statistics."""
        total_matches = len(self.match_history)
        items_matched = len(set(m.lost_item_id for m in self.match_history))
        avg_confidence = np.mean([m.confidence for m in self.match_history]) if self.match_history else 0.0
        
        return {
            'lost_items_registered': len(self.lost_items),
            'total_matches_found': total_matches,
            'items_matched': items_matched,
            'avg_confidence': avg_confidence,
            'threshold': self.threshold
        }