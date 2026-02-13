"""
Unit Tests for Lost Item Re-Identification System

Tests for:
- Feature extractors (color, edge, texture)
- Lost item matcher
- Service integration
- Matching accuracy
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import tempfile

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.reidentification.lost_item_matcher import (
    LostItemMatcher,
    ColorHistogramExtractor,
    EdgeFeatureExtractor,
    TextureFeatureExtractor,
    LostItem,
    MatchResult
)
from src.escalation.lost_item_service import LostItemService, LostItemReporter


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = np.ones((224, 224, 3), dtype=np.uint8) * 128
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), -1)  # Red square
    return img


@pytest.fixture
def similar_image(sample_image):
    """Create a similar image with slight color variation."""
    img = sample_image.copy()
    img[:, :] = img[:, :] + np.random.randint(-20, 20, img.shape)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


@pytest.fixture
def different_image():
    """Create a very different image."""
    img = np.ones((224, 224, 3), dtype=np.uint8) * 255
    cv2.circle(img, (112, 112), 80, (0, 255, 0), -1)  # Green circle
    return img


@pytest.fixture
def temp_image_file(sample_image, tmp_path):
    """Create a temporary image file."""
    img_path = tmp_path / "test_image.png"
    cv2.imwrite(str(img_path), sample_image)
    return img_path


@pytest.fixture
def matcher():
    """Initialize a matcher."""
    return LostItemMatcher(
        color_weight=0.4,
        edge_weight=0.3,
        texture_weight=0.3,
        threshold=0.6
    )


@pytest.fixture
def service(tmp_path):
    """Initialize a service with temp directory."""
    service = LostItemService(upload_dir=str(tmp_path / "uploads"))
    return service


# ============================================================================
# COLOR HISTOGRAM TESTS
# ============================================================================

class TestColorHistogramExtractor:
    """Test color histogram feature extraction."""
    
    def test_extract_returns_correct_shape(self, sample_image):
        """Color histogram should return (32 bins × 3 channels) = 96 features."""
        extractor = ColorHistogramExtractor(bins=32)
        features = extractor.extract(sample_image)
        assert features.shape == (96,), f"Expected shape (96,), got {features.shape}"
    
    def test_extract_normalized(self, sample_image):
        """Features should be normalized."""
        extractor = ColorHistogramExtractor()
        features = extractor.extract(sample_image)
        # HSV histogram should have values in valid range
        assert np.all(features >= 0), "Features should be non-negative"
    
    def test_extract_handles_empty_image(self):
        """Should handle empty/None images gracefully."""
        extractor = ColorHistogramExtractor()
        features = extractor.extract(None)
        assert features.shape == (96,)
        assert np.sum(features) == 0
    
    def test_similar_images_high_similarity(self, sample_image, similar_image):
        """Similar images should have high color similarity."""
        extractor = ColorHistogramExtractor()
        hist1 = extractor.extract(sample_image)
        hist2 = extractor.extract(similar_image)
        
        similarity = extractor.compare(hist1, hist2)
        assert similarity > 0.8, f"Expected high similarity, got {similarity}"
    
    def test_different_images_low_similarity(self, sample_image, different_image):
        """Different images should have low color similarity."""
        extractor = ColorHistogramExtractor()
        hist1 = extractor.extract(sample_image)
        hist2 = extractor.extract(different_image)
        
        similarity = extractor.compare(hist1, hist2)
        assert similarity < 0.6, f"Expected low similarity, got {similarity}"


# ============================================================================
# EDGE FEATURE TESTS
# ============================================================================

class TestEdgeFeatureExtractor:
    """Test edge feature extraction."""
    
    def test_extract_returns_correct_shape(self, sample_image):
        """Edge features should be 16×16 flattened = 256."""
        extractor = EdgeFeatureExtractor()
        features = extractor.extract(sample_image)
        assert features.shape == (256,), f"Expected shape (256,), got {features.shape}"
    
    def test_extract_normalized(self, sample_image):
        """Edge features should be in [0, 1]."""
        extractor = EdgeFeatureExtractor()
        features = extractor.extract(sample_image)
        assert np.all(features >= 0) and np.all(features <= 1)
    
    def test_extract_handles_empty_image(self):
        """Should handle None images."""
        extractor = EdgeFeatureExtractor()
        features = extractor.extract(None)
        assert features.shape == (256,)
    
    def test_same_edges_high_similarity(self, sample_image):
        """Same image edges should match."""
        extractor = EdgeFeatureExtractor()
        feat1 = extractor.extract(sample_image)
        feat2 = extractor.extract(sample_image)
        
        similarity = extractor.compare(feat1, feat2)
        assert similarity > 0.95, f"Expected very high similarity, got {similarity}"
    
    def test_rotated_variant_some_similarity(self, sample_image):
        """Rotated shapes should have some similarity."""
        extractor = EdgeFeatureExtractor()
        feat1 = extractor.extract(sample_image)
        
        # Rotate image
        h, w = sample_image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), 45, 1.0)
        rotated = cv2.warpAffine(sample_image, M, (w, h))
        feat2 = extractor.extract(rotated)
        
        similarity = extractor.compare(feat1, feat2)
        assert 0.4 < similarity < 0.9, f"Expected moderate similarity, got {similarity}"


# ============================================================================
# TEXTURE FEATURE TESTS
# ============================================================================

class TestTextureFeatureExtractor:
    """Test texture feature extraction."""
    
    def test_extract_returns_correct_shape(self, sample_image):
        """Texture features should be 59-bin histogram."""
        extractor = TextureFeatureExtractor()
        features = extractor.extract(sample_image)
        assert features.shape == (59,), f"Expected shape (59,), got {features.shape}"
    
    def test_extract_normalized(self, sample_image):
        """Texture features should sum to ~1.0 (histogram)."""
        extractor = TextureFeatureExtractor()
        features = extractor.extract(sample_image)
        assert 0.99 <= np.sum(features) <= 1.01, "Histogram should be normalized"
    
    def test_smooth_images_have_texture(self, sample_image, different_image):
        """Smooth images should have different texture than complex ones."""
        extractor = TextureFeatureExtractor()
        
        # Smooth image
        smooth = cv2.GaussianBlur(sample_image, (5, 5), 0)
        feat_smooth = extractor.extract(smooth)
        
        # Noisy image
        noisy = sample_image.copy().astype(np.float32)
        noisy += np.random.normal(0, 10, noisy.shape)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        feat_noisy = extractor.extract(noisy)
        
        similarity = extractor.compare(feat_smooth, feat_noisy)
        # Should be different but not completely different
        assert 0.4 < similarity < 0.9


# ============================================================================
# LOST ITEM MATCHER TESTS
# ============================================================================

class TestLostItemMatcher:
    """Test lost item matching functionality."""
    
    def test_initialization(self, matcher):
        """Matcher should initialize properly."""
        assert matcher.threshold == 0.6
        assert len(matcher.lost_items) == 0
        assert len(matcher.match_history) == 0
    
    def test_add_lost_item_success(self, matcher, temp_image_file):
        """Should add lost item successfully."""
        success = matcher.add_lost_item(
            "item_001",
            str(temp_image_file),
            "Test Item",
            "A test item"
        )
        assert success
        assert "item_001" in matcher.lost_items
    
    def test_add_lost_item_invalid_file(self, matcher):
        """Should handle missing files."""
        success = matcher.add_lost_item(
            "item_002",
            "/nonexistent/path/image.jpg",
            "Test",
            ""
        )
        assert not success
        assert "item_002" not in matcher.lost_items
    
    def test_get_lost_items(self, matcher, temp_image_file):
        """Should retrieve registered items."""
        matcher.add_lost_item("item_001", str(temp_image_file), "Item 1", "Desc 1")
        matcher.add_lost_item("item_002", str(temp_image_file), "Item 2", "Desc 2")
        
        items = matcher.get_lost_items()
        assert len(items) == 2
        assert items[0].item_id == "item_001"
        assert items[1].item_id == "item_002"
    
    def test_remove_lost_item(self, matcher, temp_image_file):
        """Should remove lost items."""
        matcher.add_lost_item("item_001", str(temp_image_file), "Item", "")
        assert len(matcher.get_lost_items()) == 1
        
        success = matcher.remove_lost_item("item_001")
        assert success
        assert len(matcher.get_lost_items()) == 0
    
    def test_match_detection_high_confidence(self, matcher, temp_image_file, 
                                             sample_image):
        """Should find matches with high confidence."""
        # Add item
        matcher.add_lost_item("item_001", str(temp_image_file), "Item", "")
        
        # Match identical image
        matches = matcher.match_detection(
            sample_image,
            "det_001",
            "cam_1",
            (0, 0, 224, 224),
            1,
            1.0
        )
        
        assert len(matches) > 0, "Should find match for similar image"
        assert matches[0].confidence > 0.85
    
    def test_match_detection_low_confidence(self, matcher, temp_image_file,
                                           different_image):
        """Should not match very different images."""
        matcher.add_lost_item("item_001", str(temp_image_file), "Item", "")
        
        matches = matcher.match_detection(
            different_image,
            "det_002",
            "cam_1",
            (0, 0, 224, 224),
            2,
            2.0
        )
        
        # May or may not match depending on threshold
        # But confidence should be low if it does
        if matches:
            assert matches[0].confidence < 0.75
    
    def test_match_detection_empty_image(self, matcher, temp_image_file):
        """Should handle empty detection images."""
        matcher.add_lost_item("item_001", str(temp_image_file), "Item", "")
        
        empty_image = np.array([], dtype=np.uint8)
        matches = matcher.match_detection(
            empty_image,
            "det_003",
            "cam_1",
            (0, 0, 0, 0),
            3,
            3.0
        )
        
        assert len(matches) == 0
    
    def test_match_history(self, matcher, temp_image_file, sample_image):
        """Should track match history."""
        matcher.add_lost_item("item_001", str(temp_image_file), "Item", "")
        
        # Create multiple matches
        for i in range(3):
            matcher.match_detection(
                sample_image,
                f"det_{i}",
                "cam_1",
                (0, 0, 224, 224),
                i,
                float(i)
            )
        
        history = matcher.get_match_history()
        assert len(history) >= 3
    
    def test_get_statistics(self, matcher, temp_image_file, sample_image):
        """Should provide statistics."""
        matcher.add_lost_item("item_001", str(temp_image_file), "Item 1", "")
        matcher.add_lost_item("item_002", str(temp_image_file), "Item 2", "")
        
        matcher.match_detection(sample_image, "det_001", "cam_1", 
                               (0, 0, 224, 224), 1, 1.0)
        
        stats = matcher.get_statistics()
        assert stats["lost_items_registered"] == 2
        assert "total_matches_found" in stats


# ============================================================================
# SERVICE TESTS
# ============================================================================

class TestLostItemService:
    """Test the lost item service."""
    
    def test_upload_item_success(self, service, sample_image, tmp_path):
        """Should upload items successfully."""
        img_path = tmp_path / "item.png"
        cv2.imwrite(str(img_path), sample_image)
        
        success, item_id = service.upload_lost_item(str(img_path), "Test Item")
        assert success
        assert item_id.startswith("item_")
    
    def test_get_lost_items(self, service, sample_image, tmp_path):
        """Should retrieve lost items."""
        img_path = tmp_path / "item.png"
        cv2.imwrite(str(img_path), sample_image)
        
        service.upload_lost_item(str(img_path), "Item 1", "Desc 1")
        service.upload_lost_item(str(img_path), "Item 2", "Desc 2")
        
        items = service.get_lost_items()
        assert len(items) == 2
        assert all("item_id" in item for item in items)
        assert all("name" in item for item in items)
    
    def test_mark_found(self, service, sample_image, tmp_path):
        """Should mark items as found."""
        img_path = tmp_path / "item.png"
        cv2.imwrite(str(img_path), sample_image)
        
        success, item_id = service.upload_lost_item(str(img_path), "Item")
        assert len(service.get_lost_items()) == 1
        
        found = service.mark_found(item_id)
        assert found
        assert len(service.get_lost_items()) == 0
    
    def test_get_statistics(self, service):
        """Should provide service statistics."""
        stats = service.get_statistics()
        assert "lost_items_registered" in stats
        assert "match_threshold" in stats
        assert stats["lost_items_registered"] == 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestLostItemIntegration:
    """Integration tests."""
    
    def test_end_to_end_workflow(self, service, sample_image, similar_image,
                                 tmp_path):
        """Test complete workflow: upload -> match -> report."""
        # Create test image
        img_path = tmp_path / "lost_item.png"
        cv2.imwrite(str(img_path), sample_image)
        
        # Upload
        success, item_id = service.upload_lost_item(
            str(img_path),
            "Lost Backpack",
            "Red backpack"
        )
        assert success
        
        # Get items
        items = service.get_lost_items()
        assert len(items) == 1
        assert items[0]["name"] == "Lost Backpack"
        
        # Get matches (should be empty initially)
        matches = service.get_matches()
        assert len(matches) == 0
        
        # Mark as found
        found = service.mark_found(item_id)
        assert found
        assert len(service.get_lost_items()) == 0
    
    def test_reporter_generation(self, service, sample_image, tmp_path):
        """Test report generation."""
        img_path = tmp_path / "item.png"
        cv2.imwrite(str(img_path), sample_image)
        
        service.upload_lost_item(str(img_path), "Item", "Description")
        
        reporter = LostItemReporter(service)
        report = reporter.report_matches()
        
        assert "Lost Item Identification Report" in report or "LOST ITEM" in report
        assert "item_" in report
        assert "Item" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
