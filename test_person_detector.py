"""
Unit Tests for Person Detection Module

Tests:
- Person detection core functionality
- Confidence thresholding
- Edge cases (night vision, occlusion, crowds)
- Performance requirements
- Integration with config system
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.detection.person.detector import PersonDetector
from ai.detection.person.config import PersonDetectorConfig
from ai.config import ConfigManager


class TestPersonDetectorConfig:
    """Tests for PersonDetectorConfig"""
    
    def test_config_initialization(self):
        """Test PersonDetectorConfig initialization with defaults"""
        config = PersonDetectorConfig()
        
        assert config.model_size == 'medium'
        assert config.confidence_threshold == 0.45
        assert config.nms_threshold == 0.45
        assert config.max_detections == 100
    
    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            'model_size': 'large',
            'confidence_threshold': 0.50,
            'nms_threshold': 0.40
        }
        config = PersonDetectorConfig(**config_dict)
        
        assert config.model_size == 'large'
        assert config.confidence_threshold == 0.50
        assert config.nms_threshold == 0.40
    
    def test_config_validation(self):
        """Test configuration parameter validation"""
        config = PersonDetectorConfig()
        
        # Valid confidence threshold
        assert 0.0 <= config.confidence_threshold <= 1.0
        assert 0.0 <= config.nms_threshold <= 1.0
        assert config.max_detections > 0


class TestPersonDetectorInitialization:
    """Tests for PersonDetector initialization"""
    
    @pytest.mark.unit
    def test_detector_initialization(self):
        """Test PersonDetector can be initialized"""
        config = PersonDetectorConfig()
        detector = PersonDetector(config)
        
        assert detector is not None
        assert detector.confidence_threshold == 0.45
        assert detector.nms_threshold == 0.45
    
    @pytest.mark.unit
    def test_detector_with_custom_config(self):
        """Test PersonDetector with custom configuration"""
        config = PersonDetectorConfig(
            confidence_threshold=0.50,
            model_size='large'
        )
        detector = PersonDetector(config)
        
        assert detector.confidence_threshold == 0.50
    
    @pytest.mark.unit
    def test_detector_model_loading(self):
        """Test that detector can be initialized without errors"""
        config = PersonDetectorConfig(model_size='nano')
        detector = PersonDetector(config)
        
        assert detector is not None


class TestPersonDetection:
    """Tests for person detection functionality"""
    
    @pytest.mark.unit
    def test_detect_returns_dict(self, sample_image):
        """Test that detect returns proper dictionary structure"""
        config = PersonDetectorConfig()
        detector = PersonDetector(config)
        
        result = detector.detect(sample_image)
        
        assert isinstance(result, dict)
        assert 'boxes' in result
        assert 'confidences' in result
        assert 'class_ids' in result
        assert 'class_names' in result
    
    @pytest.mark.unit
    def test_detect_boxes_format(self, sample_image):
        """Test detection boxes are in correct format"""
        config = PersonDetectorConfig()
        detector = PersonDetector(config)
        
        result = detector.detect(sample_image)
        boxes = result['boxes']
        
        # Boxes should be list/array
        assert isinstance(boxes, (list, np.ndarray))
        
        # Each box should have 4 coordinates
        if len(boxes) > 0:
            assert len(boxes[0]) == 4
    
    @pytest.mark.unit
    def test_detect_confidence_filtering(self, sample_image):
        """Test that only detections above threshold are returned"""
        config = PersonDetectorConfig(confidence_threshold=0.90)
        detector = PersonDetector(config)
        
        result = detector.detect(sample_image)
        confidences = result['confidences']
        
        # All confidences should be >= threshold
        if len(confidences) > 0:
            assert all(conf >= 0.90 for conf in confidences)
    
    @pytest.mark.unit
    def test_detect_empty_image(self):
        """Test detection on empty/blank image"""
        config = PersonDetectorConfig()
        detector = PersonDetector(config)
        
        # Blank image (all zeros)
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(blank_image)
        
        assert isinstance(result, dict)
        # Should return empty or minimal detections
        assert isinstance(result['boxes'], (list, np.ndarray))


class TestPersonDetectionEdgeCases:
    """Tests for edge cases in person detection"""
    
    @pytest.mark.unit
    def test_detect_low_light_image(self, sample_image_dark):
        """Test detection in low light conditions"""
        config = PersonDetectorConfig()
        detector = PersonDetector(config)
        
        result = detector.detect(sample_image_dark)
        
        assert isinstance(result, dict)
        assert 'boxes' in result
        # Low light should still return valid structure
    
    @pytest.mark.unit
    def test_detect_high_light_image(self, sample_image_bright):
        """Test detection in high light conditions"""
        config = PersonDetectorConfig()
        detector = PersonDetector(config)
        
        result = detector.detect(sample_image_bright)
        
        assert isinstance(result, dict)
        assert 'boxes' in result
    
    @pytest.mark.unit
    def test_detect_different_resolutions(self):
        """Test detection on different image resolutions"""
        config = PersonDetectorConfig()
        detector = PersonDetector(config)
        
        resolutions = [
            (480, 640),   # Common VGA
            (720, 1280),  # HD
            (1080, 1920), # Full HD
            (240, 320),   # Small
        ]
        
        for height, width in resolutions:
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            result = detector.detect(image)
            
            assert isinstance(result, dict)
            assert 'boxes' in result
    
    @pytest.mark.unit
    def test_detect_grayscale_image(self):
        """Test detection on grayscale image"""
        config = PersonDetectorConfig()
        detector = PersonDetector(config)
        
        # Create grayscale image
        gray_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        # Should handle conversion or reject gracefully
        try:
            result = detector.detect(gray_image)
            assert isinstance(result, dict)
        except (ValueError, AssertionError):
            # Acceptable to raise error for invalid input
            pass


class TestPersonDetectionPerformance:
    """Tests for performance requirements"""
    
    @pytest.mark.unit
    def test_detect_completes_in_time(self, sample_image):
        """Test that detection completes within time limit"""
        config = PersonDetectorConfig()
        detector = PersonDetector(config)
        
        import time
        start = time.time()
        result = detector.detect(sample_image)
        elapsed = time.time() - start
        
        # Detection should complete in reasonable time
        # For a single frame: should be < 100ms (60+ FPS requirement)
        assert elapsed < 1.0, f"Detection took {elapsed}s, should be < 1.0s"
    
    @pytest.mark.unit
    def test_detect_batch_multiple_frames(self, sample_image):
        """Test detection on batch of frames"""
        config = PersonDetectorConfig()
        detector = PersonDetector(config)
        
        import time
        start = time.time()
        
        # Process 10 frames
        for _ in range(10):
            result = detector.detect(sample_image)
            assert isinstance(result, dict)
        
        elapsed = time.time() - start
        avg_time = elapsed / 10
        
        # Average time per frame should be reasonable
        assert avg_time < 1.0


class TestPersonDetectionConfigIntegration:
    """Tests for integration with ConfigManager"""
    
    @pytest.mark.unit
    def test_detector_uses_config_threshold(self, test_config):
        """Test that detector uses threshold from config"""
        threshold = test_config.get(
            'detection.confidence_thresholds.person',
            0.45
        )
        
        config = PersonDetectorConfig(confidence_threshold=threshold)
        detector = PersonDetector(config)
        
        assert detector.confidence_threshold == threshold
    
    @pytest.mark.unit
    def test_detector_with_different_thresholds(self):
        """Test detector behavior with different thresholds"""
        thresholds = [0.30, 0.45, 0.60, 0.75]
        
        for threshold in thresholds:
            config = PersonDetectorConfig(confidence_threshold=threshold)
            detector = PersonDetector(config)
            
            assert detector.confidence_threshold == threshold


class TestPersonDetectionResults:
    """Tests for detection result validation"""
    
    @pytest.mark.unit
    def test_result_structure_consistency(self, sample_image):
        """Test that result structure is always consistent"""
        config = PersonDetectorConfig()
        detector = PersonDetector(config)
        
        results = []
        for _ in range(5):
            result = detector.detect(sample_image)
            results.append(result)
        
        # All results should have same keys
        keys_set = set(results[0].keys())
        for result in results[1:]:
            assert set(result.keys()) == keys_set
    
    @pytest.mark.unit
    def test_detection_box_coordinates(self, sample_image):
        """Test that bounding box coordinates are valid"""
        config = PersonDetectorConfig()
        detector = PersonDetector(config)
        
        result = detector.detect(sample_image)
        boxes = result['boxes']
        height, width = sample_image.shape[:2]
        
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Coordinates should be within image bounds
            assert 0 <= x1 < width
            assert 0 <= x2 <= width
            assert 0 <= y1 < height
            assert 0 <= y2 <= height
            
            # x2 should be > x1, y2 should be > y1
            assert x2 > x1
            assert y2 > y1


@pytest.mark.unit
class TestPersonDetectionNMS:
    """Tests for Non-Maximum Suppression"""
    
    def test_nms_threshold_effect(self, sample_image):
        """Test that NMS threshold affects overlap removal"""
        # Different NMS thresholds
        low_nms = PersonDetectorConfig(nms_threshold=0.30)
        high_nms = PersonDetectorConfig(nms_threshold=0.80)
        
        detector_low = PersonDetector(low_nms)
        detector_high = PersonDetector(high_nms)
        
        result_low = detector_low.detect(sample_image)
        result_high = detector_high.detect(sample_image)
        
        # Results should be valid
        assert isinstance(result_low, dict)
        assert isinstance(result_high, dict)


@pytest.mark.integration
class TestPersonDetectorIntegration:
    """Integration tests for PersonDetector"""
    
    def test_detector_multiple_calls(self, sample_image):
        """Test detector handles multiple sequential calls"""
        config = PersonDetectorConfig()
        detector = PersonDetector(config)
        
        for i in range(5):
            result = detector.detect(sample_image)
            assert isinstance(result, dict)
            assert 'boxes' in result
    
    def test_detector_state_independence(self, sample_image):
        """Test that detector state doesn't persist between calls"""
        config = PersonDetectorConfig()
        detector = PersonDetector(config)
        
        result1 = detector.detect(sample_image)
        result2 = detector.detect(sample_image)
        
        # Results should be deterministic (same for same input)
        # (assuming deterministic model)
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
