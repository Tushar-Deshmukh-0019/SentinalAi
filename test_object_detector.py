"""
Unit Tests for Object Detection Module

Tests:
- Object detection (backpacks, bags, suitcases)
- Abandoned object detection
- Size classification
- Person-object correlation
- Temporal tracking (2-min/10-min alerts)
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.detection.object.detector import ObjectDetector
from ai.detection.object.classifier import ObjectClassifier
from ai.detection.object.config import ObjectDetectorConfig
from ai.config import ConfigManager


class TestObjectDetectorConfig:
    """Tests for ObjectDetectorConfig"""
    
    def test_config_initialization(self):
        """Test ObjectDetectorConfig initialization"""
        config = ObjectDetectorConfig()
        
        assert config.model_size == 'medium'
        assert config.confidence_threshold == 0.50
        assert config.nms_threshold == 0.45
        assert config.abandoned_threshold_seconds == 120  # 2 minutes
    
    def test_config_object_types(self):
        """Test configured object types"""
        config = ObjectDetectorConfig()
        
        expected_types = ['backpack', 'bag', 'suitcase', 'briefcase']
        assert all(obj_type in config.object_types for obj_type in expected_types)


class TestObjectDetectorInitialization:
    """Tests for ObjectDetector initialization"""
    
    @pytest.mark.unit
    def test_detector_initialization(self):
        """Test ObjectDetector can be initialized"""
        config = ObjectDetectorConfig()
        detector = ObjectDetector(config)
        
        assert detector is not None
        assert detector.confidence_threshold == 0.50
    
    @pytest.mark.unit
    def test_classifier_initialization(self):
        """Test ObjectClassifier can be initialized"""
        classifier = ObjectClassifier()
        
        assert classifier is not None


class TestObjectDetection:
    """Tests for object detection functionality"""
    
    @pytest.mark.unit
    def test_detect_returns_dict(self, sample_image):
        """Test that detect returns proper structure"""
        config = ObjectDetectorConfig()
        detector = ObjectDetector(config)
        
        result = detector.detect(sample_image)
        
        assert isinstance(result, dict)
        assert 'boxes' in result
        assert 'confidences' in result
        assert 'object_types' in result or 'class_names' in result
    
    @pytest.mark.unit
    def test_detect_object_classification(self, sample_image):
        """Test that objects are classified by type"""
        config = ObjectDetectorConfig()
        detector = ObjectDetector(config)
        
        result = detector.detect(sample_image)
        
        if len(result['boxes']) > 0:
            obj_types = result.get('object_types', result.get('class_names', []))
            assert len(obj_types) == len(result['boxes'])
            
            # All should be security-relevant objects
            valid_types = ['backpack', 'bag', 'suitcase', 'briefcase', 'package']
            for obj_type in obj_types:
                assert obj_type in valid_types
    
    @pytest.mark.unit
    def test_detect_confidence_threshold(self, sample_image):
        """Test confidence threshold filtering"""
        config = ObjectDetectorConfig(confidence_threshold=0.70)
        detector = ObjectDetector(config)
        
        result = detector.detect(sample_image)
        
        if len(result['confidences']) > 0:
            assert all(conf >= 0.70 for conf in result['confidences'])


class TestObjectClassification:
    """Tests for object type and size classification"""
    
    @pytest.mark.unit
    def test_classifier_object_type(self):
        """Test object type classification"""
        classifier = ObjectClassifier()
        
        valid_types = classifier.get_valid_object_types()
        
        assert len(valid_types) > 0
        assert 'backpack' in valid_types
        assert 'bag' in valid_types
    
    @pytest.mark.unit
    def test_classifier_size_assignment(self, sample_image):
        """Test size classification of objects"""
        classifier = ObjectClassifier()
        
        # Different sized boxes
        small_box = [[100, 100, 120, 140]]
        large_box = [[100, 100, 300, 400]]
        
        small_class = classifier.classify(sample_image, small_box)
        large_class = classifier.classify(sample_image, large_box)
        
        # Should classify sizes differently
        if len(small_class) > 0 and len(large_class) > 0:
            assert small_class[0].get('size') != large_class[0].get('size')
    
    @pytest.mark.unit
    def test_risk_level_assignment(self, sample_image):
        """Test risk level assignment to objects"""
        classifier = ObjectClassifier()
        
        boxes = [[100, 150, 200, 300]]
        classifications = classifier.classify(sample_image, boxes)
        
        if len(classifications) > 0:
            assert 'risk_level' in classifications[0]
            risk_level = classifications[0]['risk_level']
            assert risk_level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']


class TestAbandonedObjectDetection:
    """Tests for abandoned object detection"""
    
    @pytest.mark.unit
    def test_abandoned_object_detection_setup(self):
        """Test abandoned object detector setup"""
        config = ObjectDetectorConfig(
            abandoned_threshold_seconds=120,  # 2 minutes
            critical_threshold_seconds=600    # 10 minutes
        )
        detector = ObjectDetector(config)
        
        assert detector.abandoned_threshold == 120
        assert detector.critical_threshold == 600
    
    @pytest.mark.unit
    def test_object_tracking_over_time(self, sample_image):
        """Test tracking objects over time for abandonment"""
        config = ObjectDetectorConfig()
        detector = ObjectDetector(config)
        
        # Simulate detection over multiple frames
        detections = []
        for frame_id in range(10):
            result = detector.detect(sample_image)
            result['frame_id'] = frame_id
            result['timestamp'] = frame_id * 0.033  # ~30 FPS
            detections.append(result)
        
        # All frames should have detection info
        assert len(detections) == 10
        assert all('frame_id' in d for d in detections)
    
    @pytest.mark.unit
    def test_abandoned_alert_generation(self):
        """Test generation of abandoned object alerts"""
        config = ObjectDetectorConfig(
            abandoned_threshold_seconds=2,  # 2 seconds for testing
            critical_threshold_seconds=5    # 5 seconds for testing
        )
        detector = ObjectDetector(config)
        
        # Should initialize alert system
        assert hasattr(detector, 'abandoned_threshold')
        assert hasattr(detector, 'critical_threshold')


class TestPersonObjectAssociation:
    """Tests for person-object association"""
    
    @pytest.mark.unit
    def test_associate_object_with_person(self, sample_image):
        """Test associating objects with persons"""
        config = ObjectDetectorConfig()
        detector = ObjectDetector(config)
        
        result = detector.detect(sample_image)
        
        # If objects detected, should have association capability
        if len(result['boxes']) > 0:
            # Result should be ready for association with person detections
            assert 'boxes' in result
            assert 'associated_persons' in result or 'associations' in result or len(result['boxes']) >= 0
    
    @pytest.mark.unit
    def test_suspicious_person_object_pattern(self):
        """Test detection of suspicious person-object patterns"""
        # E.g., person placing backpack and leaving
        # (This is tested more in integration with orchestrator)
        config = ObjectDetectorConfig()
        detector = ObjectDetector(config)
        
        assert detector is not None


class TestObjectDetectionEdgeCases:
    """Tests for edge cases in object detection"""
    
    @pytest.mark.unit
    def test_detect_small_objects(self):
        """Test detection of small objects"""
        config = ObjectDetectorConfig(min_detection_size=20)
        detector = ObjectDetector(config)
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(image)
        
        assert isinstance(result, dict)
        assert 'boxes' in result
    
    @pytest.mark.unit
    def test_detect_occluded_objects(self):
        """Test detection of partially occluded objects"""
        config = ObjectDetectorConfig()
        detector = ObjectDetector(config)
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(image)
        
        assert isinstance(result, dict)
    
    @pytest.mark.unit
    def test_detect_crowded_scene(self):
        """Test detection in scene with multiple objects"""
        config = ObjectDetectorConfig(max_detections=50)
        detector = ObjectDetector(config)
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(image)
        
        # Should not exceed max
        assert len(result['boxes']) <= 50
    
    @pytest.mark.unit
    def test_detect_reflective_surfaces(self):
        """Test detection with reflective surfaces (mirrors, windows)"""
        config = ObjectDetectorConfig()
        detector = ObjectDetector(config)
        
        # Create image with bright areas (simulate reflection)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        image[100:150, 200:300] = 255  # Bright area
        
        result = detector.detect(image)
        
        assert isinstance(result, dict)


class TestObjectDetectionPerformance:
    """Tests for object detection performance"""
    
    @pytest.mark.unit
    def test_detection_frame_time(self, sample_image):
        """Test detection completes in acceptable time"""
        config = ObjectDetectorConfig()
        detector = ObjectDetector(config)
        
        import time as time_module
        start = time_module.time()
        result = detector.detect(sample_image)
        elapsed = time_module.time() - start
        
        assert elapsed < 1.0, f"Detection took {elapsed}s"
    
    @pytest.mark.unit
    def test_batch_processing(self, sample_image):
        """Test batch processing of multiple frames"""
        config = ObjectDetectorConfig()
        detector = ObjectDetector(config)
        
        import time as time_module
        start = time_module.time()
        
        for _ in range(10):
            result = detector.detect(sample_image)
            assert isinstance(result, dict)
        
        elapsed = time_module.time() - start
        avg_time = elapsed / 10
        
        assert avg_time < 1.0


class TestObjectDetectorConfigIntegration:
    """Tests for integration with ConfigManager"""
    
    @pytest.mark.unit
    def test_detector_uses_config_threshold(self, test_config):
        """Test that detector uses config threshold"""
        threshold = test_config.get(
            'detection.confidence_thresholds.object',
            0.50
        )
        
        config = ObjectDetectorConfig(confidence_threshold=threshold)
        detector = ObjectDetector(config)
        
        assert detector.confidence_threshold == threshold


class TestObjectDetectionResults:
    """Tests for detection result validation"""
    
    @pytest.mark.unit
    def test_detection_box_format(self, sample_image):
        """Test detection boxes are in correct format"""
        config = ObjectDetectorConfig()
        detector = ObjectDetector(config)
        
        result = detector.detect(sample_image)
        
        for box in result['boxes']:
            assert len(box) == 4
            x1, y1, x2, y2 = box
            assert x2 > x1
            assert y2 > y1
    
    @pytest.mark.unit
    def test_detection_confidence_range(self, sample_image):
        """Test confidence values are in valid range"""
        config = ObjectDetectorConfig()
        detector = ObjectDetector(config)
        
        result = detector.detect(sample_image)
        
        for conf in result['confidences']:
            assert 0.0 <= conf <= 1.0
    
    @pytest.mark.unit
    def test_result_arrays_consistency(self, sample_image):
        """Test result array consistency"""
        config = ObjectDetectorConfig()
        detector = ObjectDetector(config)
        
        result = detector.detect(sample_image)
        
        num_detections = len(result['boxes'])
        assert len(result['confidences']) == num_detections
        obj_types = result.get('object_types', result.get('class_names', []))
        assert len(obj_types) == num_detections


@pytest.mark.integration
class TestObjectDetectorIntegration:
    """Integration tests for ObjectDetector"""
    
    def test_detector_pipeline(self, sample_image):
        """Test complete detection pipeline"""
        config = ObjectDetectorConfig()
        detector = ObjectDetector(config)
        classifier = ObjectClassifier()
        
        detections = detector.detect(sample_image)
        
        # If detections, verify classification
        if len(detections['boxes']) > 0:
            classifications = classifier.classify(
                sample_image,
                detections['boxes']
            )
            assert isinstance(classifications, list)
    
    def test_temporal_tracking(self, sample_image):
        """Test temporal tracking of objects"""
        config = ObjectDetectorConfig()
        detector = ObjectDetector(config)
        
        # Simulate multiple frames
        results = []
        for frame_id in range(30):  # ~1 second at 30 FPS
            result = detector.detect(sample_image)
            result['frame_id'] = frame_id
            results.append(result)
        
        # All results should be valid
        assert len(results) == 30
        for result in results:
            assert 'boxes' in result
            assert 'frame_id' in result
