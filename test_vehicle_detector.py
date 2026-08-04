"""
Unit Tests for Vehicle Detection Module

Tests:
- Vehicle detection core functionality
- Vehicle classification (cars, trucks, motorcycles, buses)
- Confidence thresholding
- Edge cases (distance, occlusion, weather)
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.detection.vehicle.detector import VehicleDetector
from ai.detection.vehicle.classifier import VehicleClassifier
from ai.detection.vehicle.config import VehicleDetectorConfig
from ai.config import ConfigManager


class TestVehicleDetectorConfig:
    """Tests for VehicleDetectorConfig"""
    
    def test_config_initialization(self):
        """Test VehicleDetectorConfig initialization with defaults"""
        config = VehicleDetectorConfig()
        
        assert config.model_size == 'medium'
        assert config.confidence_threshold == 0.50
        assert config.nms_threshold == 0.45
        assert config.max_detections == 50
    
    def test_config_vehicle_classes(self):
        """Test vehicle class definitions"""
        config = VehicleDetectorConfig()
        
        # Should support standard vehicle types
        expected_types = ['car', 'truck', 'motorcycle', 'bus']
        assert all(vtype in config.vehicle_types for vtype in expected_types)


class TestVehicleDetectorInitialization:
    """Tests for VehicleDetector initialization"""
    
    @pytest.mark.unit
    def test_detector_initialization(self):
        """Test VehicleDetector can be initialized"""
        config = VehicleDetectorConfig()
        detector = VehicleDetector(config)
        
        assert detector is not None
        assert detector.confidence_threshold == 0.50
    
    @pytest.mark.unit
    def test_classifier_initialization(self):
        """Test VehicleClassifier can be initialized"""
        classifier = VehicleClassifier()
        
        assert classifier is not None


class TestVehicleDetection:
    """Tests for vehicle detection functionality"""
    
    @pytest.mark.unit
    def test_detect_returns_dict(self, sample_image):
        """Test that detect returns proper dictionary structure"""
        config = VehicleDetectorConfig()
        detector = VehicleDetector(config)
        
        result = detector.detect(sample_image)
        
        assert isinstance(result, dict)
        assert 'boxes' in result
        assert 'confidences' in result
        assert 'class_ids' in result
        assert 'class_names' in result
    
    @pytest.mark.unit
    def test_detect_vehicles_classification(self, sample_image):
        """Test that detected vehicles are classified"""
        config = VehicleDetectorConfig()
        detector = VehicleDetector(config)
        
        result = detector.detect(sample_image)
        
        # If vehicles detected, should have classification
        if len(result['boxes']) > 0:
            assert len(result['class_names']) == len(result['boxes'])
            
            # All class names should be valid vehicle types
            valid_types = ['car', 'truck', 'motorcycle', 'bus']
            for class_name in result['class_names']:
                assert class_name in valid_types
    
    @pytest.mark.unit
    def test_detect_confidence_threshold(self, sample_image):
        """Test confidence threshold filtering for vehicles"""
        config = VehicleDetectorConfig(confidence_threshold=0.80)
        detector = VehicleDetector(config)
        
        result = detector.detect(sample_image)
        
        if len(result['confidences']) > 0:
            assert all(conf >= 0.80 for conf in result['confidences'])


class TestVehicleClassification:
    """Tests for vehicle classification"""
    
    @pytest.mark.unit
    def test_classifier_output_format(self, sample_image):
        """Test vehicle classifier output format"""
        classifier = VehicleClassifier()
        
        # Mock detection boxes
        boxes = [[100, 150, 200, 300]]
        
        classifications = classifier.classify(sample_image, boxes)
        
        assert isinstance(classifications, list)
        if len(classifications) > 0:
            assert 'type' in classifications[0]
            assert 'size' in classifications[0]
            assert 'confidence' in classifications[0]
    
    @pytest.mark.unit
    def test_vehicle_size_classification(self, sample_image):
        """Test vehicle size classification"""
        classifier = VehicleClassifier()
        
        # Small vehicle box
        small_box = [[100, 100, 120, 140]]
        # Large vehicle box
        large_box = [[100, 100, 250, 300]]
        
        small_class = classifier.classify(sample_image, small_box)
        large_class = classifier.classify(sample_image, large_box)
        
        # Both should return classifications
        assert isinstance(small_class, list)
        assert isinstance(large_class, list)


class TestVehicleDetectionEdgeCases:
    """Tests for edge cases in vehicle detection"""
    
    @pytest.mark.unit
    def test_detect_distant_vehicles(self):
        """Test detection of distant/small vehicles"""
        config = VehicleDetectorConfig()
        detector = VehicleDetector(config)
        
        # Create image with small vehicle-like objects
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = detector.detect(image)
        
        assert isinstance(result, dict)
        assert 'boxes' in result
    
    @pytest.mark.unit
    def test_detect_occluded_vehicles(self):
        """Test detection of partially occluded vehicles"""
        config = VehicleDetectorConfig()
        detector = VehicleDetector(config)
        
        # Create image simulating occlusion
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = detector.detect(image)
        
        assert isinstance(result, dict)
    
    @pytest.mark.unit
    def test_detect_crowded_scene(self):
        """Test detection in scene with multiple vehicles"""
        config = VehicleDetectorConfig(max_detections=50)
        detector = VehicleDetector(config)
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = detector.detect(image)
        
        # Should not exceed max_detections
        assert len(result['boxes']) <= 50
    
    @pytest.mark.unit
    def test_detect_weather_conditions(self):
        """Test detection in various weather conditions"""
        config = VehicleDetectorConfig()
        detector = VehicleDetector(config)
        
        # Normal image
        normal = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        
        # Foggy (reduced contrast)
        foggy = np.random.randint(150, 200, (480, 640, 3), dtype=np.uint8)
        
        # Rainy (noise)
        rainy = normal + np.random.randint(-30, 30, (480, 640, 3), dtype=np.uint8)
        
        results = [
            detector.detect(normal),
            detector.detect(foggy),
            detector.detect(rainy)
        ]
        
        for result in results:
            assert isinstance(result, dict)
            assert 'boxes' in result


class TestVehicleDetectionPerformance:
    """Tests for vehicle detection performance"""
    
    @pytest.mark.unit
    def test_detect_frame_time(self, sample_image):
        """Test detection completes in acceptable time"""
        config = VehicleDetectorConfig()
        detector = VehicleDetector(config)
        
        import time
        start = time.time()
        result = detector.detect(sample_image)
        elapsed = time.time() - start
        
        # Vehicle detection should complete in reasonable time
        assert elapsed < 1.0, f"Detection took {elapsed}s"
    
    @pytest.mark.unit
    def test_detect_multiple_frames(self, sample_image):
        """Test detection on multiple frames"""
        config = VehicleDetectorConfig()
        detector = VehicleDetector(config)
        
        import time
        start = time.time()
        
        for _ in range(5):
            result = detector.detect(sample_image)
            assert isinstance(result, dict)
        
        elapsed = time.time() - start
        avg_time = elapsed / 5
        
        assert avg_time < 1.0


class TestVehicleDetectorConfigIntegration:
    """Tests for integration with ConfigManager"""
    
    @pytest.mark.unit
    def test_detector_uses_config_threshold(self, test_config):
        """Test that detector uses threshold from config"""
        threshold = test_config.get(
            'detection.confidence_thresholds.vehicle',
            0.50
        )
        
        config = VehicleDetectorConfig(confidence_threshold=threshold)
        detector = VehicleDetector(config)
        
        assert detector.confidence_threshold == threshold


class TestVehicleDetectionResults:
    """Tests for detection result validation"""
    
    @pytest.mark.unit
    def test_detection_box_format(self, sample_image):
        """Test detection boxes are in correct format"""
        config = VehicleDetectorConfig()
        detector = VehicleDetector(config)
        
        result = detector.detect(sample_image)
        
        for box in result['boxes']:
            assert len(box) == 4
            x1, y1, x2, y2 = box
            assert x2 > x1
            assert y2 > y1
    
    @pytest.mark.unit
    def test_detection_confidence_range(self, sample_image):
        """Test that confidence values are in valid range"""
        config = VehicleDetectorConfig()
        detector = VehicleDetector(config)
        
        result = detector.detect(sample_image)
        
        for conf in result['confidences']:
            assert 0.0 <= conf <= 1.0
    
    @pytest.mark.unit
    def test_result_arrays_consistency(self, sample_image):
        """Test that result arrays have consistent lengths"""
        config = VehicleDetectorConfig()
        detector = VehicleDetector(config)
        
        result = detector.detect(sample_image)
        
        num_detections = len(result['boxes'])
        assert len(result['confidences']) == num_detections
        assert len(result['class_ids']) == num_detections
        assert len(result['class_names']) == num_detections


@pytest.mark.integration
class TestVehicleDetectorIntegration:
    """Integration tests for VehicleDetector"""
    
    def test_detector_pipeline(self, sample_image):
        """Test complete detection pipeline"""
        config = VehicleDetectorConfig()
        detector = VehicleDetector(config)
        classifier = VehicleClassifier()
        
        detections = detector.detect(sample_image)
        
        # If detections, run through classifier
        if len(detections['boxes']) > 0:
            classifications = classifier.classify(
                sample_image,
                detections['boxes']
            )
            assert isinstance(classifications, list)
    
    def test_detector_sequential_frames(self, sample_image):
        """Test detector on sequential frames"""
        config = VehicleDetectorConfig()
        detector = VehicleDetector(config)
        
        results = []
        for _ in range(3):
            result = detector.detect(sample_image)
            results.append(result)
        
        # All results should be consistent
        for result in results:
            assert isinstance(result, dict)
            assert set(result.keys()) == set(results[0].keys())
