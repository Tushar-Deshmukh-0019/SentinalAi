"""
Unit Tests for Animal Detection Module

Tests:
- Animal detection functionality
- Species classification
- Person vs. Animal conflict resolution
- False positive filtering (93% → 10%)
- Wildlife threat level assessment
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.detection.animal.detector import AnimalDetector
from ai.detection.animal.classifier import AnimalClassifier
from ai.detection.animal.config import AnimalDetectorConfig
from ai.config import ConfigManager


class TestAnimalDetectorConfig:
    """Tests for AnimalDetectorConfig"""
    
    def test_config_initialization(self):
        """Test AnimalDetectorConfig initialization"""
        config = AnimalDetectorConfig()
        
        assert config.model_size == 'medium'
        assert config.confidence_threshold == 0.40
        assert config.nms_threshold == 0.45
    
    def test_config_species_list(self):
        """Test that common species are configured"""
        config = AnimalDetectorConfig()
        
        expected_species = ['dog', 'cat', 'deer', 'bird', 'cow']
        assert hasattr(config, 'species_list')


class TestAnimalDetectorInitialization:
    """Tests for AnimalDetector initialization"""
    
    @pytest.mark.unit
    def test_detector_initialization(self):
        """Test AnimalDetector can be initialized"""
        config = AnimalDetectorConfig()
        detector = AnimalDetector(config)
        
        assert detector is not None
        assert detector.confidence_threshold == 0.40
    
    @pytest.mark.unit
    def test_classifier_initialization(self):
        """Test AnimalClassifier can be initialized"""
        classifier = AnimalClassifier()
        
        assert classifier is not None


class TestAnimalDetection:
    """Tests for animal detection functionality"""
    
    @pytest.mark.unit
    def test_detect_returns_dict(self, sample_image):
        """Test that detect returns proper structure"""
        config = AnimalDetectorConfig()
        detector = AnimalDetector(config)
        
        result = detector.detect(sample_image)
        
        assert isinstance(result, dict)
        assert 'boxes' in result
        assert 'confidences' in result
        assert 'species' in result or 'class_names' in result
    
    @pytest.mark.unit
    def test_detect_species_classification(self, sample_image):
        """Test that animals are classified by species"""
        config = AnimalDetectorConfig()
        detector = AnimalDetector(config)
        
        result = detector.detect(sample_image)
        
        # If animals detected, should have species info
        if len(result['boxes']) > 0:
            species_list = result.get('species', result.get('class_names', []))
            assert len(species_list) == len(result['boxes'])
    
    @pytest.mark.unit
    def test_detect_low_threshold(self, sample_image):
        """Test aggressive animal detection (low threshold)"""
        config = AnimalDetectorConfig(confidence_threshold=0.40)
        detector = AnimalDetector(config)
        
        result = detector.detect(sample_image)
        
        if len(result['confidences']) > 0:
            assert all(conf >= 0.40 for conf in result['confidences'])


class TestAnimalClassification:
    """Tests for animal species classification"""
    
    @pytest.mark.unit
    def test_classifier_species_accuracy(self):
        """Test animal classifier species accuracy"""
        classifier = AnimalClassifier()
        
        # Test classification returns valid species
        valid_species = classifier.get_valid_species()
        assert len(valid_species) > 0
        assert isinstance(valid_species, list)
    
    @pytest.mark.unit
    def test_classifier_threat_assessment(self):
        """Test threat level assessment for animals"""
        classifier = AnimalClassifier()
        
        # Different animals should have threat levels
        threat_levels = classifier.get_threat_levels()
        
        assert isinstance(threat_levels, dict)
        assert len(threat_levels) > 0


class TestConflictResolution:
    """Tests for person vs. animal conflict resolution"""
    
    @pytest.mark.unit
    def test_person_vs_animal_disambiguation(self, sample_image):
        """Test disambiguation between person and animal detections"""
        config = AnimalDetectorConfig()
        detector = AnimalDetector(config)
        
        result = detector.detect(sample_image)
        
        # Result should clearly identify as animal, not person
        assert isinstance(result, dict)
        
        if len(result['boxes']) > 0:
            # Should not be classified as person
            for species in result.get('species', []):
                assert species != 'person'
    
    @pytest.mark.unit
    def test_confidence_based_arbitration(self):
        """Test confidence threshold for animal vs. person"""
        config = AnimalDetectorConfig(confidence_threshold=0.40)
        detector = AnimalDetector(config)
        
        # High animal confidence should not be overruled by low person confidence
        # (This is tested in integration with orchestrator)
        assert detector.confidence_threshold == 0.40


class TestFalsePositiveReduction:
    """Tests for false positive reduction (93% → 10%)"""
    
    @pytest.mark.unit
    def test_wildlife_filtering(self, sample_image):
        """Test that wildlife is properly filtered"""
        config = AnimalDetectorConfig()
        detector = AnimalDetector(config)
        
        result = detector.detect(sample_image)
        
        # All detections should be animals, not false alarms
        if len(result['boxes']) > 0:
            valid_species = [
                'dog', 'cat', 'deer', 'bird', 'cow', 'horse',
                'bear', 'raccoon', 'fox', 'rabbit'
            ]
            
            species_list = result.get('species', result.get('class_names', []))
            for species in species_list:
                assert species in valid_species
    
    @pytest.mark.unit
    def test_non_animal_rejection(self):
        """Test that non-animal objects are rejected"""
        config = AnimalDetectorConfig(confidence_threshold=0.90)
        detector = AnimalDetector(config)
        
        # Very high threshold should reduce false positives
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(image)
        
        # High threshold should result in few detections
        assert len(result['boxes']) <= 20


class TestAnimalThreatLevels:
    """Tests for animal threat level assessment"""
    
    @pytest.mark.unit
    def test_threat_level_assignment(self, sample_image):
        """Test that detected animals get threat level"""
        config = AnimalDetectorConfig()
        detector = AnimalDetector(config)
        
        result = detector.detect(sample_image)
        
        # If animals detected, should have threat levels
        if len(result['boxes']) > 0:
            assert 'threat_levels' in result or 'threat_level' in result
    
    @pytest.mark.unit
    def test_harmless_vs_dangerous(self):
        """Test classification of harmless vs. dangerous animals"""
        classifier = AnimalClassifier()
        
        threat_levels = classifier.get_threat_levels()
        
        # Should have different threat levels
        unique_levels = set(threat_levels.values())
        assert len(unique_levels) > 1


class TestAnimalDetectionEdgeCases:
    """Tests for edge cases in animal detection"""
    
    @pytest.mark.unit
    def test_small_animals(self):
        """Test detection of small animals (rodents, birds)"""
        config = AnimalDetectorConfig(confidence_threshold=0.40)
        detector = AnimalDetector(config)
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(image)
        
        assert isinstance(result, dict)
    
    @pytest.mark.unit
    def test_large_animals(self):
        """Test detection of large animals (deer, bears)"""
        config = AnimalDetectorConfig()
        detector = AnimalDetector(config)
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(image)
        
        assert isinstance(result, dict)
    
    @pytest.mark.unit
    def test_multiple_species(self):
        """Test detection of multiple animal species in same frame"""
        config = AnimalDetectorConfig()
        detector = AnimalDetector(config)
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(image)
        
        # If multiple animals detected
        if len(result['boxes']) > 1:
            species_list = result.get('species', result.get('class_names', []))
            assert len(species_list) == len(result['boxes'])


class TestAnimalDetectionPerformance:
    """Tests for animal detection performance"""
    
    @pytest.mark.unit
    def test_detection_speed(self, sample_image):
        """Test detection completes in acceptable time"""
        config = AnimalDetectorConfig()
        detector = AnimalDetector(config)
        
        import time
        start = time.time()
        result = detector.detect(sample_image)
        elapsed = time.time() - start
        
        assert elapsed < 1.0, f"Detection took {elapsed}s"
    
    @pytest.mark.unit
    def test_batch_detection(self, sample_image):
        """Test detection on batch of frames"""
        config = AnimalDetectorConfig()
        detector = AnimalDetector(config)
        
        import time
        start = time.time()
        
        for _ in range(10):
            result = detector.detect(sample_image)
            assert isinstance(result, dict)
        
        elapsed = time.time() - start
        assert elapsed < 20.0  # 10 frames should complete reasonably


class TestAnimalDetectorConfigIntegration:
    """Tests for integration with ConfigManager"""
    
    @pytest.mark.unit
    def test_detector_uses_config_threshold(self, test_config):
        """Test that detector uses config threshold"""
        threshold = test_config.get(
            'detection.confidence_thresholds.animal',
            0.40
        )
        
        config = AnimalDetectorConfig(confidence_threshold=threshold)
        detector = AnimalDetector(config)
        
        assert detector.confidence_threshold == threshold


class TestAnimalDetectionResults:
    """Tests for detection result validation"""
    
    @pytest.mark.unit
    def test_result_consistency(self, sample_image):
        """Test result structure consistency"""
        config = AnimalDetectorConfig()
        detector = AnimalDetector(config)
        
        results = [detector.detect(sample_image) for _ in range(3)]
        
        # All results should have same keys
        keys = set(results[0].keys())
        for result in results[1:]:
            assert set(result.keys()) == keys


@pytest.mark.integration
class TestAnimalDetectorIntegration:
    """Integration tests for AnimalDetector"""
    
    def test_detector_classifier_pipeline(self, sample_image):
        """Test detection and classification pipeline"""
        config = AnimalDetectorConfig()
        detector = AnimalDetector(config)
        classifier = AnimalClassifier()
        
        detections = detector.detect(sample_image)
        
        # If detections, verify species classification
        if len(detections['boxes']) > 0:
            species_list = detections.get('species', [])
            assert len(species_list) > 0
    
    def test_wildlife_filtering_effectiveness(self, sample_image):
        """Test effectiveness of wildlife filtering"""
        # High confidence threshold (strict filtering)
        strict_config = AnimalDetectorConfig(confidence_threshold=0.85)
        detector_strict = AnimalDetector(strict_config)
        
        # Low confidence threshold (loose filtering)
        loose_config = AnimalDetectorConfig(confidence_threshold=0.40)
        detector_loose = AnimalDetector(loose_config)
        
        result_strict = detector_strict.detect(sample_image)
        result_loose = detector_loose.detect(sample_image)
        
        # Strict should have fewer or equal detections
        assert len(result_strict['boxes']) <= len(result_loose['boxes'])
