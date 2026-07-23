"""
Object Detection Configuration

Configuration for detecting and tracking objects in surveillance.

Key focus:
- Detect objects that persons carry
- Identify abandoned objects (critical threat)
- Track object-person associations
- Assess risk based on context
"""

from dataclasses import dataclass
from typing import Literal, Optional, List


@dataclass
class ObjectDetectionConfig:
    """
    Configuration for object detection system.
    """
    
    # Model Configuration
    model_size: Literal['n', 's', 'm', 'l', 'x'] = 'm'
    """YOLOv8 model size - consistent with other detectors."""
    
    # Detection Thresholds
    confidence_threshold: float = 0.50
    """
    Minimum confidence for object detection.
    
    Why 0.50?
    - Objects are usually clear (bags, backpacks)
    - Don't want false object detections
    - Higher than person (0.45) and animal (0.40)
    
    Adjustments:
    - Weapon detection: 0.40 (don't miss)
    - General objects: 0.50 (standard)
    - Small objects: 0.55 (reduce noise)
    """
    
    iou_threshold: float = 0.50
    """NMS threshold."""
    
    # Processing Configuration
    input_size: tuple[int, int] = (640, 640)
    """Standard input resolution."""
    
    max_detections: int = 30
    """
    Maximum objects per frame.
    
    Why 30?
    - Checkpoints rarely have >30 bags/objects
    - Keeps processing manageable
    - If exceeded, triggers crowded-area mode
    """
    
    # Device Configuration
    device: Literal['cpu', 'cuda', 'mps'] = 'cuda'
    """Processing device."""
    
    half_precision: bool = True
    """Use FP16."""
    
    # Object-Specific Parameters
    min_object_area: int = 500
    """
    Minimum bounding box area for valid object.
    
    Why 500?
    - Objects smaller than this are hard to identify
    - Reduces noise from small artifacts
    - Still catches bottles, books (25x20 = 500)
    
    Compare:
    - Person: 400 px²
    - Vehicle: 2000 px²
    - Animal: 200 px²
    - Object: 500 px² (middle ground)
    """
    
    detect_backpacks: bool = True
    """Detect backpacks (critical for security)."""
    
    detect_handbags: bool = True
    """Detect handbags and purses."""
    
    detect_suitcases: bool = True
    """Detect suitcases (travel, cargo)."""
    
    detect_sports_equipment: bool = True
    """
    Detect sports equipment.
    
    Why track sports equipment?
    - Baseball bats can be weapons
    - Context awareness (sports venue vs. office)
    - Unusual equipment in wrong place = suspicious
    """
    
    detect_small_items: bool = False
    """
    Detect small items (bottles, books, etc.).
    
    Usually disabled to reduce noise.
    Enable for:
    - High-security checkpoints
    - Forensic analysis
    - Detailed activity logging
    """
    
    # Person-Object Association
    enable_person_association: bool = True
    """
    Associate objects with nearby persons.
    
    Critical feature for:
    - Abandoned object detection
    - Owner identification
    - Threat assessment
    """
    
    association_distance_threshold: int = 100
    """
    Maximum distance (pixels) to associate object with person.
    
    At 640x640 resolution:
    - 100 pixels ≈ 15% of frame width
    - Reasonable "carrying distance"
    - Close enough to assume association
    
    Adjustments:
    - Crowded areas: 60-80 (closer required)
    - Open spaces: 120-150 (more tolerance)
    """
    
    abandoned_time_threshold: float = 120.0
    """
    Time (seconds) before object considered abandoned.
    
    Why 120 seconds (2 minutes)?
    - Person steps away briefly: < 2 min (normal)
    - Person leaves object: > 2 min (suspicious)
    
    Context dependent:
    - Airport: 60 seconds (high security)
    - Office: 180 seconds (more tolerance)
    - Public area: 120 seconds (balanced)
    """
    
    critical_abandoned_time: float = 600.0
    """
    Time (seconds) before abandoned object is CRITICAL.
    
    10 minutes unattended = potential IED
    Requires immediate response.
    """
    
    # Risk Assessment
    enable_risk_assessment: bool = True
    """Calculate risk levels for detected objects."""
    
    restricted_zones: Optional[List[tuple]] = None
    """
    List of restricted zone coordinates [(x1,y1,x2,y2), ...].
    
    Objects in these zones get higher risk scores.
    """
    
    high_security_mode: bool = False
    """
    High security mode:
    - Lower confidence thresholds
    - Shorter abandoned time threshold
    - More aggressive risk assessment
    
    Use for:
    - Critical infrastructure
    - Government buildings
    - High-threat environments
    """
    
    # Size Classification Thresholds
    size_threshold_small_medium: int = 5000
    """
    Area threshold between SMALL and MEDIUM objects.
    
    At 640x640:
    - < 5,000: SMALL (bottles, books, small purses)
    - 5,000-15,000: MEDIUM (handbags, small backpacks)
    - > 15,000: LARGE (large backpacks, suitcases)
    """
    
    size_threshold_medium_large: int = 15000
    """Area threshold between MEDIUM and LARGE."""
    
    # Edge Case Handling
    low_light_boost: bool = True
    """Enhancement for low-light conditions."""
    
    skip_frames: int = 0
    """
    Frame skipping.
    
    Objects move slowly (carried by persons),
    can usually skip 1-2 frames without loss.
    
    - 0: Every frame (checkpoints)
    - 1-2: Performance optimization
    """
    
    # Logging
    log_all_objects: bool = False
    """
    Log all detected objects.
    
    Usually False (too much data).
    Enable for:
    - Forensic analysis
    - Training data collection
    - Incident investigation
    """
    
    log_abandoned_objects: bool = True
    """Always log abandoned objects (critical)."""
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        
        if self.min_object_area < 0:
            raise ValueError("min_object_area must be non-negative")
        
        if self.abandoned_time_threshold < 0:
            raise ValueError("abandoned_time_threshold must be non-negative")
        
        if self.association_distance_threshold < 0:
            raise ValueError("association_distance_threshold must be non-negative")
        
        if self.half_precision and self.device == 'cpu':
            self.half_precision = False
            print("Warning: half_precision disabled for CPU inference")
        
        # Size thresholds validation
        if self.size_threshold_small_medium >= self.size_threshold_medium_large:
            raise ValueError(
                "size_threshold_small_medium must be less than size_threshold_medium_large"
            )
        
        # High security mode adjustments
        if self.high_security_mode:
            self.confidence_threshold = max(0.40, self.confidence_threshold - 0.10)
            self.abandoned_time_threshold = min(60.0, self.abandoned_time_threshold)
            print(f"High security mode: confidence={self.confidence_threshold}, "
                  f"abandoned_time={self.abandoned_time_threshold}s")
