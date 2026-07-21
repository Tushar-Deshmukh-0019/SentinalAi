"""
Animal Detection Configuration

Configuration for the false-positive filtering system.

Key consideration:
We want to catch ALL animals (low threshold) to prevent false person alerts,
but we don't want to miss actual persons (careful conflict resolution).
"""

from dataclasses import dataclass
from typing import Literal, List, Optional


@dataclass
class AnimalDetectionConfig:
    """
    Configuration for animal detection system.
    
    Tuned for false-positive reduction while maintaining person detection accuracy.
    """
    
    # Model Configuration
    model_size: Literal['n', 's', 'm', 'l', 'x'] = 'm'
    """
    YOLOv8 model size.
    
    Recommendation: Same as person detection ('m') for consistency.
    """
    
    # Detection Thresholds
    confidence_threshold: float = 0.40
    """
    Minimum confidence for animal detection.
    
    Why 0.40 (lower than person's 0.45)?
    - Want to catch all potential animals
    - Better to have false animal detection than false person detection
    - Low-confidence animals still useful for filtering
    
    Philosophy:
    - Low confidence person (0.48) + High confidence animal (0.85) = Filter as animal
    - We're trading precision for recall on animals
    
    Settings by environment:
    - Urban (few animals): 0.45-0.50
    - Rural/forest (many animals): 0.35-0.40
    - Wildlife preserve: 0.30-0.35
    """
    
    iou_threshold: float = 0.45
    """NMS threshold for overlapping detections."""
    
    # Processing Configuration
    input_size: tuple[int, int] = (640, 640)
    """Input resolution - same as person/vehicle for consistency."""
    
    max_detections: int = 50
    """
    Maximum animals per frame.
    
    Why 50?
    - Herd animals (deer, sheep) can appear in groups
    - Bird flocks can trigger many detections
    - Still reasonable for processing
    
    Exception:
    - Bird feeding area: 100+
    - Livestock area: 200+
    """
    
    # Device Configuration
    device: Literal['cpu', 'cuda', 'mps'] = 'cuda'
    """Processing device."""
    
    half_precision: bool = True
    """Use FP16 for speedup."""
    
    # Animal-Specific Parameters
    min_animal_area: int = 200
    """
    Minimum bounding box area for valid animal detection.
    
    Why 200 (vs. 400 for person, 2000 for vehicle)?
    - Animals can be small (cats, birds)
    - Even small animals can trigger false alerts
    - Want to catch them all
    
    Examples at 640x640:
    - 15x15 = 225 px² (small bird)
    - 20x30 = 600 px² (cat)
    - 40x80 = 3200 px² (deer)
    """
    
    detect_birds: bool = True
    """
    Whether to detect birds.
    
    Birds are common false positive triggers:
    - Large birds (crows, eagles) can look like movement
    - Flocks create motion alerts
    - Usually want to filter them out
    
    Disable only if:
    - Birds are not present in area
    - Camera angle doesn't capture ground level
    """
    
    detect_small_animals: bool = True
    """
    Whether to detect cats and small animals.
    
    Small animals less likely to trigger person detection,
    but can still cause motion alerts.
    """
    
    detect_livestock: bool = True
    """
    Whether to detect cows, sheep, horses.
    
    In agricultural/rural settings, livestock is expected.
    Detection allows for:
    - Logging livestock movement
    - Verifying they're in correct zones
    - Detecting unusual livestock behavior
    """
    
    # Conflict Resolution
    enable_conflict_resolution: bool = True
    """
    Whether to resolve person/animal conflicts.
    
    When both person and animal detected at similar location:
    - Compare confidence scores
    - Use size/shape heuristics
    - Make determination: person or animal
    
    Critical feature for reducing false positives.
    """
    
    conflict_confidence_threshold: float = 0.20
    """
    Confidence difference threshold for conflict resolution.
    
    If animal confidence - person confidence > 0.20:
        → Classify as animal
    If person confidence - animal confidence > 0.20:
        → Classify as person
    Else:
        → Use heuristics or default to person (safe choice)
    
    Example:
    - Person: 0.52, Animal: 0.88 → Difference: 0.36 → Animal ✓
    - Person: 0.65, Animal: 0.50 → Difference: -0.15 → Person ✓
    - Person: 0.60, Animal: 0.55 → Difference: -0.05 → Heuristics needed
    """
    
    proximity_threshold: int = 50
    """
    Distance threshold (pixels) to consider person/animal "near" each other.
    
    Used for:
    - Dog with person detection (normal)
    - Determining if animal is "alone"
    - Context for threat assessment
    
    50 pixels at 640x640 ≈ 8% of frame width (reasonable proximity)
    """
    
    # Size Classification Thresholds
    size_threshold_small_medium: int = 3000
    """
    Area threshold between SMALL and MEDIUM animals.
    
    At 640x640 resolution:
    - < 3,000 px²: SMALL (birds, cats, small dogs)
    - 3,000-15,000 px²: MEDIUM (dogs, sheep, coyotes)
    - > 15,000 px²: LARGE (deer, bears, cattle, horses)
    
    Compare to person: ~8,000-25,000 px²
    """
    
    size_threshold_medium_large: int = 15000
    """Area threshold between MEDIUM and LARGE animals."""
    
    # Behavioral Analysis
    enable_movement_analysis: bool = True
    """
    Whether to analyze movement patterns.
    
    Movement helps distinguish:
    - Quadrupedal (four-legged) vs. bipedal (two-legged)
    - Flight pattern (birds) vs. walking
    - Grazing (stationary with head down) vs. alert posture
    
    Requires temporal data (multiple frames) - done in tracking module.
    This flag enables preparation for that analysis.
    """
    
    # Filtering Options
    auto_filter_wildlife: bool = True
    """
    Automatically filter out high-confidence wildlife detections.
    
    If True:
    - Deer with 0.90 confidence → Filtered, no alert
    - Bear with 0.85 confidence → Logged, possibly alerted (dangerous animal)
    - Bird with 0.95 confidence → Filtered completely
    
    If False:
    - All detections logged and sent to operator
    - Useful for wildlife monitoring scenarios
    """
    
    wildlife_confidence_threshold: float = 0.70
    """
    Minimum confidence to auto-filter wildlife.
    
    Only filter if:
    - Is wildlife (deer, bird, etc.)
    - Confidence > this threshold
    - Threat level is NONE or LOW
    
    Why 0.70?
    - Must be confident it's actually an animal
    - Don't want to filter potential persons
    - Lower confidence wildlife still logged, not filtered
    """
    
    log_filtered_detections: bool = True
    """
    Whether to log detections that were filtered out.
    
    Logging provides:
    - Wildlife activity patterns
    - System performance metrics
    - Audit trail
    - False positive statistics
    
    Recommended: True (storage is cheap, data is valuable)
    """
    
    # Edge Case Handling
    low_light_boost: bool = True
    """
    Apply preprocessing for low-light conditions.
    
    Animals are often most active at dawn/dusk/night.
    Enhancement helps detect them for filtering.
    """
    
    skip_frames: int = 0
    """
    Frame skipping for performance.
    
    Animals move slower than detection rate, can often skip frames.
    
    Settings:
    - 0: Every frame (real-time critical)
    - 1-2: Performance optimization
    - 3-5: Batch processing
    """
    
    # Expected Animals by Zone
    expected_animals: Optional[List[str]] = None
    """
    List of animal types expected in this zone.
    
    Example:
    - Livestock area: ['cow', 'sheep', 'horse']
    - Forest edge: ['deer', 'bear', 'bird']
    - Urban perimeter: ['dog', 'cat']
    
    If None, all animals considered unexpected (default).
    Expected animals get lower threat scores.
    """
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        
        if self.min_animal_area < 0:
            raise ValueError("min_animal_area must be non-negative")
        
        if not 0.0 <= self.conflict_confidence_threshold <= 1.0:
            raise ValueError("conflict_confidence_threshold must be between 0 and 1")
        
        if not 0.0 <= self.wildlife_confidence_threshold <= 1.0:
            raise ValueError("wildlife_confidence_threshold must be between 0 and 1")
        
        if self.half_precision and self.device == 'cpu':
            self.half_precision = False
            print("Warning: half_precision disabled for CPU inference")
        
        # Size thresholds validation
        if self.size_threshold_small_medium >= self.size_threshold_medium_large:
            raise ValueError(
                "size_threshold_small_medium must be less than size_threshold_medium_large"
            )
        
        # Parse expected animals to enum values if provided
        if self.expected_animals:
            from .classifier import AnimalType
            parsed = []
            for animal_str in self.expected_animals:
                try:
                    animal_type = AnimalType[animal_str.upper()]
                    parsed.append(animal_type)
                except KeyError:
                    print(f"Warning: Unknown animal type '{animal_str}' in expected_animals")
            self.expected_animals = parsed
