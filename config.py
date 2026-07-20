"""
Vehicle Detection Configuration

Configuration parameters for vehicle detection system.

In real defense systems, these are tuned based on:
- Environment (highway, border road, urban, off-road)
- Camera placement (overhead, angle, distance)
- Operational requirements (detect all vehicles vs. only large vehicles)
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class VehicleDetectionConfig:
    """
    Configuration for vehicle detection system.
    
    Similar to person detection but with vehicle-specific considerations.
    """
    
    # Model Configuration
    model_size: Literal['n', 's', 'm', 'l', 'x'] = 'm'
    """
    YOLOv8 model size.
    
    Same options as person detection:
    - 'n' (nano): Fastest
    - 's' (small): Edge devices
    - 'm' (medium): RECOMMENDED - good balance
    - 'l' (large): Higher accuracy
    - 'x' (xlarge): Maximum accuracy
    
    Note: Vehicles are usually larger than persons in frame,
    so smaller models often perform adequately.
    """
    
    # Detection Thresholds
    confidence_threshold: float = 0.50
    """
    Minimum confidence for valid vehicle detection.
    
    Why 0.50 (vs. 0.45 for persons)?
    - Vehicles are larger, easier to detect clearly
    - Less affected by lighting (larger surface area)
    - Shape is more distinctive
    - Lower false positive tolerance (person FN > vehicle FN)
    
    Environment adjustments:
    - Highway/clear view: 0.55-0.60
    - Forest road/partial view: 0.45-0.50
    - Night/adverse weather: 0.40-0.45
    """
    
    iou_threshold: float = 0.50
    """
    NMS threshold for overlapping detections.
    
    Slightly higher than person detection (0.45) because:
    - Vehicles less likely to overlap tightly
    - When they do overlap, usually want to merge
    - In convoy, vehicles maintain spacing
    """
    
    # Processing Configuration
    input_size: tuple[int, int] = (640, 640)
    """
    Input resolution for detection.
    
    Same considerations as person detection:
    - 640x640: Standard, good performance
    - 1280x1280: Long-range detection (border cameras)
    - 416x416: Fast processing, short range
    """
    
    max_detections: int = 50
    """
    Maximum vehicles per frame.
    
    Why 50 (vs. 100 for persons)?
    - Vehicles take more space, fewer fit in frame
    - >50 vehicles = traffic scenario (different module)
    - Border surveillance rarely sees >10 vehicles
    
    Exception:
    - Highway monitoring: 100+
    - Parking lot surveillance: 200+
    """
    
    # Device Configuration
    device: Literal['cpu', 'cuda', 'mps'] = 'cuda'
    """Processing device."""
    
    half_precision: bool = True
    """Use FP16 for 2x speedup."""
    
    # Vehicle-Specific Parameters
    min_vehicle_area: int = 2000
    """
    Minimum bounding box area for valid vehicle detection.
    
    Why 2000 vs. 400 for persons?
    - Vehicles are inherently larger
    - Very small detections usually distant/irrelevant
    - Prevents false positives from small objects
    
    Example:
    - 50x40 pixel box = 2000 pixels²
    - Motorcycle at 100m distance
    - Smaller than this is too far for actionable intelligence
    
    Adjust for:
    - Long-range cameras: 1000-1500
    - Close-range gates: 3000-5000
    """
    
    max_vehicle_area: Optional[int] = None
    """
    Maximum area threshold.
    
    Why limit maximum?
    - Unusually large detection might be building edge
    - Camera obstruction (someone standing very close)
    - Stitching artifact in multi-camera setup
    
    Set to None to disable (most scenarios).
    Set to frame_width * frame_height * 0.8 to prevent full-frame detections.
    """
    
    detect_motorcycles: bool = True
    """
    Whether to detect motorcycles.
    
    Motorcycles have different tactical profile:
    - High mobility
    - Small, hard to track
    - Can go off-road
    - Usually 1-2 occupants
    
    In some scenarios (heavy border), motorcycles are HIGH priority.
    In others (vehicle checkpoints), might be filtered out.
    """
    
    detect_large_vehicles: bool = True
    """
    Whether to detect trucks and buses.
    
    Large vehicles:
    - High cargo capacity
    - Usually commercial/authorized
    - When unauthorized = major threat
    
    Usually True, but in some urban scenarios, might filter.
    """
    
    license_plate_detection: bool = True
    """
    Whether to attempt license plate region detection.
    
    Note: This is REGION detection, not OCR (Optical Character Recognition).
    OCR will be added in later module (Day 40+).
    
    Why detect region?
    - Presence/absence of plate is evidence
    - Region extraction for downstream OCR
    - Covered/missing plates = suspicion indicator
    """
    
    # Edge Case Handling
    low_light_boost: bool = True
    """
    Apply preprocessing for low-light conditions.
    
    Same as person detection but less critical:
    - Vehicles often have reflectors, lights
    - Larger surface area catches more light
    - Headlights in night create strong signal
    """
    
    skip_frames: int = 0
    """
    Frame skipping for performance.
    
    Vehicles move slower than detection rate:
    - At 30 FPS, vehicle moves ~1-2 meters between frames at highway speed
    - Can often skip 1-2 frames without missing vehicle
    - Useful for multi-camera setups
    
    Settings:
    - 0: Every frame (intersections, gates)
    - 1: Every other frame (highways)
    - 2-3: Long-range perimeter monitoring
    """
    
    stationary_detection: bool = True
    """
    Detect stationary (parked) vehicles.
    
    Why separate flag?
    - Stationary vehicles blend with background
    - Might want to filter parked cars in some scenarios
    - Different threat profile (parked = potential IED, surveillance)
    
    Implementation:
    - Temporal analysis (vehicle in same spot across frames)
    - Done in tracking module (Day 13+)
    - This flag enables/disables the analysis
    """
    
    # Size Classification Thresholds
    size_threshold_small_medium: int = 15000
    """
    Area threshold between SMALL and MEDIUM vehicles (pixels²).
    
    Approximate classifications:
    - < 15,000: SMALL (motorcycles, compact cars)
    - 15,000 - 40,000: MEDIUM (sedans, SUVs, pickups)
    - > 40,000: LARGE (trucks, buses)
    
    Note: These are at 640x640 input resolution.
    Adjust proportionally for different input sizes.
    """
    
    size_threshold_medium_large: int = 40000
    """Area threshold between MEDIUM and LARGE vehicles."""
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        
        if self.min_vehicle_area < 0:
            raise ValueError("min_vehicle_area must be non-negative")
        
        if self.max_vehicle_area is not None and self.max_vehicle_area < self.min_vehicle_area:
            raise ValueError("max_vehicle_area must be greater than min_vehicle_area")
        
        if self.half_precision and self.device == 'cpu':
            self.half_precision = False
            print("Warning: half_precision disabled for CPU inference")
        
        # Size thresholds validation
        if self.size_threshold_small_medium >= self.size_threshold_medium_large:
            raise ValueError(
                "size_threshold_small_medium must be less than size_threshold_medium_large"
            )
