"""
Person Detection Configuration

Configuration parameters that affect detection behavior.
In real defense systems, these are tuned based on:
- Environment (desert, forest, urban)
- Camera specifications (resolution, FPS, night vision capability)
- Operational requirements (speed vs accuracy tradeoff)
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class PersonDetectionConfig:
    """
    Configuration for person detection system.
    
    These parameters are critical for balancing:
    - False Positives (detecting person when there isn't one)
    - False Negatives (missing actual persons)
    
    In defense scenarios:
    - False Negative = Catastrophic (missed infiltrator)
    - False Positive = Resource waste (unnecessary alerts)
    
    Therefore we tune for lower false negatives at cost of some false positives.
    """
    
    # Model Configuration
    model_size: Literal['n', 's', 'm', 'l', 'x'] = 'm'
    """
    YOLOv8 model size:
    - 'n' (nano): Fastest, least accurate (~2ms per frame)
    - 's' (small): Balanced for edge devices
    - 'm' (medium): Good balance - RECOMMENDED for most deployments
    - 'l' (large): Higher accuracy, slower
    - 'x' (xlarge): Maximum accuracy, requires high-end GPU
    
    Real scenario: Border posts use 'm', central command uses 'x'
    """
    
    # Detection Thresholds
    confidence_threshold: float = 0.45
    """
    Minimum confidence to consider a detection valid.
    
    Why 0.45 and not 0.5?
    - At night or in fog, even valid detections have lower confidence
    - Better to alert human operator about 45% confidence detection
      than miss a potential threat
    - Downstream layers will further validate
    
    In practice:
    - Daytime/clear: 0.5-0.6
    - Night/fog: 0.4-0.45
    - Thermal cameras: 0.35-0.4
    """
    
    iou_threshold: float = 0.45
    """
    Intersection over Union threshold for Non-Maximum Suppression.
    
    What this means:
    When multiple bounding boxes overlap, we need to decide:
    "Are these two boxes detecting the same person, or two different people?"
    
    Example scenario:
    Two people standing very close together (like in a group).
    - Too high (0.7): Might merge them into one detection
    - Too low (0.3): Might create duplicate boxes for same person
    - 0.45: Good balance
    """
    
    # Processing Configuration
    input_size: tuple[int, int] = (640, 640)
    """
    Size to which input frames are resized before detection.
    
    Trade-off:
    - Larger (1280x1280): Better for distant objects, slower
    - Smaller (320x320): Faster, might miss small/distant persons
    - 640x640: Industry standard, good balance
    
    Real scenario:
    - Long-range border cameras: 1280x1280
    - Indoor/close-range: 640x640
    - Drone feeds (power constrained): 416x416
    """
    
    max_detections: int = 100
    """
    Maximum number of persons to detect in single frame.
    
    Why limit?
    - Prevents system overload during crowd scenarios
    - In border surveillance, >100 people is extremely rare
    - If hit, triggers different alert (crowd detection)
    
    Exceptions:
    - Crowd monitoring mode: 500+
    - Border patrol: 50-100
    - Restricted zones: 10-20
    """
    
    # Device Configuration
    device: Literal['cpu', 'cuda', 'mps'] = 'cuda'
    """
    Processing device:
    - 'cuda': NVIDIA GPU (recommended for production)
    - 'cpu': Fallback, much slower (~10x)
    - 'mps': Apple Silicon (M1/M2)
    
    Real deployment:
    - Field stations: NVIDIA Jetson (embedded CUDA)
    - Central command: High-end GPU servers
    - Mobile units: CPU with optimized models
    """
    
    half_precision: bool = True
    """
    Use FP16 (half precision) instead of FP32.
    
    Benefits:
    - 2x faster inference
    - 2x less memory
    - Minimal accuracy loss (<1%)
    
    When to disable:
    - CPU inference (not supported)
    - Debugging accuracy issues
    - Older GPUs without Tensor Cores
    """
    
    # Operational Parameters
    min_detection_area: int = 400
    """
    Minimum bounding box area (in pixels) to consider valid.
    
    Why filter small detections?
    - Very distant persons appear as tiny boxes
    - Might be false positives (birds, debris)
    - If operationally relevant, zoom cameras should be used
    
    Example:
    - 20x20 pixel box = 400 pixels² = threshold
    - Smaller than this is usually noise
    
    Exception:
    - High-resolution cameras: increase to 900-1600
    - Thermal cameras: decrease to 200-300
    """
    
    skip_frames: int = 0
    """
    Number of frames to skip between detections.
    
    Use case:
    - 0: Process every frame (real-time critical scenarios)
    - 1: Process every other frame (2x speedup, minimal loss)
    - 2-4: Longer-term monitoring where second-by-second isn't critical
    
    Defense scenario:
    - Active threat: 0 (every frame)
    - Routine monitoring: 1-2
    - Historical analysis: 5-10
    """
    
    # Edge Case Handling
    low_light_boost: bool = True
    """
    Apply preprocessing for low-light conditions.
    
    Techniques:
    - Histogram equalization
    - Adaptive brightness adjustment
    - Noise reduction
    
    Trade-off:
    - Improves night detection
    - Adds ~2ms processing time
    - May introduce artifacts
    """
    
    occlusion_handling: bool = True
    """
    Enhanced detection for partially occluded persons.
    
    Real scenario:
    - Person hiding behind tree
    - Person crouching behind wall
    - Person in dense foliage
    
    Method:
    - Lower confidence threshold for partial detections
    - Track incomplete bounding boxes
    - Correlate with temporal data
    """
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        
        if self.min_detection_area < 0:
            raise ValueError("min_detection_area must be non-negative")
        
        if self.half_precision and self.device == 'cpu':
            # FP16 not supported on CPU, auto-disable
            self.half_precision = False
            print("Warning: half_precision disabled for CPU inference")
