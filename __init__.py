"""
Object Detection Module

Completes Layer 0 by detecting objects carried by persons or left unattended.

Real-world considerations:
- Objects change threat assessment (backpack vs. briefcase)
- Abandoned objects are critical threats (potential IED)
- Object-person association is key intelligence
- Size and type matter (large backpack vs. small purse)
- Some objects are weapons (require immediate response)

Why this matters:
A person walking through a checkpoint is different from:
- Person with large backpack (requires inspection)
- Person with briefcase (normal business)
- Person with gun-shaped object (immediate threat)

Context transforms detection into intelligence.
"""

from .detector import ObjectDetector
from .config import ObjectDetectionConfig
from .classifier import ObjectType, ObjectSize, RiskLevel, ObjectCharacteristics

__all__ = [
    'ObjectDetector',
    'ObjectDetectionConfig',
    'ObjectType',
    'ObjectSize',
    'RiskLevel',
    'ObjectCharacteristics'
]
__version__ = '0.1.0'
