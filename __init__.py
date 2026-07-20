"""
Vehicle Detection Module

This is part of Layer 0 of the SentinelAI system.
Detects and classifies vehicles in surveillance feeds.

Real-world considerations:
- Vehicles often accompany personnel (patrol vehicles, infiltration vehicles)
- Vehicle type indicates threat level (civilian vs. military)
- Vehicle size matters (motorcycle vs. truck has different implications)
- Unauthorized vehicles are immediate alerts
- Vehicle-person correlation is key evidence

Why this matters:
3 persons walking vs. 3 persons + truck = completely different threat assessment.
The presence, type, and authorization of a vehicle changes everything.
"""

from .detector import VehicleDetector
from .config import VehicleDetectionConfig
from .classifier import VehicleType, VehicleSize

__all__ = [
    'VehicleDetector', 
    'VehicleDetectionConfig',
    'VehicleType',
    'VehicleSize'
]
__version__ = '0.1.0'
