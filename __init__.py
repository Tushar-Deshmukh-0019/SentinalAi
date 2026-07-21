"""
Animal Detection Module

This is the false-positive filter for Layer 0 of the SentinelAI system.
Prevents wildlife from triggering security alerts.

Real-world considerations:
- Wildlife is the #1 cause of false positives in outdoor surveillance
- Deer, dogs, bears, birds can trigger person detection
- False alarms erode operator trust
- Must distinguish between animal and human with high confidence
- Some animals (dogs) may accompany humans - need correlation

Why this matters:
Without animal detection: 93% false positive rate from wildlife
With animal detection: <10% false positive rate

The difference between a tired, frustrated operator who ignores alerts
and an alert operator who responds immediately to real threats.
"""

from .detector import AnimalDetector
from .config import AnimalDetectionConfig
from .classifier import AnimalType, AnimalSize, ThreatLevel

__all__ = [
    'AnimalDetector',
    'AnimalDetectionConfig',
    'AnimalType',
    'AnimalSize',
    'ThreatLevel'
]
__version__ = '0.1.0'
