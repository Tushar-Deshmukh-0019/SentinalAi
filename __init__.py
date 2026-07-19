"""
Person Detection Module

This is Layer 0 of the SentinelAI system.
Foundation module that detects human presence in surveillance feeds.

Real-world considerations:
- Must work in varying light conditions (day, night, twilight)
- Must handle partial occlusion (person behind objects)
- Must distinguish humans from animals
- Must provide confidence scores for downstream analysis
- Must be fast enough for real-time processing (>30 FPS)

Why this matters:
Without reliable person detection, all higher-level analysis
(behavior, threat scoring, identity) becomes meaningless.
"""

from .detector import PersonDetector
from .config import PersonDetectionConfig

__all__ = ['PersonDetector', 'PersonDetectionConfig']
__version__ = '0.1.0'
