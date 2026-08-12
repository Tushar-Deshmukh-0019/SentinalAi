"""
Tracking Module for SentinelAI

Multi-object tracking system with:
- ByteTrack for efficient multi-object tracking
- DeepSORT for person re-identification
- Track persistence and ID management
- Cross-camera tracking coordination

Usage:
    from ai.tracking.bytetrack import ByteTracker
    
    tracker = ByteTracker(frame_rate=30)
    detections = detector.detect(frame)
    
    # Track detected objects
    tracked_objects = tracker.update(detections['boxes'], detections['confs'])
"""

from .bytetrack import ByteTracker, Track, STrack

__all__ = ['ByteTracker', 'Track', 'STrack']
