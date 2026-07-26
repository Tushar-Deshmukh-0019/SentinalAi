"""
Processing Pipelines

Infrastructure for multi-camera, real-time surveillance processing.

Components:
- Camera feed management (multiple streams)
- Frame buffering and queuing
- Detection pipeline orchestration
- Load balancing and optimization
- Performance monitoring

Why this matters:
Detection is only useful if we can process all cameras in real-time.
This infrastructure makes the detection system operational at scale.
"""

__version__ = '0.1.0'
