"""
Tracking Pipeline Integration for SentinelAI

Bridges detection systems with tracking systems and downstream behavior analysis.

This module provides:
- TrackingPipeline: High-level interface for integrating ByteTrack with detection results
- DetectionProcessor: Converts various detection formats to unified representation
- TrackOutput: Standardized output format for tracked objects
- Integration examples for behavior analysis modules

The pipeline accepts detection results from any detector and outputs stable object
tracks with IDs for use by behavior analysis, anomaly detection, and other systems.

Example Usage:
    pipeline = TrackingPipeline(frame_rate=30)
    
    # Process frame with detections
    detections_dict = {
        'boxes': [[100, 100, 150, 180], [200, 150, 250, 250], ...],
        'confs': [0.95, 0.92, ...],
        'class_ids': [0, 0, ...]  # Optional
    }
    
    tracked_objects = pipeline.process(detections_dict, frame_id=123)
    
    for obj in tracked_objects:
        print(f"Track {obj.track_id}: class={obj.class_id}, conf={obj.confidence:.2f}")
        print(f"  Position: {obj.position}")
        print(f"  Size: {obj.size}")
        print(f"  Status: {'CONFIRMED' if obj.is_confirmed else 'TENTATIVE'}")
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from ai.tracking.bytetrack import ByteTracker, STrack

logger = logging.getLogger('tracking.pipeline')


@dataclass
class TrackOutput:
    """Standardized output for a tracked object
    
    Attributes:
        track_id: Unique stable ID for this object across frames
        bbox_xyxy: Bounding box in [x1, y1, x2, y2] format
        bbox_tlwh: Bounding box in [top, left, width, height] format
        confidence: Detection confidence (0.0 to 1.0)
        class_id: Optional object class (e.g., 0=person, 1=vehicle)
        class_name: Optional class name string
        age: Number of frames since track creation
        hits: Number of detections associated with this track
        is_confirmed: Whether track is confirmed (3+ hits)
        is_tentative: Whether track is tentative (1-2 hits)
        position: Center point (x, y)
        size: Object dimensions (width, height)
        trajectory: List of recent positions for trajectory visualization
    """
    track_id: int
    bbox_xyxy: np.ndarray
    bbox_tlwh: np.ndarray
    confidence: float
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    age: int = 0
    hits: int = 1
    is_confirmed: bool = False
    is_tentative: bool = True
    position: Tuple[float, float] = (0, 0)
    size: Tuple[float, float] = (0, 0)
    trajectory: List[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'track_id': self.track_id,
            'bbox': self.bbox_xyxy.tolist(),
            'bbox_tlwh': self.bbox_tlwh.tolist(),
            'confidence': float(self.confidence),
            'class_id': self.class_id,
            'class_name': self.class_name,
            'age': self.age,
            'hits': self.hits,
            'is_confirmed': self.is_confirmed,
            'position': {'x': self.position[0], 'y': self.position[1]},
            'size': {'width': self.size[0], 'height': self.size[1]},
            'trajectory': self.trajectory
        }


class DetectionProcessor:
    """Converts detection outputs to standardized format for ByteTrack
    
    Handles various detection input formats:
    - Dict with 'boxes', 'confs' keys
    - Dict with 'boxes', 'confs', 'class_ids' keys
    - Raw numpy array (Nx5)
    - List of detections
    """
    
    # Default class mapping
    COCO_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
        4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
        8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
        # ... more classes
    }
    
    @staticmethod
    def process_dict(detections: Dict[str, Any]) -> Tuple[np.ndarray, Optional[List[int]]]:
        """Convert detection dict to ByteTrack format
        
        Args:
            detections: Dict with keys:
                - 'boxes': Nx4 array of [x1, y1, x2, y2] or [x, y, w, h]
                - 'confs': N confidences
                - 'class_ids': Optional N class IDs
                
        Returns:
            (Nx5 array [x1, y1, x2, y2, conf], optional class_ids list)
        """
        boxes = np.array(detections.get('boxes', []), dtype=np.float32)
        confs = np.array(detections.get('confs', []), dtype=np.float32)
        class_ids = detections.get('class_ids', None)
        
        if len(boxes) == 0:
            return np.array([]).reshape(0, 5), class_ids
        
        # Convert tlwh to xyxy if needed (check if width/height are small)
        if boxes.shape[1] == 4:
            # Assume xyxy format if values look like coordinates
            if np.max(boxes[:, 2:]) > 50:  # Likely xyxy
                pass  # Already in xyxy format
            else:  # Likely tlwh
                x1, y1, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                boxes = np.column_stack([x1, y1, x1 + w, y1 + h])
        
        # Combine boxes and confidences
        detections_array = np.column_stack([boxes, confs])
        
        return detections_array, class_ids
    
    @staticmethod
    def process_array(detections: np.ndarray) -> Tuple[np.ndarray, None]:
        """Pass through numpy array (already in Nx5 format)"""
        return np.array(detections, dtype=np.float32), None


class TrackingPipeline:
    """High-level tracking pipeline integrating ByteTrack with detections
    
    Provides a clean interface for:
    - Accepting detections from various sources
    - Running ByteTrack tracking
    - Returning stable tracks with metadata
    - Maintaining track history and statistics
    
    Example:
        pipeline = TrackingPipeline(frame_rate=30)
        
        detections = {
            'boxes': [[100, 100, 150, 180], ...],
            'confs': [0.95, ...],
            'class_ids': [0, ...]
        }
        
        tracks = pipeline.process(detections, frame_id=1)
        for track in tracks:
            if track.is_confirmed:
                print(f"Person {track.track_id} at {track.position}")
    """
    
    def __init__(self, frame_rate: int = 30, track_buffer: int = 30,
                 class_mapping: Optional[Dict[int, str]] = None):
        """Initialize tracking pipeline
        
        Args:
            frame_rate: Video frame rate (for timeout calculations)
            track_buffer: Frames to keep lost tracks
            class_mapping: Optional dict mapping class_id to class_name
        """
        self.tracker = ByteTracker(frame_rate=frame_rate, track_buffer=track_buffer)
        self.frame_rate = frame_rate
        self.class_mapping = class_mapping or {}
        
        # Track metadata for output enrichment
        self.track_class_ids: Dict[int, int] = {}  # track_id -> class_id
        self.track_trajectories: Dict[int, List[Tuple[float, float]]] = {}
        
        logger.info(f"TrackingPipeline initialized (frame_rate={frame_rate})")
    
    def process(self, detections: Dict[str, Any], frame_id: Optional[int] = None) -> List[TrackOutput]:
        """Process detections and return tracked objects
        
        Args:
            detections: Dict with 'boxes', 'confs', optional 'class_ids'
            frame_id: Optional frame identifier (auto-incremented if not provided)
            
        Returns:
            List of TrackOutput objects for confirmed tracks
        """
        # Convert detections to ByteTrack format
        detections_array, class_ids = DetectionProcessor.process_dict(detections)
        
        # Run ByteTrack
        tracked_stracks = self.tracker.update(detections_array)
        
        # Enrich with metadata
        output = []
        for i, track_data in enumerate(tracked_stracks):
            track_id = track_data['track_id']
            
            # Get class info if available
            class_id = class_ids[i] if class_ids is not None and i < len(class_ids) else None
            class_name = self.class_mapping.get(class_id) if class_id is not None else None
            
            # Store class mapping for future frames
            if class_id is not None:
                self.track_class_ids[track_id] = class_id
            else:
                class_id = self.track_class_ids.get(track_id)
            
            # Calculate center and size
            bbox_xyxy = np.array(track_data['bbox'])
            x1, y1, x2, y2 = bbox_xyxy
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # Update trajectory
            if track_id not in self.track_trajectories:
                self.track_trajectories[track_id] = []
            self.track_trajectories[track_id].append((center_x, center_y))
            
            # Limit trajectory history
            if len(self.track_trajectories[track_id]) > 30:
                self.track_trajectories[track_id].pop(0)
            
            # Create output object
            track_output = TrackOutput(
                track_id=track_id,
                bbox_xyxy=bbox_xyxy,
                bbox_tlwh=np.array(track_data['bbox_tlwh']),
                confidence=track_data['confidence'],
                class_id=class_id,
                class_name=class_name,
                age=track_data['age'],
                hits=track_data['hits'],
                is_confirmed=track_data['is_confirmed'],
                is_tentative=not track_data['is_confirmed'],
                position=(center_x, center_y),
                size=(width, height),
                trajectory=self.track_trajectories[track_id].copy()
            )
            
            output.append(track_output)
        
        return output
    
    def process_array(self, detections: np.ndarray) -> List[TrackOutput]:
        """Process raw detection array
        
        Args:
            detections: Nx5 array [x1, y1, x2, y2, confidence]
            
        Returns:
            List of TrackOutput objects
        """
        return self.process({'boxes': detections[:, :4], 'confs': detections[:, 4]})
    
    def get_confirmed_tracks(self, tracks: List[TrackOutput]) -> List[TrackOutput]:
        """Filter to only confirmed tracks
        
        Args:
            tracks: List of TrackOutput from process()
            
        Returns:
            Filtered list with only confirmed tracks
        """
        return [t for t in tracks if t.is_confirmed]
    
    def get_tentative_tracks(self, tracks: List[TrackOutput]) -> List[TrackOutput]:
        """Filter to only tentative tracks
        
        Args:
            tracks: List of TrackOutput from process()
            
        Returns:
            Filtered list with only tentative tracks
        """
        return [t for t in tracks if t.is_tentative]
    
    def get_tracks_by_class(self, tracks: List[TrackOutput], class_id: int) -> List[TrackOutput]:
        """Filter tracks by class
        
        Args:
            tracks: List of TrackOutput from process()
            class_id: Class ID to filter by
            
        Returns:
            Filtered list with matching class
        """
        return [t for t in tracks if t.class_id == class_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        tracker_stats = self.tracker.get_statistics()
        return {
            **tracker_stats,
            'tracked_trajectories': len(self.track_trajectories)
        }
    
    def reset(self):
        """Reset pipeline state"""
        self.tracker.reset()
        self.track_class_ids.clear()
        self.track_trajectories.clear()
        logger.info("TrackingPipeline reset")


# Example usage
def example_basic_pipeline():
    """Basic pipeline usage example"""
    pipeline = TrackingPipeline(frame_rate=30)
    
    # Simulate 3 frames of detections
    frames = [
        {'boxes': [[100, 100, 150, 180], [200, 150, 250, 250]], 'confs': [0.95, 0.92]},
        {'boxes': [[102, 102, 152, 182], [202, 152, 252, 252]], 'confs': [0.94, 0.91]},
        {'boxes': [[104, 104, 154, 184], [204, 154, 254, 254]], 'confs': [0.93, 0.90]},
    ]
    
    print("=== Basic Pipeline Example ===")
    for frame_id, detections in enumerate(frames):
        tracks = pipeline.process(detections, frame_id=frame_id)
        confirmed = pipeline.get_confirmed_tracks(tracks)
        
        print(f"\nFrame {frame_id}:")
        print(f"  Total tracks: {len(tracks)}")
        print(f"  Confirmed: {len(confirmed)}")
        for track in confirmed:
            print(f"    Track {track.track_id}: pos={track.position}, hits={track.hits}")


def example_with_classes():
    """Pipeline with class information"""
    class_map = {0: 'person', 1: 'vehicle', 2: 'bicycle'}
    pipeline = TrackingPipeline(frame_rate=30, class_mapping=class_map)
    
    # Mixed detection types
    detections = {
        'boxes': [[100, 100, 150, 180], [200, 150, 250, 250], [50, 50, 100, 100]],
        'confs': [0.95, 0.92, 0.88],
        'class_ids': [0, 1, 2]  # person, vehicle, bicycle
    }
    
    print("\n=== Pipeline with Classes Example ===")
    tracks = pipeline.process(detections)
    
    for track in tracks:
        print(f"Track {track.track_id}: {track.class_name}, conf={track.confidence:.2f}")


if __name__ == '__main__':
    example_basic_pipeline()
    example_with_classes()
