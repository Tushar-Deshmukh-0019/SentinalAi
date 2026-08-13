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
        appearance_feature: [128] appearance vector (DeepSORT, optional)
        appearance_gallery: List of recent appearance vectors (DeepSORT, optional)
        appearance_confidence: Inverse of appearance distance (0-1 scale, DeepSORT)
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
    appearance_feature: Optional[np.ndarray] = None  # [128] feature vector
    appearance_gallery: Optional[List[np.ndarray]] = None  # List of recent features
    appearance_confidence: Optional[float] = None  # 0-1, confidence in appearance match
    
    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = []
        if self.appearance_gallery is None:
            self.appearance_gallery = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
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
        
        # Add appearance info if available
        if self.appearance_feature is not None:
            result['appearance_feature'] = self.appearance_feature.tolist() if isinstance(self.appearance_feature, np.ndarray) else self.appearance_feature
        if self.appearance_gallery:
            result['appearance_gallery_size'] = len(self.appearance_gallery)
        if self.appearance_confidence is not None:
            result['appearance_confidence'] = float(self.appearance_confidence)
        
        return result
    
    def get_appearance_distance(self, other_feature: np.ndarray) -> Optional[float]:
        """Compute distance to another appearance feature
        
        Args:
            other_feature: [128] feature vector (should be L2 normalized)
            
        Returns:
            Distance value (0-2 for cosine), or None if no feature
        """
        if self.appearance_feature is None:
            return None
        
        # Cosine distance
        sim = np.dot(self.appearance_feature, other_feature)
        return float(1.0 - sim)
    
    def get_gallery(self) -> List[np.ndarray]:
        """Get appearance gallery (list of recent features)
        
        Returns:
            List of feature vectors, empty if not available
        """
        return self.appearance_gallery.copy() if self.appearance_gallery else []


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
                 class_mapping: Optional[Dict[int, str]] = None,
                 tracker_type: str = 'bytetrack'):
        """Initialize tracking pipeline
        
        Args:
            frame_rate: Video frame rate (for timeout calculations)
            track_buffer: Frames to keep lost tracks
            class_mapping: Optional dict mapping class_id to class_name
            tracker_type: 'bytetrack' or 'deepsort'
            
        Raises:
            ValueError: If tracker_type invalid
        """
        if tracker_type == 'deepsort':
            try:
                from ai.tracking.deepsort import DeepSORTTracker
                self.tracker = DeepSORTTracker(
                    config={
                        'frame_rate': frame_rate,
                        'track_buffer': track_buffer,
                    }
                )
                self.tracker_type = 'deepsort'
                logger.info("TrackingPipeline initialized with DeepSORT")
            except ImportError:
                logger.warning("DeepSORT not available, falling back to ByteTrack")
                self.tracker = ByteTracker(frame_rate=frame_rate, track_buffer=track_buffer)
                self.tracker_type = 'bytetrack'
        else:
            self.tracker = ByteTracker(frame_rate=frame_rate, track_buffer=track_buffer)
            self.tracker_type = 'bytetrack'
        
        self.frame_rate = frame_rate
        self.class_mapping = class_mapping or {}
        
        # Track metadata for output enrichment
        self.track_class_ids: Dict[int, int] = {}  # track_id -> class_id
        self.track_trajectories: Dict[int, List[Tuple[float, float]]] = {}
        
        logger.info(f"TrackingPipeline initialized (frame_rate={frame_rate}, tracker={self.tracker_type})")
    
    def process(self, detections: Dict[str, Any], frame: Optional[np.ndarray] = None, 
                frame_id: Optional[int] = None) -> List[TrackOutput]:
        """Process detections and return tracked objects
        
        Args:
            detections: Dict with 'boxes', 'confs', optional 'class_ids'
            frame: Optional [H, W, 3] frame (required for DeepSORT)
            frame_id: Optional frame identifier (auto-incremented if not provided)
            
        Returns:
            List of TrackOutput objects for confirmed tracks
        """
        # Convert detections to tracker format
        detections_array, class_ids = DetectionProcessor.process_dict(detections)
        
        # Run tracker
        if self.tracker_type == 'deepsort':
            # DeepSORT requires frame
            if frame is None:
                logger.warning("DeepSORT requires frame, falling back to spatial matching only")
                frame = np.zeros((1, 1, 3), dtype=np.uint8)
            
            tracked_stracks = self.tracker.update(detections_array, frame)
        else:
            # ByteTrack
            tracked_stracks = self.tracker.update(detections_array)
        
        # Enrich with metadata
        output = []
        for i, track_data in enumerate(tracked_stracks):
            # Handle both DeepSORT and ByteTrack output formats
            if isinstance(track_data, dict):
                # Standard format from tracker
                track_id = track_data.get('track_id')
                bbox_xyxy = np.array(track_data.get('bbox_xyxy', track_data.get('bbox', [0, 0, 1, 1])))
                bbox_tlwh = np.array(track_data.get('bbox_tlwh', [0, 0, 1, 1]))
                confidence_val = track_data.get('confidence', 1.0)
                age_val = track_data.get('age', 0)
                hits_val = track_data.get('hits', 1)
                is_confirmed_val = track_data.get('is_confirmed', False)
            else:
                # Skip non-dict entries
                continue
            
            # Get class info if available
            class_id = class_ids[i] if class_ids is not None and i < len(class_ids) else None
            class_name = self.class_mapping.get(class_id) if class_id is not None else None
            
            # Store class mapping for future frames
            if class_id is not None:
                self.track_class_ids[track_id] = class_id
            else:
                class_id = self.track_class_ids.get(track_id)
            
            # Calculate center and size
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
            
            # Extract appearance info if available (DeepSORT)
            appearance_feature = None
            appearance_gallery = []
            appearance_confidence = None
            
            if self.tracker_type == 'deepsort' and hasattr(self.tracker, 'appearance_manager'):
                # Try to get appearance feature and gallery
                track_obj = None
                for t in self.tracker.tracked_tracks:
                    if t.track_id == track_id:
                        track_obj = t
                        break
                
                if track_obj is not None:
                    # Get mean feature
                    mean_feat = self.tracker.appearance_manager.get_mean_feature(track_id)
                    if mean_feat is not None:
                        appearance_feature = mean_feat
                        
                        # Compute confidence (inverse of distance, scaled to 0-1)
                        # Distance range [0, 2] -> confidence range [1, -1], clamp to [0, 1]
                        appearance_confidence = max(0.0, 1.0 - (
                            self.tracker.appearance_manager.compute_distance(mean_feat, track_id) / 2.0
                        ))
                    
                    # Get gallery
                    gallery = self.tracker.appearance_manager.get_gallery(track_id)
                    if gallery:
                        appearance_gallery = gallery[:10]  # Keep last 10 features
            
            # Create output object
            track_output = TrackOutput(
                track_id=track_id,
                bbox_xyxy=bbox_xyxy,
                bbox_tlwh=bbox_tlwh,
                confidence=confidence_val,
                class_id=class_id,
                class_name=class_name,
                age=age_val,
                hits=hits_val,
                is_confirmed=is_confirmed_val,
                is_tentative=not is_confirmed_val,
                position=(center_x, center_y),
                size=(width, height),
                trajectory=self.track_trajectories[track_id].copy(),
                appearance_feature=appearance_feature,
                appearance_gallery=appearance_gallery,
                appearance_confidence=appearance_confidence,
            )
            
            output.append(track_output)
        
        return output
    
    def process_array(self, detections: np.ndarray, frame: Optional[np.ndarray] = None) -> List[TrackOutput]:
        """Process raw detection array
        
        Args:
            detections: Nx5 array [x1, y1, x2, y2, confidence]
            frame: Optional [H, W, 3] frame (for DeepSORT)
            
        Returns:
            List of TrackOutput objects
        """
        return self.process({'boxes': detections[:, :4], 'confs': detections[:, 4]}, frame=frame)
    
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
        if hasattr(self.tracker, 'get_statistics'):
            tracker_stats = self.tracker.get_statistics()
        else:
            tracker_stats = {}
        
        return {
            **tracker_stats,
            'tracked_trajectories': len(self.track_trajectories),
            'tracker_type': self.tracker_type,
        }
    
    def reset(self):
        """Reset pipeline state"""
        if hasattr(self.tracker, 'reset'):
            self.tracker.reset()
        self.track_class_ids.clear()
        self.track_trajectories.clear()
        logger.info("TrackingPipeline reset")


class DeepSORTPipeline(TrackingPipeline):
    """DeepSORT-specific pipeline (convenience wrapper)
    
    Same interface as TrackingPipeline but always uses DeepSORT tracker.
    """
    
    def __init__(self, frame_rate: int = 30, track_buffer: int = 30,
                 class_mapping: Optional[Dict[int, str]] = None,
                 deepsort_config: Optional[Dict[str, Any]] = None):
        """Initialize DeepSORT pipeline
        
        Args:
            frame_rate: Video frame rate
            track_buffer: Frames to keep lost tracks
            class_mapping: Optional class mapping
            deepsort_config: Optional DeepSORT configuration dict
        """
        from ai.tracking.deepsort import DeepSORTTracker
        
        config = deepsort_config or {}
        config.setdefault('frame_rate', frame_rate)
        config.setdefault('track_buffer', track_buffer)
        
        self.tracker = DeepSORTTracker(config=config)
        self.tracker_type = 'deepsort'
        self.frame_rate = frame_rate
        self.class_mapping = class_mapping or {}
        self.track_class_ids = {}
        self.track_trajectories = {}
        
        logger.info(f"DeepSORTPipeline initialized (frame_rate={frame_rate})")


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
    
    print("=== Basic Pipeline Example (ByteTrack) ===")
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


def example_deepsort_pipeline():
    """DeepSORT pipeline usage example"""
    print("\n=== DeepSORT Pipeline Example ===")
    
    try:
        pipeline = DeepSORTPipeline(frame_rate=30)
        
        # Simulate frames with detections and dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = {
            'boxes': [[100, 100, 150, 180], [200, 150, 250, 250]],
            'confs': [0.95, 0.92]
        }
        
        # Process with DeepSORT
        tracks = pipeline.process(detections, frame=frame)
        confirmed = pipeline.get_confirmed_tracks(tracks)
        
        print(f"Processed {len(tracks)} tracks, {len(confirmed)} confirmed")
        
        for track in tracks:
            print(f"Track {track.track_id}: confidence={track.confidence:.2f}, "
                  f"appearance_conf={track.appearance_confidence}")
    
    except ImportError:
        print("DeepSORT not available in this environment")


if __name__ == '__main__':
    example_basic_pipeline()
    example_with_classes()
    example_deepsort_pipeline()
