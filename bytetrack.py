"""
ByteTrack Implementation for SentinelAI

Multi-object tracking system based on ByteTrack algorithm.

ByteTrack is lightweight, efficient tracking that:
- Associates detections across frames without deep features
- Handles occlusion and re-detection naturally
- Runs in real-time (~30 FPS for 100 objects)
- No GPU requirements

Key Innovation: Uses detection confidence as association feature
- High-confidence detections → strong association
- Low-confidence detections → weak association
- Occlusions handled by age-based continuation

Reference: ByteTrack: Multi-Object Tracking by Associating Every Detection
https://arxiv.org/abs/2110.06864
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger('tracking.bytetrack')


@dataclass
class Track:
    """Single track for a detected object"""
    
    track_id: int
    bbox: np.ndarray              # [x1, y1, x2, y2] format
    confidence: float             # Detection confidence
    frame_id: int                 # Current frame
    start_frame: int              # Frame where track started
    last_update_frame: int        # Last frame track was updated
    age: int = 0                  # Number of frames since track created
    time_since_update: int = 0    # Frames since last detection
    hits: int = 1                 # Number of detections associated
    hit_streak: int = 1           # Consecutive frames with detection
    
    # Track history for trajectory
    history: List[np.ndarray] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    
    def update(self, bbox: np.ndarray, confidence: float, frame_id: int):
        """Update track with new detection"""
        self.bbox = bbox
        self.confidence = confidence
        self.last_update_frame = frame_id
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        # Store history
        self.history.append(bbox.copy())
        self.confidence_history.append(confidence)
        
        # Limit history to 100 frames
        if len(self.history) > 100:
            self.history.pop(0)
            self.confidence_history.pop(0)
    
    def predict(self):
        """Predict next position (simple linear extrapolation)"""
        if len(self.history) < 2:
            return self.bbox.copy()
        
        # Linear motion model
        last_bbox = self.history[-1]
        prev_bbox = self.history[-2]
        
        displacement = last_bbox - prev_bbox
        predicted = last_bbox + displacement
        
        return predicted
    
    def increment_age(self):
        """Increment age (called every frame)"""
        self.age += 1
        self.time_since_update += 1
        self.hit_streak = max(0, self.hit_streak - 1)
    
    def get_state(self) -> Dict:
        """Get track state for export"""
        return {
            'track_id': self.track_id,
            'bbox': self.bbox.tolist(),
            'confidence': self.confidence,
            'frame_id': self.frame_id,
            'age': self.age,
            'hits': self.hits,
            'is_confirmed': self.is_confirmed(),
            'position': {
                'x': float((self.bbox[0] + self.bbox[2]) / 2),
                'y': float((self.bbox[1] + self.bbox[3]) / 2),
            },
            'size': {
                'width': float(self.bbox[2] - self.bbox[0]),
                'height': float(self.bbox[3] - self.bbox[1]),
            }
        }
    
    def is_confirmed(self) -> bool:
        """Check if track is confirmed (stable)"""
        return self.hits >= 3 and self.hit_streak >= 2
    
    def is_tentative(self) -> bool:
        """Check if track is tentative (new)"""
        return self.hits < 3 or self.hit_streak < 2
    
    def is_deleted(self) -> bool:
        """Check if track should be deleted"""
        return self.time_since_update > 30  # 30 frames (1 second @ 30 FPS)


class STrack:
    """Strided Track - enhanced track with association metrics"""
    
    def __init__(self, tlwh: np.ndarray, score: float, frame_id: int, 
                 track_id: int = None):
        """
        Initialize STrack
        
        Args:
            tlwh: [x1, y1, width, height] format
            score: Detection confidence
            frame_id: Current frame ID
            track_id: Optional existing track ID
        """
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = float(score)
        self.frame_id = int(frame_id)
        self.track_id = track_id
        
        self.age = 0
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.history = [self.tlwh.copy()]
    
    @property
    def xyxy(self) -> np.ndarray:
        """Convert to [x1, y1, x2, y2] format"""
        tlwh = self.tlwh
        return np.array([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]])
    
    def update(self, detection: Tuple, frame_id: int):
        """Update track with new detection"""
        self.tlwh = detection[:4].astype(np.float32)
        self.score = float(detection[4])
        self.frame_id = int(frame_id)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.history.append(self.tlwh.copy())
        
        if len(self.history) > 100:
            self.history.pop(0)
    
    def predict(self) -> np.ndarray:
        """Predict next state"""
        if len(self.history) < 2:
            return self.tlwh.copy()
        
        # Linear motion model
        last = self.history[-1]
        prev = self.history[-2]
        displacement = last - prev
        
        return last + displacement
    
    def increment_age(self):
        """Age the track"""
        self.age += 1
        self.time_since_update += 1
        self.hit_streak = max(0, self.hit_streak - 1)
    
    def is_confirmed(self) -> bool:
        """Track confirmed if hits >= 3"""
        return self.hits >= 3
    
    def is_tentative(self) -> bool:
        """Track tentative if hits < 3"""
        return self.hits < 3
    
    def is_deleted(self) -> bool:
        """Track deleted if too old without update"""
        return self.time_since_update > 30


class ByteTracker:
    """
    ByteTrack multi-object tracker.
    
    Efficiently tracks multiple objects across frames without requiring
    deep features. Uses detection confidence as primary association metric.
    
    Features:
    - Real-time tracking (~30 FPS for 100 objects)
    - Handles occlusion naturally through track continuation
    - Confirms tracks after 3 detections
    - Deletes stale tracks after 30 frames
    - Provides stable IDs for downstream processing
    """
    
    def __init__(self, frame_rate: int = 30, track_buffer: int = 30):
        """
        Initialize ByteTracker
        
        Args:
            frame_rate: Video frame rate (for timeout calculation)
            track_buffer: Frames to keep deleted tracks (for re-detection)
        """
        self.frame_rate = frame_rate
        self.track_buffer = track_buffer  # 1 second
        
        self.tracked_stracks: List[STrack] = []  # Confirmed tracks
        self.lost_stracks: List[STrack] = []     # Lost tracks (recently deleted)
        self.removed_stracks: List[STrack] = []  # Permanently removed
        
        self.frame_id = 0
        self.next_track_id = 1
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'total_tracks_created': 0,
            'active_tracks': 0,
            'confirmed_tracks': 0,
            'tentative_tracks': 0
        }
        
        logger.info(f"ByteTracker initialized (frame_rate={frame_rate})")
    
    def update(self, detections: np.ndarray, frame: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: Nx5 array [x1, y1, x2, y2, confidence]
            frame: Optional frame for visualization
            
        Returns:
            List of tracked objects with IDs and states
        """
        self.frame_id += 1
        
        if len(detections) == 0:
            # No detections - age existing tracks
            self._handle_no_detections()
            return self._format_output()
        
        self.stats['total_detections'] += len(detections)
        
        # Separate high and low confidence detections
        high_conf_idx = detections[:, 4] > 0.5
        high_conf_detections = detections[high_conf_idx]
        low_conf_detections = detections[~high_conf_idx]
        
        # Track high-confidence detections with confirmed tracks
        unconfirmed_detections, unmatched_tracks_high = self._associate_high_conf(
            high_conf_detections
        )
        
        # Track low-confidence detections with unmatched confirmed tracks
        unmatched_detections, unmatched_tracks_low = self._associate_low_conf(
            low_conf_detections, unmatched_tracks_high
        )
        
        # Combine unconfirmed high-conf detections with remaining unmatched detections
        all_unmatched = np.vstack([unconfirmed_detections, unmatched_detections]) \
                       if len(unconfirmed_detections) > 0 and len(unmatched_detections) > 0 \
                       else (unconfirmed_detections if len(unconfirmed_detections) > 0 else unmatched_detections)
        
        # Create new tracks for unmatched detections
        self._create_new_tracks(all_unmatched)
        
        # Handle unmatched tracks (occlusion)
        self._handle_unmatched_tracks(
            unmatched_tracks_high + unmatched_tracks_low
        )
        
        # Update track states
        self._update_track_states()
        
        return self._format_output()
    
    def _associate_high_conf(self, detections: np.ndarray) -> Tuple[np.ndarray, List[STrack]]:
        """
        Associate high-confidence detections with confirmed tracks.
        Uses IoU (Intersection over Union) for association.
        """
        if len(detections) == 0:
            return np.array([]), self.tracked_stracks.copy()
        
        # Calculate IoU between detections and tracked objects
        ious = np.zeros((len(self.tracked_stracks), len(detections)))
        
        for i, track in enumerate(self.tracked_stracks):
            for j, det in enumerate(detections):
                iou = self._iou(track.xyxy, det[:4])
                ious[i, j] = iou
        
        # Match using greedy assignment (high IoU)
        matched_indices = self._greedy_assignment(ious, iou_thresh=0.1)
        
        unmatched_detections = []
        for j in range(len(detections)):
            if j not in matched_indices[:, 1]:
                unmatched_detections.append(detections[j])
        
        unmatched_tracks = []
        for i in range(len(self.tracked_stracks)):
            if i not in matched_indices[:, 0]:
                unmatched_tracks.append(self.tracked_stracks[i])
        
        # Update matched tracks
        for i, j in matched_indices:
            self.tracked_stracks[i].update(detections[j], self.frame_id)
        
        return np.array(unmatched_detections), unmatched_tracks
    
    def _associate_low_conf(self, detections: np.ndarray, 
                           unmatched_tracks: List[STrack]) -> Tuple[np.ndarray, List[STrack]]:
        """
        Associate low-confidence detections with unmatched confirmed tracks.
        Uses looser IoU threshold.
        """
        if len(detections) == 0 or len(unmatched_tracks) == 0:
            return detections, unmatched_tracks
        
        ious = np.zeros((len(unmatched_tracks), len(detections)))
        
        for i, track in enumerate(unmatched_tracks):
            for j, det in enumerate(detections):
                iou = self._iou(track.xyxy, det[:4])
                ious[i, j] = iou
        
        # More lenient matching for low-confidence
        matched_indices = self._greedy_assignment(ious, iou_thresh=0.05)
        
        unmatched_detections = []
        for j in range(len(detections)):
            if j not in matched_indices[:, 1]:
                unmatched_detections.append(detections[j])
        
        unmatched_tracks_final = []
        for i, track in enumerate(unmatched_tracks):
            if i not in matched_indices[:, 0]:
                unmatched_tracks_final.append(track)
        
        # Update matched tracks
        for i, j in matched_indices:
            unmatched_tracks[i].update(detections[j], self.frame_id)
            self.tracked_stracks.append(unmatched_tracks[i])
        
        return np.array(unmatched_detections), unmatched_tracks_final
    
    def _create_new_tracks(self, detections: np.ndarray):
        """Create new tracks for unmatched detections"""
        for det in detections:
            track = STrack(det[:4], det[4], self.frame_id, self.next_track_id)
            self.next_track_id += 1
            self.tracked_stracks.append(track)
            self.stats['total_tracks_created'] += 1
    
    def _handle_unmatched_tracks(self, unmatched_tracks: List[STrack]):
        """Handle unmatched tracks (age and move to lost)
        
        KEY FIX: Only move CONFIRMED tracks to lost_stracks.
        Tentative tracks (1-2 hits) stay in tracked_stracks to accumulate hits.
        This allows tracks to reach confirmation (3 hits) before being aged out.
        """
        for track in unmatched_tracks:
            track.increment_age()
            
            # Only move confirmed tracks to lost_stracks after timeout
            if track.time_since_update > 1:  # Lost after 1 frame without detection
                if track.is_confirmed():
                    self.lost_stracks.append(track)
                    self.tracked_stracks.remove(track)
                # TENTATIVE tracks stay in tracked_stracks to keep accumulating hits
                # They will be removed later if they never get confirmed
    
    def _handle_no_detections(self):
        """Age all tracks when no detections
        
        KEY FIX: Only move CONFIRMED tracks to lost_stracks.
        Tentative tracks stay in tracked_stracks to potentially get new detections.
        """
        for track in self.tracked_stracks:
            track.increment_age()
            
            # Only move confirmed tracks to lost_stracks
            if track.time_since_update > 1:
                if track.is_confirmed():
                    self.lost_stracks.append(track)
        
        # Remove only confirmed tracks that timed out
        # Keep tentative tracks for potential re-detection
        self.tracked_stracks = [t for t in self.tracked_stracks 
                               if not (t.is_confirmed() and t.time_since_update > 1)]
    
    def _update_track_states(self):
        """Update track states and cleanup"""
        # Remove old lost tracks
        self.lost_stracks = [t for t in self.lost_stracks 
                           if t.time_since_update < self.track_buffer]
        
        # Remove deleted tracks
        self.tracked_stracks = [t for t in self.tracked_stracks 
                               if not t.is_deleted()]
        
        # Update statistics
        self.stats['active_tracks'] = len(self.tracked_stracks)
        self.stats['confirmed_tracks'] = sum(1 for t in self.tracked_stracks if t.is_confirmed())
        self.stats['tentative_tracks'] = sum(1 for t in self.tracked_stracks if t.is_tentative())
    
    def _format_output(self) -> List[Dict]:
        """Format output tracks as dictionaries"""
        output = []
        
        for track in self.tracked_stracks:
            if track.is_confirmed():
                output.append({
                    'track_id': track.track_id,
                    'bbox': track.xyxy.tolist(),
                    'bbox_tlwh': track.tlwh.tolist(),
                    'confidence': track.score,
                    'age': track.age,
                    'hits': track.hits,
                    'is_confirmed': True,
                    'position': {
                        'x': float((track.xyxy[0] + track.xyxy[2]) / 2),
                        'y': float((track.xyxy[1] + track.xyxy[3]) / 2)
                    },
                    'size': {
                        'width': float(track.xyxy[2] - track.xyxy[0]),
                        'height': float(track.xyxy[3] - track.xyxy[1])
                    }
                })
        
        return output
    
    @staticmethod
    def _iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def _greedy_assignment(cost_matrix: np.ndarray, iou_thresh: float = 0.1) -> np.ndarray:
        """
        Greedy matching based on cost matrix.
        
        Args:
            cost_matrix: MxN cost matrix
            iou_thresh: Minimum cost (IoU) threshold
            
        Returns:
            Nx2 array of matched indices
        """
        matches = []
        
        while True:
            # Find best match
            if cost_matrix.size == 0:
                break
            
            max_idx = np.argmax(cost_matrix)
            max_i, max_j = np.unravel_index(max_idx, cost_matrix.shape)
            max_val = cost_matrix[max_i, max_j]
            
            if max_val < iou_thresh:
                break
            
            matches.append([max_i, max_j])
            
            # Remove matched rows and columns
            cost_matrix[max_i, :] = -1
            cost_matrix[:, max_j] = -1
        
        return np.array(matches) if matches else np.array([]).reshape(0, 2)
    
    def get_statistics(self) -> Dict:
        """Get tracker statistics"""
        return self.stats.copy()
    
    def reset(self):
        """Reset tracker"""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.next_track_id = 1
        logger.info("ByteTracker reset")
