"""
Day 16: Track Management Integration - Complete Implementation

Purpose:
  Unified track lifecycle management across ByteTrack, DeepSORT, and Global IDs.
  Handles track creation, linking, merging, and cross-camera coordination.

Author: Kiro Development System
Date: Day 16
Status: Production Ready
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import uuid
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS - State Machine
# ============================================================================

class TrackState(Enum):
    """Track lifecycle state machine"""
    TENTATIVE = "tentative"      # 0-2 hits, not yet confirmed
    CONFIRMED = "confirmed"      # 3+ hits, validated person
    LOST = "lost"                # Unmatched > 1 frame, might return
    MERGED = "merged"            # Merged into another track
    DELETED = "deleted"          # Timed out or insufficient evidence


class AssignmentType(Enum):
    """How track was assigned (from Day 15 Global ID System)"""
    NEW = "new"                  # New person created
    LINK = "link"                # Linked to existing person
    RESOLVED = "resolved"        # Resolved from multiple candidates
    MERGED = "merged"            # Result of merge operation


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Detection:
    """A single person detection in one frame"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) - bounding box
    confidence: float                  # Detection confidence 0-1
    camera_id: int                     # Which camera
    frame_num: int                     # Frame number
    appearance_feature: Optional[List[float]] = None  # 128-dim CNN feature


@dataclass
class TrackInfo:
    """Complete information for one tracked person"""
    # Identity
    global_id: str                                        # UUID
    local_track_id: int                                   # ByteTrack ID
    
    # State Management
    state: TrackState = TrackState.TENTATIVE
    hits: int = 0                                         # Match count
    age: int = 0                                          # Frames since creation
    
    # Detection History
    detections: List[Detection] = field(default_factory=list)
    appearance_gallery: List[List[float]] = field(default_factory=list)
    
    # Assignment Info (from Day 15)
    assignment_type: AssignmentType = AssignmentType.NEW
    assignment_confidence: float = 0.0
    
    # Merging Info (Day 16 - NEW)
    merged_into: Optional[str] = None                     # If merged, where
    merged_from: List[str] = field(default_factory=list) # IDs merged into this
    
    # Spatial Info
    cameras_visited: Set[int] = field(default_factory=set)
    
    # Temporal Info
    created_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    last_seen_frame: int = 0
    
    # Statistics
    max_confidence: float = 0.0
    hits_history: List[int] = field(default_factory=list)


# ============================================================================
# TRACK MANAGER - Main Class
# ============================================================================

class TrackManager:
    """
    Unified track management across all tracking systems:
    - ByteTrack: Local tracking (single camera)
    - DeepSORT: Appearance-based matching (Day 14)
    - Global IDs: Cross-camera identification (Day 15)
    - Track Manager: Unified lifecycle & merging (Day 16)
    """
    
    def __init__(self, 
                 min_hits_confirm: int = 3,
                 max_age_lost: int = 70,
                 max_gallery_size: int = 100):
        """
        Initialize Track Manager
        
        Args:
            min_hits_confirm: Hits needed for CONFIRMED state (default: 3)
            max_age_lost: Delete LOST track after N frames (default: 70)
            max_gallery_size: Max appearance features per track (default: 100)
        """
        self.tracks: Dict[str, TrackInfo] = {}              # global_id → TrackInfo
        self.local_to_global: Dict[int, str] = {}           # local_id → global_id
        self.merged_mapping: Dict[str, str] = {}            # old_id → new_id
        
        self.min_hits_confirm = min_hits_confirm
        self.max_age_lost = max_age_lost
        self.max_gallery_size = max_gallery_size
        
        self.frame_count = 0
        
        # Statistics
        self.stats = {
            "tracks_created": 0,
            "tracks_confirmed": 0,
            "tracks_merged": 0,
            "tracks_deleted": 0,
            "total_detections": 0,
        }
        
        logger.info(f"TrackManager initialized: min_hits={min_hits_confirm}, "
                   f"max_age={max_age_lost}, gallery_size={max_gallery_size}")
    
    # ========================================================================
    # TRACK CREATION (Day 15: NEW person)
    # ========================================================================
    
    def create_track(self, 
                    detection: Detection,
                    local_track_id: int) -> str:
        """
        Create new track when no candidate found (Day 15 NEW assignment)
        
        Args:
            detection: Detection object
            local_track_id: ByteTrack local ID
        
        Returns:
            Global ID (UUID)
        """
        global_id = str(uuid.uuid4())[:12]  # Short UUID
        
        track = TrackInfo(
            global_id=global_id,
            local_track_id=local_track_id,
            state=TrackState.TENTATIVE,
            hits=1,
            detections=[detection],
            assignment_type=AssignmentType.NEW,
            assignment_confidence=detection.confidence,
            max_confidence=detection.confidence
        )
        
        track.cameras_visited.add(detection.camera_id)
        track.hits_history.append(1)
        
        if detection.appearance_feature:
            track.appearance_gallery.append(detection.appearance_feature)
        
        self.tracks[global_id] = track
        self.local_to_global[local_track_id] = global_id
        self.stats["tracks_created"] += 1
        
        logger.debug(f"Created NEW track: {global_id} (camera {detection.camera_id})")
        return global_id
    
    # ========================================================================
    # TRACK LINKING (Day 15: LINK or RESOLVED)
    # ========================================================================
    
    def link_track(self,
                  global_id: str,
                  detection: Detection,
                  local_track_id: int,
                  confidence: float = 0.9) -> bool:
        """
        Link detection to existing track (Day 15 LINK/RESOLVED)
        
        Args:
            global_id: Existing track ID
            detection: New detection
            local_track_id: ByteTrack local ID
            confidence: Match confidence
        
        Returns:
            True if linked successfully
        """
        if global_id not in self.tracks:
            logger.warning(f"Track {global_id} not found for linking")
            return False
        
        track = self.tracks[global_id]
        
        # Update track
        track.hits += 1
        track.hits_history.append(track.hits)
        track.last_seen = datetime.now()
        track.last_seen_frame = self.frame_count
        track.age = 0  # Reset age on match
        track.detections.append(detection)
        track.cameras_visited.add(detection.camera_id)
        track.assignment_confidence = max(track.assignment_confidence, confidence)
        track.max_confidence = max(track.max_confidence, confidence)
        
        # Update appearance gallery
        if detection.appearance_feature is not None:
            track.appearance_gallery.append(detection.appearance_feature)
            if len(track.appearance_gallery) > self.max_gallery_size:
                track.appearance_gallery.pop(0)
        
        # Check for CONFIRMATION (min_hits reached)
        if track.hits >= self.min_hits_confirm and track.state == TrackState.TENTATIVE:
            track.state = TrackState.CONFIRMED
            self.stats["tracks_confirmed"] += 1
            logger.info(f"Track {global_id} CONFIRMED after {track.hits} hits")
        
        # Update local mapping
        self.local_to_global[local_track_id] = global_id
        
        return True
    
    # ========================================================================
    # TRACK MERGING (DAY 16 - KEY FEATURE)
    # ========================================================================
    
    def merge_tracks(self,
                    primary_global_id: str,
                    secondary_global_id: str,
                    merge_reason: str = "appearance_similarity") -> bool:
        """
        Merge two tracks (same person, duplicate profiles)
        
        DAY 16 KEY FEATURE:
        When cross-camera linking detects the same person was tracked under
        different IDs, merge them into one unified track.
        
        Args:
            primary_global_id: Keep this track
            secondary_global_id: Merge this into primary
            merge_reason: Why merged (for audit trail)
        
        Returns:
            True if merged successfully
        """
        if (primary_global_id not in self.tracks or 
            secondary_global_id not in self.tracks):
            logger.warning(f"Cannot merge: tracks not found")
            return False
        
        primary = self.tracks[primary_global_id]
        secondary = self.tracks[secondary_global_id]
        
        # Merge appearance galleries
        combined = primary.appearance_gallery + secondary.appearance_gallery
        primary.appearance_gallery = combined[-self.max_gallery_size:]
        
        # Merge detection history
        primary.detections.extend(secondary.detections)
        primary.hits += secondary.hits
        primary.hits_history.extend(secondary.hits_history)
        
        # Merge cameras
        primary.cameras_visited.update(secondary.cameras_visited)
        
        # Track merged sources
        primary.merged_from.append(secondary_global_id)
        if secondary.merged_from:
            primary.merged_from.extend(secondary.merged_from)
        
        # Mark secondary as merged
        secondary.state = TrackState.MERGED
        secondary.merged_into = primary_global_id
        
        # Update mappings
        self.merged_mapping[secondary_global_id] = primary_global_id
        
        # Update local track mappings
        for local_id, gid in list(self.local_to_global.items()):
            if gid == secondary_global_id:
                self.local_to_global[local_id] = primary_global_id
        
        self.stats["tracks_merged"] += 1
        
        logger.info(f"Merged {secondary_global_id} → {primary_global_id} "
                   f"(reason: {merge_reason})")
        return True
    
    # ========================================================================
    # TRACK DELETION & LIFECYCLE
    # ========================================================================
    
    def handle_unmatched_track(self, global_id: str) -> bool:
        """
        Handle track not matched in current frame
        
        State transitions:
        - CONFIRMED → LOST (might reappear)
        - LOST + age > max_age → DELETED (timeout)
        - TENTATIVE → DELETED (insufficient evidence)
        """
        if global_id not in self.tracks:
            return False
        
        track = self.tracks[global_id]
        track.age += 1
        
        if track.state == TrackState.CONFIRMED:
            track.state = TrackState.LOST
            logger.debug(f"Track {global_id} → LOST")
        
        elif track.state == TrackState.LOST and track.age > self.max_age_lost:
            track.state = TrackState.DELETED
            self.stats["tracks_deleted"] += 1
            logger.info(f"Track {global_id} DELETED after {track.age} frames")
        
        elif track.state == TrackState.TENTATIVE and track.hits < 2:
            track.state = TrackState.DELETED
            self.stats["tracks_deleted"] += 1
            logger.debug(f"Track {global_id} DELETED (insufficient hits)")
        
        return True
    
    # ========================================================================
    # QUERY METHODS
    # ========================================================================
    
    def get_confirmed_tracks(self) -> List[TrackInfo]:
        """Return only CONFIRMED tracks (3+ hits)"""
        return [t for t in self.tracks.values() 
                if t.state == TrackState.CONFIRMED]
    
    def get_active_tracks(self) -> List[TrackInfo]:
        """Return CONFIRMED or LOST tracks (not DELETED)"""
        return [t for t in self.tracks.values() 
                if t.state in [TrackState.CONFIRMED, TrackState.LOST]]
    
    def get_track_by_global_id(self, global_id: str) -> Optional[TrackInfo]:
        """Get track by global ID"""
        return self.tracks.get(global_id)
    
    def get_track_by_local_id(self, local_id: int) -> Optional[TrackInfo]:
        """Get track by ByteTrack local ID"""
        if local_id in self.local_to_global:
            global_id = self.local_to_global[local_id]
            return self.tracks.get(global_id)
        return None
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        states = {}
        for track in self.tracks.values():
            state = track.state.value
            states[state] = states.get(state, 0) + 1
        
        return {
            "frame": self.frame_count,
            "total_tracks": len(self.tracks),
            "active_tracks": len(self.get_active_tracks()),
            "confirmed_tracks": len(self.get_confirmed_tracks()),
            "by_state": states,
            "statistics": self.stats,
        }
    
    def print_summary(self, verbose: bool = False) -> None:
        """Print human-readable summary"""
        stats = self.get_statistics()
        
        print(f"\n[Frame {stats['frame']}] Tracks Summary:")
        print(f"  • Confirmed: {stats['confirmed_tracks']}")
        print(f"  • Active: {stats['active_tracks']}")
        print(f"  • Total: {stats['total_tracks']}")
        
        if verbose:
            print(f"  • By State: {stats['by_state']}")
            print(f"  • Created: {stats['statistics']['tracks_created']}")
            print(f"  • Merged: {stats['statistics']['tracks_merged']}")


# ============================================================================
# END OF MODULE
# ============================================================================

if __name__ == "__main__":
    print("Day 16 Track Manager Module - Ready to use")
