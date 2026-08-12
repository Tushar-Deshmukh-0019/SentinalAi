"""
Track Visualization for SentinelAI

Provides tools for visualizing tracked objects on video frames.

Features:
- Draw bounding boxes with track IDs
- Draw trajectory paths (motion history)
- Color-code by confidence, age, or threat level
- Annotate with metadata (class, confidence, age)
- Performance metrics overlay
- Multiple visualization styles

Example Usage:
    visualizer = TrackVisualizer(frame_width=1280, frame_height=720)
    
    # Draw tracks on frame
    frame_with_tracks = visualizer.draw_tracks(frame, tracks)
    
    # Draw with trajectories
    frame_with_trajectories = visualizer.draw_trajectories(frame, tracks)
    
    # Draw with threat coloring
    frame_colored = visualizer.draw_tracks(
        frame, tracks, color_mode='threat'
    )
    
    # Draw full annotation including statistics
    frame_annotated = visualizer.draw_full_annotation(
        frame, tracks, stats=tracker_stats
    )
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging

from ai.tracking.pipeline_integration import TrackOutput

logger = logging.getLogger('tracking.visualizer')


class TrackVisualizer:
    """Visualize tracked objects on video frames
    
    Supports multiple visualization modes:
    - Basic: Bounding boxes with track IDs
    - Detailed: Boxes + class names + confidence
    - Trajectory: Boxes + motion history paths
    - Threat: Color-coded by confidence/threat level
    """
    
    # Color palettes
    COLORS = {
        'confirmed': (0, 255, 0),      # Green for confirmed tracks
        'tentative': (255, 165, 0),    # Orange for tentative
        'high_confidence': (0, 255, 0), # Green
        'medium_confidence': (255, 255, 0), # Yellow
        'low_confidence': (0, 165, 255),   # Orange
        'threat_high': (0, 0, 255),    # Red
        'threat_medium': (0, 165, 255), # Orange
        'threat_low': (0, 255, 0),     # Green
        'trajectory': (100, 100, 255),  # Light red
    }
    
    def __init__(self, frame_width: int = 1280, frame_height: int = 720,
                 font_scale: float = 0.7, font_thickness: int = 2,
                 box_thickness: int = 2, line_thickness: int = 1,
                 trajectory_length: int = 30):
        """Initialize visualizer
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            font_scale: Font size scale for text
            font_thickness: Font thickness for text
            box_thickness: Bounding box line thickness
            line_thickness: Trajectory line thickness
            trajectory_length: Number of points to show in trajectory
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.box_thickness = box_thickness
        self.line_thickness = line_thickness
        self.trajectory_length = trajectory_length
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[TrackOutput],
                   color_mode: str = 'status', show_confidence: bool = True,
                   show_id: bool = True) -> np.ndarray:
        """Draw bounding boxes for tracked objects
        
        Args:
            frame: Input frame (BGR)
            tracks: List of TrackOutput objects
            color_mode: 'status' (confirmed/tentative), 'confidence', or 'fixed'
            show_confidence: Whether to show confidence scores
            show_id: Whether to show track IDs
            
        Returns:
            Frame with drawn bounding boxes
        """
        frame = frame.copy()
        
        for track in tracks:
            # Get color based on mode
            color = self._get_color(track, color_mode)
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, track.bbox_xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
            
            # Draw label
            label_text = self._format_label(track, show_confidence, show_id)
            label_size, baseline = cv2.getTextSize(
                label_text, self.font, self.font_scale, self.font_thickness
            )
            
            label_y = max(y1 - 10, label_size[1])
            cv2.rectangle(
                frame,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0], label_y + baseline),
                color,
                -1
            )
            
            cv2.putText(
                frame, label_text,
                (x1, label_y),
                self.font, self.font_scale,
                (255, 255, 255),  # White text
                self.font_thickness
            )
            
            # Draw center point
            cx, cy = int(track.position[0]), int(track.position[1])
            cv2.circle(frame, (cx, cy), 4, color, -1)
        
        return frame
    
    def draw_trajectories(self, frame: np.ndarray, tracks: List[TrackOutput],
                         show_endpoints: bool = True,
                         max_trajectory_length: Optional[int] = None) -> np.ndarray:
        """Draw motion trajectories for tracked objects
        
        Args:
            frame: Input frame (BGR)
            tracks: List of TrackOutput objects
            show_endpoints: Whether to highlight trajectory endpoints
            max_trajectory_length: Limit trajectory points (None = all)
            
        Returns:
            Frame with drawn trajectories
        """
        frame = frame.copy()
        
        for track in tracks:
            if not track.trajectory or len(track.trajectory) < 2:
                continue
            
            # Get trajectory points to draw
            trajectory = track.trajectory
            if max_trajectory_length:
                trajectory = trajectory[-max_trajectory_length:]
            
            # Draw trajectory line
            points = np.array([(int(x), int(y)) for x, y in trajectory],
                             dtype=np.int32)
            
            # Draw with fading effect (lighter = older)
            for i in range(len(points) - 1):
                # Fade color based on age
                alpha = (i + 1) / len(points)
                color = tuple(int(c * alpha) for c in self.COLORS['trajectory'])
                cv2.line(frame, tuple(points[i]), tuple(points[i + 1]),
                        color, self.line_thickness)
            
            # Draw endpoints
            if show_endpoints:
                # Start point (oldest)
                cv2.circle(frame, tuple(points[0]), 3, (100, 100, 100), -1)
                # End point (newest)
                cv2.circle(frame, tuple(points[-1]), 5, (0, 255, 0), -1)
        
        return frame
    
    def draw_trajectories_with_boxes(self, frame: np.ndarray,
                                     tracks: List[TrackOutput]) -> np.ndarray:
        """Draw boxes and trajectories combined
        
        Args:
            frame: Input frame (BGR)
            tracks: List of TrackOutput objects
            
        Returns:
            Frame with boxes and trajectories
        """
        # Draw trajectories first (background)
        frame = self.draw_trajectories(frame, tracks)
        # Then draw boxes on top
        frame = self.draw_tracks(frame, tracks)
        return frame
    
    def draw_detailed_annotation(self, frame: np.ndarray,
                                tracks: List[TrackOutput]) -> np.ndarray:
        """Draw detailed annotation with all metadata
        
        Args:
            frame: Input frame (BGR)
            tracks: List of TrackOutput objects
            
        Returns:
            Frame with detailed annotations
        """
        frame = frame.copy()
        
        for track in tracks:
            # Get colors
            color = self._get_color(track, 'status')
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, track.bbox_xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
            
            # Draw center point
            cx, cy = int(track.position[0]), int(track.position[1])
            cv2.circle(frame, (cx, cy), 4, color, -1)
            
            # Build detailed label
            status = "CONFIRMED" if track.is_confirmed else "TENTATIVE"
            class_str = f"{track.class_name}" if track.class_name else "obj"
            conf_str = f"{track.confidence:.2f}"
            
            labels = [
                f"ID: {track.track_id}",
                f"Class: {class_str}",
                f"Conf: {conf_str}",
                f"Age: {track.age}",
                f"Hits: {track.hits}",
                f"Status: {status}"
            ]
            
            # Draw label background and text
            label_y = y1
            for i, label in enumerate(labels):
                label_size, baseline = cv2.getTextSize(
                    label, self.font, self.font_scale - 0.1, 1
                )
                
                label_y_pos = label_y - (len(labels) - i - 1) * 20 - 10
                
                cv2.rectangle(
                    frame,
                    (x1, label_y_pos - label_size[1] - 3),
                    (x1 + label_size[0] + 3, label_y_pos + baseline),
                    color,
                    -1
                )
                
                cv2.putText(
                    frame, label,
                    (x1 + 2, label_y_pos),
                    self.font, self.font_scale - 0.1,
                    (255, 255, 255),
                    1
                )
        
        return frame
    
    def draw_full_annotation(self, frame: np.ndarray, tracks: List[TrackOutput],
                            stats: Optional[Dict[str, Any]] = None,
                            show_trajectories: bool = True) -> np.ndarray:
        """Draw complete annotation including tracks, trajectories, and stats
        
        Args:
            frame: Input frame (BGR)
            tracks: List of TrackOutput objects
            stats: Optional tracker statistics
            show_trajectories: Whether to draw trajectories
            
        Returns:
            Fully annotated frame
        """
        frame = frame.copy()
        
        # Draw trajectories if requested
        if show_trajectories:
            frame = self.draw_trajectories(frame, tracks)
        
        # Draw track boxes
        frame = self.draw_tracks(frame, tracks, color_mode='status')
        
        # Draw statistics overlay
        if stats:
            frame = self._draw_stats_overlay(frame, stats)
        
        # Draw legend
        frame = self._draw_legend(frame)
        
        return frame
    
    def draw_threat_level(self, frame: np.ndarray, tracks: List[TrackOutput],
                         threat_calculator=None) -> np.ndarray:
        """Draw tracks colored by threat level
        
        Args:
            frame: Input frame (BGR)
            tracks: List of TrackOutput objects
            threat_calculator: Optional function(track) -> threat_level (0-1)
            
        Returns:
            Frame with threat-colored tracks
        """
        frame = frame.copy()
        
        for track in tracks:
            # Calculate threat level
            if threat_calculator:
                threat = threat_calculator(track)
            else:
                # Default: high confidence = low threat, low confidence = high threat
                threat = 1.0 - track.confidence
            
            # Get threat color
            if threat > 0.66:
                color = self.COLORS['threat_high']
                threat_str = "HIGH"
            elif threat > 0.33:
                color = self.COLORS['threat_medium']
                threat_str = "MEDIUM"
            else:
                color = self.COLORS['threat_low']
                threat_str = "LOW"
            
            # Draw box
            x1, y1, x2, y2 = map(int, track.bbox_xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
            
            # Draw threat label
            label = f"ID:{track.track_id} {threat_str}"
            cv2.putText(frame, label, (x1, y1 - 5),
                       self.font, self.font_scale,
                       (255, 255, 255), self.font_thickness)
        
        return frame
    
    def _get_color(self, track: TrackOutput, mode: str) -> Tuple[int, int, int]:
        """Get color for a track based on mode
        
        Args:
            track: Track to color
            mode: 'status', 'confidence', or 'fixed'
            
        Returns:
            BGR color tuple
        """
        if mode == 'status':
            return (self.COLORS['confirmed'] if track.is_confirmed
                   else self.COLORS['tentative'])
        
        elif mode == 'confidence':
            if track.confidence > 0.85:
                return self.COLORS['high_confidence']
            elif track.confidence > 0.7:
                return self.COLORS['medium_confidence']
            else:
                return self.COLORS['low_confidence']
        
        elif mode == 'age':
            # Newer = green, older = blue
            age_factor = min(track.age / 50, 1.0)  # Normalize to 0-1
            return (int(255 * age_factor), 255, 0)
        
        else:  # fixed
            # Use track ID to determine color
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 255, 0), (255, 0, 255), (0, 255, 255)
            ]
            return colors[track.track_id % len(colors)]
    
    def _format_label(self, track: TrackOutput, show_confidence: bool,
                     show_id: bool) -> str:
        """Format label text for track
        
        Args:
            track: Track to label
            show_confidence: Include confidence score
            show_id: Include track ID
            
        Returns:
            Formatted label string
        """
        parts = []
        
        if show_id:
            parts.append(f"ID:{track.track_id}")
        
        if track.class_name:
            parts.append(track.class_name)
        
        if show_confidence:
            parts.append(f"{track.confidence:.2f}")
        
        return " ".join(parts) if parts else "Track"
    
    def _draw_stats_overlay(self, frame: np.ndarray,
                           stats: Dict[str, Any]) -> np.ndarray:
        """Draw statistics overlay in corner
        
        Args:
            frame: Input frame
            stats: Statistics dictionary
            
        Returns:
            Frame with stats overlay
        """
        # Prepare stats text
        stat_lines = [
            f"Active Tracks: {stats.get('active_tracks', 0)}",
            f"Confirmed: {stats.get('confirmed_tracks', 0)}",
            f"Tentative: {stats.get('tentative_tracks', 0)}",
            f"Total Detections: {stats.get('total_detections', 0)}",
            f"Total Tracks Created: {stats.get('total_tracks_created', 0)}",
        ]
        
        # Draw semi-transparent background
        line_height = 25
        num_lines = len(stat_lines)
        bg_height = num_lines * line_height + 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        y_pos = 35
        for line in stat_lines:
            cv2.putText(frame, line, (20, y_pos),
                       self.font, self.font_scale - 0.1,
                       (0, 255, 0), 1)
            y_pos += line_height
        
        return frame
    
    def _draw_legend(self, frame: np.ndarray) -> np.ndarray:
        """Draw legend for colors
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with legend
        """
        legend_items = [
            ("Confirmed", self.COLORS['confirmed']),
            ("Tentative", self.COLORS['tentative']),
        ]
        
        # Draw legend in top-right
        start_x = frame.shape[1] - 200
        start_y = 20
        
        for i, (label, color) in enumerate(legend_items):
            y = start_y + i * 25
            cv2.rectangle(frame, (start_x, y), (start_x + 15, y + 15),
                         color, -1)
            cv2.putText(frame, label, (start_x + 25, y + 12),
                       self.font, self.font_scale - 0.2,
                       (255, 255, 255), 1)
        
        return frame
    
    @staticmethod
    def video_output_example(input_video: str, output_video: str,
                            tracker_instance, visualizer: 'TrackVisualizer'):
        """Example of processing video with tracking and visualization
        
        Args:
            input_video: Path to input video file
            output_video: Path to save output video
            tracker_instance: Initialized tracker instance
            visualizer: TrackVisualizer instance
        """
        import sys
        from pathlib import Path
        
        cap = cv2.VideoCapture(input_video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame (placeholder - requires actual detector)
            # detections = detector.detect(frame)
            # tracks = tracker_instance.update(detections)
            # frame = visualizer.draw_full_annotation(frame, tracks)
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        logger.info(f"Processed {frame_count} frames, saved to {output_video}")


if __name__ == '__main__':
    print("Track visualizer module loaded")
    print("Use TrackVisualizer to draw tracks on frames")
