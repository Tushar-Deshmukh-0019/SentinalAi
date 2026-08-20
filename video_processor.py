"""
Day 16: Video Frame Processing with Track Manager

Purpose:
  Read video frames, detect persons, run track management, show results.

Features:
  • Video input/output with OpenCV
  • Frame-by-frame detection
  • Track visualization on frames
  • Cross-camera merging
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging
from track_manager import TrackManager, Detection, TrackState

logger = logging.getLogger(__name__)


class VideoTracker:
    """Process video frames with track management"""
    
    def __init__(self, video_path: str, output_path: str = "output.mp4"):
        """
        Initialize video tracker
        
        Args:
            video_path: Input video file
            output_path: Output video file with tracking visualization
        """
        self.video_path = video_path
        self.output_path = output_path
        self.track_manager = TrackManager(min_hits_confirm=3, max_age_lost=70)
        
        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                                   (self.width, self.height))
        
        self.frame_count = 0
        logger.info(f"VideoTracker initialized: {self.width}x{self.height}@{self.fps}fps")
    
    def detect_persons(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect persons in frame (simulated - use YOLOv8 in production)
        
        Returns:
            List of Detection objects
        """
        # In production: use YOLOv8
        # For now: detect green rectangles drawn on frame
        detections = []
        
        # Convert to HSV for green detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Minimum size
                x, y, w, h = cv2.boundingRect(contour)
                
                # Determine camera zone
                if x < self.width // 3:
                    camera_id = 1
                elif x < 2 * self.width // 3:
                    camera_id = 2
                else:
                    camera_id = 3
                
                detection = Detection(
                    bbox=(x, y, x+w, y+h),
                    confidence=0.95,
                    camera_id=camera_id,
                    frame_num=self.frame_count,
                    appearance_feature=None
                )
                detections.append(detection)
        
        return detections
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process single frame:
        1. Detect persons
        2. Run track manager
        3. Visualize results
        
        Returns:
            (annotated_frame, tracking_info)
        """
        self.track_manager.frame_count = self.frame_count
        
        # Detect persons
        detections = self.detect_persons(frame)
        
        tracking_info = {
            "frame": self.frame_count,
            "detections": len(detections),
            "confirmed_tracks": len(self.track_manager.get_confirmed_tracks()),
            "actions": []
        }
        
        # Process detections through track manager
        for idx, detection in enumerate(detections):
            local_id = idx
            
            # Try to link to existing track
            linked = False
            for global_id, track in self.track_manager.tracks.items():
                if track.state in [TrackState.TENTATIVE, TrackState.CONFIRMED]:
                    if abs(detection.camera_id - track.cameras_visited.__iter__().__next__()) <= 1:
                        if self.track_manager.link_track(global_id, detection, 
                                                         local_id, detection.confidence):
                            tracking_info["actions"].append(f"LINK {global_id[:8]}")
                            linked = True
                            break
            
            # Create new track if not linked
            if not linked:
                global_id = self.track_manager.create_track(detection, local_id)
                tracking_info["actions"].append(f"NEW {global_id[:8]}")
        
        # Visualize
        annotated = self._visualize_tracks(frame)
        
        return annotated, tracking_info
    
    def _visualize_tracks(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracks on frame"""
        result = frame.copy()
        
        # Draw confirmed tracks
        for track in self.track_manager.get_confirmed_tracks():
            if track.detections:
                last_det = track.detections[-1]
                x1, y1, x2, y2 = last_det.bbox
                
                # Green box for confirmed
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(result, f"ID:{track.global_id[:6]}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
        
        # Draw tentative tracks
        for track in self.track_manager.tracks.values():
            if track.state == TrackState.TENTATIVE:
                if track.detections:
                    last_det = track.detections[-1]
                    x1, y1, x2, y2 = last_det.bbox
                    
                    # Yellow box for tentative
                    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(result, f"T:{track.global_id[:6]}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 255, 255), 1)
        
        # Statistics overlay
        stats = self.track_manager.get_statistics()
        cv2.putText(result, f"Frame: {self.frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(result, f"Confirmed: {stats['confirmed_tracks']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, f"Total: {stats['total_tracks']}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return result
    
    def process_video(self) -> Dict:
        """Process entire video and generate output"""
        print(f"\nProcessing video: {self.video_path}")
        print(f"Total frames: {self.total_frames}")
        
        results = {
            "total_frames": 0,
            "final_tracks": 0,
            "confirmed_tracks": 0,
            "frame_logs": []
        }
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            annotated, info = self.process_frame(frame)
            
            # Write to output
            self.out.write(annotated)
            
            results["frame_logs"].append(info)
            results["total_frames"] = self.frame_count
            
            self.frame_count += 1
            
            # Progress
            if self.frame_count % 30 == 0:
                print(f"  Processed {self.frame_count}/{self.total_frames} frames")
        
        # Finalize
        stats = self.track_manager.get_statistics()
        results["final_tracks"] = stats["total_tracks"]
        results["confirmed_tracks"] = stats["confirmed_tracks"]
        
        self.cap.release()
        self.out.release()
        
        print(f"✓ Output saved: {self.output_path}")
        
        return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    tracker = VideoTracker("input_video.mp4", "output_tracked.mp4")
    results = tracker.process_video()
    
    print("\nFinal Results:")
    print(f"  Total frames: {results['total_frames']}")
    print(f"  Final tracks: {results['final_tracks']}")
    print(f"  Confirmed: {results['confirmed_tracks']}")
