"""
Camera Feed Manager

Manages multiple camera streams with thread-safe ingestion.

Real-world requirements:
- Handle 10-20 cameras simultaneously
- Maintain 30 FPS per camera
- Recover from network failures
- Timestamp every frame
- Thread-safe operation
- Memory-efficient buffering

This is the entry point for all surveillance data.
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraStatus(Enum):
    """Camera connection status."""
    INITIALIZING = "initializing"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    
    camera_id: str
    """Unique identifier for this camera."""
    
    source: str
    """
    Camera source:
    - RTSP: "rtsp://192.168.1.100:554/stream"
    - USB: 0, 1, 2 (device index)
    - File: "path/to/video.mp4"
    """
    
    name: str = "Unknown Camera"
    """Human-readable camera name."""
    
    location: str = "Unknown"
    """Physical location (e.g., "Main Gate", "North Fence")."""
    
    fps: int = 30
    """Target FPS for this camera."""
    
    priority: int = 1
    """
    Processing priority (1-10).
    Higher = more important = processed first.
    
    Example priorities:
    - 10: Critical entrance/exit points
    - 5: General perimeter
    - 1: Low-priority monitoring areas
    """
    
    max_reconnect_attempts: int = 5
    """Maximum reconnection attempts before marking as failed."""
    
    reconnect_delay: float = 5.0
    """Seconds to wait before reconnection attempt."""
    
    buffer_size: int = 10
    """Maximum frames in buffer (prevents memory overflow)."""


@dataclass
class Frame:
    """Frame data with metadata."""
    
    camera_id: str
    image: np.ndarray
    timestamp: float
    frame_number: int
    
    # Metadata
    width: int = 0
    height: int = 0
    channels: int = 3
    
    def __post_init__(self):
        """Extract image properties."""
        if self.image is not None and len(self.image.shape) >= 2:
            self.height, self.width = self.image.shape[:2]
            self.channels = self.image.shape[2] if len(self.image.shape) == 3 else 1


class CameraStream:
    """
    Manages a single camera stream in a dedicated thread.
    
    Responsibilities:
    - Open and maintain connection
    - Read frames continuously
    - Handle reconnection
    - Buffer frames
    - Report statistics
    """
    
    def __init__(self, config: CameraConfig):
        """Initialize camera stream."""
        self.config = config
        self.status = CameraStatus.INITIALIZING
        
        # Threading
        self.thread = None
        self.stop_event = threading.Event()
        
        # Frame buffer (thread-safe queue)
        self.frame_buffer = queue.Queue(maxsize=config.buffer_size)
        
        # Statistics
        self.frames_captured = 0
        self.frames_dropped = 0
        self.reconnect_attempts = 0
        self.last_frame_time = 0.0
        self.fps_actual = 0.0
        
        # Camera capture object
        self.capture = None
    
    def start(self):
        """Start camera stream in background thread."""
        if self.thread and self.thread.is_alive():
            logger.warning(f"Camera {self.config.camera_id} already running")
            return
        
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started camera stream: {self.config.camera_id} ({self.config.name})")
    
    def stop(self):
        """Stop camera stream."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        
        if self.capture:
            self.capture.release()
        
        self.status = CameraStatus.STOPPED
        logger.info(f"Stopped camera stream: {self.config.camera_id}")
    
    def _connect(self) -> bool:
        """
        Establish connection to camera.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse source
            if isinstance(self.config.source, int):
                # USB camera
                self.capture = cv2.VideoCapture(self.config.source)
            elif self.config.source.startswith('rtsp://') or self.config.source.startswith('http://'):
                # Network stream
                self.capture = cv2.VideoCapture(self.config.source)
                # Set buffer size for network streams (reduce latency)
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                # File path
                self.capture = cv2.VideoCapture(self.config.source)
            
            # Verify connection
            if not self.capture.isOpened():
                logger.error(f"Failed to open camera: {self.config.camera_id}")
                return False
            
            # Try to read first frame
            ret, frame = self.capture.read()
            if not ret or frame is None:
                logger.error(f"Failed to read from camera: {self.config.camera_id}")
                self.capture.release()
                return False
            
            self.status = CameraStatus.CONNECTED
            logger.info(f"Connected to camera: {self.config.camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Connection error for {self.config.camera_id}: {e}")
            return False
    
    def _capture_loop(self):
        """Main capture loop (runs in thread)."""
        frame_number = 0
        last_fps_calc = time.time()
        fps_counter = 0
        
        # Initial connection
        if not self._connect():
            self.status = CameraStatus.FAILED
            return
        
        while not self.stop_event.is_set():
            try:
                # Read frame
                ret, image = self.capture.read()
                
                if not ret or image is None:
                    # Frame read failed - try reconnecting
                    logger.warning(f"Frame read failed: {self.config.camera_id}")
                    if not self._reconnect():
                        break
                    continue
                
                current_time = time.time()
                
                # Create Frame object
                frame = Frame(
                    camera_id=self.config.camera_id,
                    image=image,
                    timestamp=current_time,
                    frame_number=frame_number
                )
                
                # Add to buffer (non-blocking)
                try:
                    self.frame_buffer.put_nowait(frame)
                    self.frames_captured += 1
                    frame_number += 1
                except queue.Full:
                    # Buffer full - drop oldest frame
                    try:
                        self.frame_buffer.get_nowait()  # Remove oldest
                        self.frame_buffer.put_nowait(frame)  # Add new
                        self.frames_dropped += 1
                    except:
                        pass
                
                # Update FPS calculation
                fps_counter += 1
                if current_time - last_fps_calc >= 1.0:
                    self.fps_actual = fps_counter / (current_time - last_fps_calc)
                    fps_counter = 0
                    last_fps_calc = current_time
                
                self.last_frame_time = current_time
                
                # Frame rate control (if needed)
                if self.config.fps > 0:
                    time.sleep(1.0 / self.config.fps)
                
            except Exception as e:
                logger.error(f"Capture error for {self.config.camera_id}: {e}")
                if not self._reconnect():
                    break
        
        # Cleanup
        if self.capture:
            self.capture.release()
    
    def _reconnect(self) -> bool:
        """
        Attempt to reconnect to camera.
        
        Returns:
            True if reconnection successful, False otherwise
        """
        self.status = CameraStatus.RECONNECTING
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        while (self.reconnect_attempts < self.config.max_reconnect_attempts and
               not self.stop_event.is_set()):
            
            self.reconnect_attempts += 1
            logger.info(f"Reconnection attempt {self.reconnect_attempts}/"
                       f"{self.config.max_reconnect_attempts} for {self.config.camera_id}")
            
            time.sleep(self.config.reconnect_delay)
            
            if self._connect():
                self.reconnect_attempts = 0
                return True
        
        # Reconnection failed
        self.status = CameraStatus.FAILED
        logger.error(f"Reconnection failed for {self.config.camera_id}")
        return False
    
    def get_frame(self, timeout: float = 0.1) -> Optional[Frame]:
        """
        Get next frame from buffer.
        
        Args:
            timeout: Maximum time to wait for frame (seconds)
            
        Returns:
            Frame object or None if timeout
        """
        try:
            return self.frame_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_statistics(self) -> Dict:
        """Get camera statistics."""
        return {
            'camera_id': self.config.camera_id,
            'name': self.config.name,
            'status': self.status.value,
            'frames_captured': self.frames_captured,
            'frames_dropped': self.frames_dropped,
            'drop_rate': (
                self.frames_dropped / self.frames_captured * 100
                if self.frames_captured > 0 else 0
            ),
            'fps_actual': self.fps_actual,
            'fps_target': self.config.fps,
            'buffer_size': self.frame_buffer.qsize(),
            'buffer_max': self.config.buffer_size,
            'reconnect_attempts': self.reconnect_attempts,
            'last_frame_age': time.time() - self.last_frame_time if self.last_frame_time > 0 else None
        }


class CameraFeedManager:
    """
    Manages multiple camera streams.
    
    Central coordinator for all camera feeds in the surveillance system.
    """
    
    def __init__(self):
        """Initialize camera feed manager."""
        self.cameras: Dict[str, CameraStream] = {}
        self.running = False
        
        logger.info("Camera Feed Manager initialized")
    
    def add_camera(self, config: CameraConfig) -> bool:
        """
        Add a camera to the system.
        
        Args:
            config: Camera configuration
            
        Returns:
            True if added successfully, False otherwise
        """
        if config.camera_id in self.cameras:
            logger.warning(f"Camera {config.camera_id} already exists")
            return False
        
        camera = CameraStream(config)
        self.cameras[config.camera_id] = camera
        
        # Auto-start if manager is running
        if self.running:
            camera.start()
        
        logger.info(f"Added camera: {config.camera_id} ({config.name})")
        return True
    
    def remove_camera(self, camera_id: str) -> bool:
        """Remove a camera from the system."""
        if camera_id not in self.cameras:
            logger.warning(f"Camera {camera_id} not found")
            return False
        
        camera = self.cameras[camera_id]
        camera.stop()
        del self.cameras[camera_id]
        
        logger.info(f"Removed camera: {camera_id}")
        return True
    
    def start_all(self):
        """Start all camera streams."""
        self.running = True
        for camera in self.cameras.values():
            camera.start()
        logger.info(f"Started all cameras ({len(self.cameras)})")
    
    def stop_all(self):
        """Stop all camera streams."""
        self.running = False
        for camera in self.cameras.values():
            camera.stop()
        logger.info("Stopped all cameras")
    
    def get_frame(self, camera_id: str, timeout: float = 0.1) -> Optional[Frame]:
        """Get next frame from specific camera."""
        if camera_id not in self.cameras:
            return None
        return self.cameras[camera_id].get_frame(timeout)
    
    def get_all_frames(self, timeout: float = 0.1) -> Dict[str, Frame]:
        """
        Get latest frame from all cameras.
        
        Returns:
            Dictionary mapping camera_id to Frame
        """
        frames = {}
        for camera_id, camera in self.cameras.items():
            frame = camera.get_frame(timeout)
            if frame:
                frames[camera_id] = frame
        return frames
    
    def get_statistics(self) -> Dict:
        """Get statistics for all cameras."""
        stats = {
            'total_cameras': len(self.cameras),
            'cameras_connected': sum(
                1 for cam in self.cameras.values()
                if cam.status == CameraStatus.CONNECTED
            ),
            'cameras_failed': sum(
                1 for cam in self.cameras.values()
                if cam.status == CameraStatus.FAILED
            ),
            'total_frames_captured': sum(
                cam.frames_captured for cam in self.cameras.values()
            ),
            'total_frames_dropped': sum(
                cam.frames_dropped for cam in self.cameras.values()
            ),
            'cameras': {
                cam_id: cam.get_statistics()
                for cam_id, cam in self.cameras.items()
            }
        }
        
        # Calculate aggregate FPS
        total_fps = sum(
            cam.fps_actual for cam in self.cameras.values()
            if cam.status == CameraStatus.CONNECTED
        )
        stats['total_fps'] = total_fps
        
        return stats
    
    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("CAMERA FEED MANAGER STATISTICS")
        print("="*70)
        print(f"Total Cameras: {stats['total_cameras']}")
        print(f"Connected: {stats['cameras_connected']}")
        print(f"Failed: {stats['cameras_failed']}")
        print(f"Total FPS: {stats['total_fps']:.1f}")
        print(f"Frames Captured: {stats['total_frames_captured']}")
        print(f"Frames Dropped: {stats['total_frames_dropped']}")
        
        if stats['total_frames_captured'] > 0:
            drop_rate = (stats['total_frames_dropped'] / 
                        stats['total_frames_captured'] * 100)
            print(f"Drop Rate: {drop_rate:.2f}%")
        
        print("\nPer-Camera Statistics:")
        print("-"*70)
        for cam_id, cam_stats in stats['cameras'].items():
            print(f"\n{cam_stats['name']} ({cam_id}):")
            print(f"  Status: {cam_stats['status']}")
            print(f"  FPS: {cam_stats['fps_actual']:.1f} / {cam_stats['fps_target']}")
            print(f"  Frames: {cam_stats['frames_captured']} "
                  f"(dropped: {cam_stats['frames_dropped']})")
            print(f"  Buffer: {cam_stats['buffer_size']}/{cam_stats['buffer_max']}")
            
            if cam_stats['last_frame_age'] is not None:
                print(f"  Last frame: {cam_stats['last_frame_age']:.2f}s ago")


# Example usage
if __name__ == "__main__":
    # Create manager
    manager = CameraFeedManager()
    
    # Add cameras
    manager.add_camera(CameraConfig(
        camera_id="cam_01",
        source=0,  # USB camera
        name="Main Gate",
        location="North Entrance",
        fps=30,
        priority=10
    ))
    
    manager.add_camera(CameraConfig(
        camera_id="cam_02",
        source="path/to/video.mp4",  # File
        name="Perimeter",
        location="East Fence",
        fps=30,
        priority=5
    ))
    
    # Start all cameras
    manager.start_all()
    
    try:
        # Process frames for 10 seconds
        start_time = time.time()
        while time.time() - start_time < 10:
            frames = manager.get_all_frames()
            
            for camera_id, frame in frames.items():
                print(f"Received frame from {camera_id}: "
                      f"{frame.width}x{frame.height} @ {frame.timestamp:.2f}")
            
            time.sleep(0.1)
    
    finally:
        # Stop and print statistics
        manager.print_statistics()
        manager.stop_all()
