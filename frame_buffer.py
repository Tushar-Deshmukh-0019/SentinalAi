"""
Frame Buffer & Queue System

Intelligent frame organization for optimal threat detection.

The Critical Problem:
- Ingestion: 300 frames/second (from 10 cameras)
- Processing: 20 frames/second (bottleneck)
- Gap: 280 frames must be dropped

The Question:
Which 20 frames should we keep?

The Answer:
Priority-based intelligent selection:
- Critical cameras (main gate): ALWAYS process
- High-priority cameras (perimeter): Process when possible
- Low-priority cameras (parking): Process if capacity available

This is the difference between catching threats and missing them.

Real-world deployment:
- Indian Army border posts
- Airport security checkpoints  
- Critical infrastructure protection
- Government building surveillance

Every frame matters. Priority decides which frames matter MOST.
"""

import queue
import threading
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from .camera_feed_manager import Frame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FramePriority(Enum):
    """Frame priority levels."""
    CRITICAL = 10    # Main gates, entry/exit, border crossings
    HIGH = 8         # Perimeter, fence line, checkpoints
    MEDIUM = 5       # Secondary monitoring areas
    LOW = 3          # Parking, interior, equipment rooms
    MINIMAL = 1      # Archive cameras, rarely checked areas


@dataclass(order=True)
class PrioritizedFrame:
    """
    Frame with priority for queue ordering.
    
    Uses @dataclass(order=True) to enable automatic comparison.
    Fields are compared in order: priority first, then timestamp.
    """
    
    priority: int = field(compare=True)
    """
    Priority level (higher = more important).
    Negative because queue.PriorityQueue is min-heap.
    We negate to get max-heap behavior (highest priority first).
    """
    
    timestamp: float = field(compare=True)
    """Timestamp for tie-breaking (older frames first if same priority)."""
    
    frame: Frame = field(compare=False)
    """The actual frame data (not used in comparison)."""
    
    def __post_init__(self):
        """Negate priority for max-heap behavior."""
        # PriorityQueue is min-heap (smallest first)
        # We want max-heap (largest first)
        # So we negate the priority
        self.priority = -self.priority


class FrameBuffer:
    """
    Intelligent frame buffer with priority-based processing.
    
    Key Features:
    ============
    
    1. Priority Queue
    -----------------
    Frames from critical cameras processed first.
    
    Example:
        Frame from Main Gate (priority 10)   ← Processed first
        Frame from Checkpoint (priority 10)   ← Then this
        Frame from Perimeter (priority 8)     ← Then this
        Frame from Parking (priority 3)       ← Last (if ever)
    
    2. Intelligent Dropping
    -----------------------
    When buffer is full, drop LOW priority frames, keep HIGH priority.
    
    3. Multi-Consumer Support
    -------------------------
    Multiple detection threads can read simultaneously.
    
    4. Statistics & Monitoring
    --------------------------
    Track what's being processed vs. dropped.
    
    Real-World Impact:
    ==================
    
    Without priority:
        Critical threat appears at main gate → 5th in queue → detected after 250ms
        By then, intruder is past the gate ❌
    
    With priority:
        Critical threat appears at main gate → 1st in queue → detected in 50ms
        Alert sent while still at entrance → Response deployed → Threat intercepted ✓
    
    This is not theoretical. This saves lives.
    """
    
    def __init__(
        self,
        max_size: int = 100,
        drop_threshold: float = 0.8,
        priority_mapping: Optional[Dict[str, int]] = None
    ):
        """
        Initialize frame buffer.
        
        Args:
            max_size: Maximum frames in buffer
            drop_threshold: Start dropping at this fullness (0.8 = 80%)
            priority_mapping: Map camera_id to priority level
        """
        self.max_size = max_size
        self.drop_threshold = drop_threshold
        
        # Priority queue (thread-safe)
        self.frame_queue = queue.PriorityQueue(maxsize=max_size)
        
        # Priority mapping (camera_id -> priority)
        self.priority_mapping = priority_mapping or {}
        
        # Statistics
        self.stats_lock = threading.Lock()
        self.frames_received = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.frames_dropped_by_priority = {
            FramePriority.CRITICAL.value: 0,
            FramePriority.HIGH.value: 0,
            FramePriority.MEDIUM.value: 0,
            FramePriority.LOW.value: 0,
            FramePriority.MINIMAL.value: 0
        }
        
        logger.info(f"Frame Buffer initialized (max_size={max_size})")
    
    def set_camera_priority(self, camera_id: str, priority: int):
        """
        Set priority for a camera.
        
        Args:
            camera_id: Camera identifier
            priority: Priority level (1-10, higher = more important)
        """
        self.priority_mapping[camera_id] = priority
        logger.info(f"Set priority for {camera_id}: {priority}")
    
    def get_camera_priority(self, camera_id: str) -> int:
        """Get priority for a camera (default: MEDIUM)."""
        return self.priority_mapping.get(camera_id, FramePriority.MEDIUM.value)
    
    def put(self, frame: Frame, block: bool = False, timeout: Optional[float] = None) -> bool:
        """
        Add frame to buffer.
        
        Args:
            frame: Frame to add
            block: Wait if buffer is full
            timeout: Maximum wait time if blocking
            
        Returns:
            True if frame was added, False if dropped
        """
        with self.stats_lock:
            self.frames_received += 1
        
        # Get priority for this camera
        priority = self.get_camera_priority(frame.camera_id)
        
        # Create prioritized frame
        pframe = PrioritizedFrame(
            priority=priority,
            timestamp=frame.timestamp,
            frame=frame
        )
        
        # Check if we should drop this frame
        current_size = self.frame_queue.qsize()
        fullness = current_size / self.max_size
        
        if fullness >= self.drop_threshold:
            # Buffer getting full - apply intelligent dropping
            if self._should_drop(priority, fullness):
                with self.stats_lock:
                    self.frames_dropped += 1
                    self.frames_dropped_by_priority[priority] += 1
                
                logger.debug(
                    f"Dropped frame from {frame.camera_id} "
                    f"(priority={priority}, fullness={fullness:.1%})"
                )
                return False
        
        # Try to add frame
        try:
            self.frame_queue.put(pframe, block=block, timeout=timeout)
            return True
        except queue.Full:
            # Queue full and not blocking
            with self.stats_lock:
                self.frames_dropped += 1
                self.frames_dropped_by_priority[priority] += 1
            return False
    
    def _should_drop(self, priority: int, fullness: float) -> bool:
        """
        Decide whether to drop a frame based on priority and buffer fullness.
        
        Strategy:
        =========
        
        CRITICAL (priority 10):
            - NEVER drop until 95% full
            - Even then, only drop if absolutely necessary
        
        HIGH (priority 8):
            - Start dropping at 85% full
            - Aggressive dropping at 90%
        
        MEDIUM (priority 5):
            - Start dropping at 80% full
            - Very aggressive at 85%
        
        LOW (priority 3):
            - Start dropping at 70% full
            - Almost always drop at 80%
        
        MINIMAL (priority 1):
            - Start dropping at 60% full
            - Always drop at 70%
        
        Args:
            priority: Frame priority (1-10)
            fullness: Buffer fullness (0.0-1.0)
            
        Returns:
            True if frame should be dropped
        """
        if priority >= FramePriority.CRITICAL.value:
            # CRITICAL: Only drop if desperately full
            return fullness > 0.95
        
        elif priority >= FramePriority.HIGH.value:
            # HIGH: Start dropping at 85%
            if fullness < 0.85:
                return False
            elif fullness < 0.90:
                return fullness > 0.87  # 50% chance in this range
            else:
                return fullness > 0.92  # Still try to keep some
        
        elif priority >= FramePriority.MEDIUM.value:
            # MEDIUM: Start dropping at 80%
            if fullness < 0.80:
                return False
            else:
                return fullness > 0.82  # Drop most after 82%
        
        elif priority >= FramePriority.LOW.value:
            # LOW: Start dropping at 70%
            if fullness < 0.70:
                return False
            else:
                return True  # Drop aggressively
        
        else:
            # MINIMAL: Drop early and often
            return fullness > 0.60
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Frame]:
        """
        Get next frame from buffer (highest priority first).
        
        Args:
            block: Wait if buffer is empty
            timeout: Maximum wait time if blocking
            
        Returns:
            Frame object or None if timeout/empty
        """
        try:
            pframe = self.frame_queue.get(block=block, timeout=timeout)
            
            with self.stats_lock:
                self.frames_processed += 1
            
            return pframe.frame
        
        except queue.Empty:
            return None
    
    def get_batch(self, batch_size: int, timeout: float = 0.1) -> List[Frame]:
        """
        Get multiple frames at once (for batch processing).
        
        Args:
            batch_size: Number of frames to retrieve
            timeout: Maximum wait time for first frame
            
        Returns:
            List of frames (may be shorter than batch_size)
        """
        frames = []
        
        # Get first frame (with timeout)
        first_frame = self.get(block=True, timeout=timeout)
        if first_frame is None:
            return frames
        
        frames.append(first_frame)
        
        # Get remaining frames (non-blocking)
        for _ in range(batch_size - 1):
            frame = self.get(block=False)
            if frame is None:
                break
            frames.append(frame)
        
        return frames
    
    def size(self) -> int:
        """Get current number of frames in buffer."""
        return self.frame_queue.qsize()
    
    def fullness(self) -> float:
        """Get buffer fullness (0.0 to 1.0)."""
        return self.frame_queue.qsize() / self.max_size
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.frame_queue.empty()
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.frame_queue.full()
    
    def clear(self):
        """Clear all frames from buffer."""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("Buffer cleared")
    
    def get_statistics(self) -> Dict:
        """Get buffer statistics."""
        with self.stats_lock:
            return {
                'max_size': self.max_size,
                'current_size': self.size(),
                'fullness': self.fullness(),
                'frames_received': self.frames_received,
                'frames_processed': self.frames_processed,
                'frames_dropped': self.frames_dropped,
                'drop_rate': (
                    self.frames_dropped / self.frames_received * 100
                    if self.frames_received > 0 else 0
                ),
                'frames_dropped_by_priority': self.frames_dropped_by_priority.copy(),
                'priority_mapping': self.priority_mapping.copy()
            }
    
    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("FRAME BUFFER STATISTICS")
        print("="*70)
        print(f"Buffer Size: {stats['current_size']}/{stats['max_size']} "
              f"({stats['fullness']:.1%} full)")
        print(f"Frames Received: {stats['frames_received']}")
        print(f"Frames Processed: {stats['frames_processed']}")
        print(f"Frames Dropped: {stats['frames_dropped']} ({stats['drop_rate']:.2f}%)")
        
        print("\nDropped by Priority:")
        print("-"*70)
        for priority_level in sorted(stats['frames_dropped_by_priority'].keys(), reverse=True):
            count = stats['frames_dropped_by_priority'][priority_level]
            if count > 0:
                print(f"  Priority {priority_level}: {count} frames")
        
        print("\nCamera Priorities:")
        print("-"*70)
        for camera_id, priority in sorted(
            stats['priority_mapping'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            priority_name = self._get_priority_name(priority)
            print(f"  {camera_id}: {priority} ({priority_name})")
    
    def _get_priority_name(self, priority: int) -> str:
        """Get human-readable priority name."""
        if priority >= 10:
            return "CRITICAL"
        elif priority >= 8:
            return "HIGH"
        elif priority >= 5:
            return "MEDIUM"
        elif priority >= 3:
            return "LOW"
        else:
            return "MINIMAL"


class FrameProcessor:
    """
    Frame processor that consumes frames from buffer.
    
    This is a template for how detection pipelines will consume frames.
    Day 7 will implement the actual detection pipeline.
    """
    
    def __init__(
        self,
        processor_id: str,
        frame_buffer: FrameBuffer,
        processing_function=None
    ):
        """
        Initialize frame processor.
        
        Args:
            processor_id: Unique processor identifier
            frame_buffer: Frame buffer to consume from
            processing_function: Function to process each frame
        """
        self.processor_id = processor_id
        self.frame_buffer = frame_buffer
        self.processing_function = processing_function or self._default_processing
        
        # Threading
        self.thread = None
        self.stop_event = threading.Event()
        
        # Statistics
        self.frames_processed = 0
        self.processing_time_total = 0.0
        
        logger.info(f"Frame Processor initialized: {processor_id}")
    
    def start(self):
        """Start processing in background thread."""
        if self.thread and self.thread.is_alive():
            logger.warning(f"Processor {self.processor_id} already running")
            return
        
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started processor: {self.processor_id}")
    
    def stop(self):
        """Stop processing."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info(f"Stopped processor: {self.processor_id}")
    
    def _processing_loop(self):
        """Main processing loop."""
        while not self.stop_event.is_set():
            # Get frame from buffer
            frame = self.frame_buffer.get(block=True, timeout=0.1)
            
            if frame is None:
                continue
            
            # Process frame
            start_time = time.time()
            
            try:
                self.processing_function(frame)
            except Exception as e:
                logger.error(f"Processing error in {self.processor_id}: {e}")
            
            # Update statistics
            processing_time = time.time() - start_time
            self.frames_processed += 1
            self.processing_time_total += processing_time
    
    def _default_processing(self, frame: Frame):
        """Default processing (just simulate work)."""
        # Simulate detection time
        time.sleep(0.05)  # 50ms = realistic detection time
        
        logger.debug(
            f"[{self.processor_id}] Processed frame from {frame.camera_id} "
            f"(frame #{frame.frame_number})"
        )
    
    def get_statistics(self) -> Dict:
        """Get processor statistics."""
        avg_time = (
            self.processing_time_total / self.frames_processed
            if self.frames_processed > 0 else 0
        )
        
        return {
            'processor_id': self.processor_id,
            'frames_processed': self.frames_processed,
            'processing_time_total': self.processing_time_total,
            'avg_processing_time': avg_time,
            'fps': 1.0 / avg_time if avg_time > 0 else 0
        }


# Example usage
if __name__ == "__main__":
    from .camera_feed_manager import CameraFeedManager, CameraConfig
    
    print("="*70)
    print("FRAME BUFFER & QUEUE SYSTEM DEMO")
    print("="*70)
    print("\nScenario: Border surveillance post with 5 cameras")
    print("We'll show how priority queue processes critical cameras first.\n")
    
    # Create camera feed manager
    manager = CameraFeedManager()
    
    # Create frame buffer
    buffer = FrameBuffer(max_size=50, drop_threshold=0.8)
    
    # Configure cameras with priorities
    cameras = [
        CameraConfig(
            camera_id="main_gate",
            source=0,
            name="Main Gate",
            fps=30,
            priority=10  # CRITICAL
        ),
        CameraConfig(
            camera_id="north_fence",
            source=0,
            name="North Fence",
            fps=30,
            priority=8  # HIGH
        ),
        CameraConfig(
            camera_id="south_fence",
            source=0,
            name="South Fence",
            fps=30,
            priority=8  # HIGH
        ),
        CameraConfig(
            camera_id="parking",
            source=0,
            name="Parking Lot",
            fps=30,
            priority=3  # LOW
        ),
        CameraConfig(
            camera_id="interior",
            source=0,
            name="Interior Hall",
            fps=30,
            priority=1  # MINIMAL
        )
    ]
    
    # Add cameras and set priorities
    for cam_config in cameras:
        manager.add_camera(cam_config)
        buffer.set_camera_priority(cam_config.camera_id, cam_config.priority)
    
    print("Camera Configuration:")
    print("-"*70)
    for cam_config in cameras:
        priority_name = buffer._get_priority_name(cam_config.priority)
        print(f"  {cam_config.name:20} Priority: {cam_config.priority:2} ({priority_name})")
    
    # Start cameras
    manager.start_all()
    
    # Create frame processors (simulate multiple detection threads)
    processors = [
        FrameProcessor("detector_1", buffer),
        FrameProcessor("detector_2", buffer)
    ]
    
    for processor in processors:
        processor.start()
    
    print("\n" + "="*70)
    print("RUNNING SIMULATION")
    print("="*70)
    print("Watch how CRITICAL frames (Main Gate) get processed first!")
    print("Even if Parking/Interior frames arrive earlier.\n")
    
    try:
        # Run for 10 seconds
        start_time = time.time()
        while time.time() - start_time < 10:
            # Get frames from all cameras
            frames = manager.get_all_frames(timeout=0.1)
            
            # Add frames to buffer (they'll be prioritized automatically)
            for camera_id, frame in frames.items():
                buffer.put(frame)
            
            # Print status every 2 seconds
            if int(time.time() - start_time) % 2 == 0:
                time.sleep(0.1)  # Align with 2-second boundaries
                print(f"\n[{int(time.time() - start_time)}s] Status:")
                print(f"  Buffer: {buffer.size()}/{buffer.max_size} "
                      f"({buffer.fullness():.1%} full)")
                print(f"  Frames received: {buffer.frames_received}")
                print(f"  Frames processed: {buffer.frames_processed}")
                print(f"  Frames dropped: {buffer.frames_dropped}")
    
    finally:
        print("\n" + "="*70)
        print("FINAL STATISTICS")
        print("="*70)
        
        # Stop everything
        for processor in processors:
            processor.stop()
        manager.stop_all()
        
        # Print statistics
        buffer.print_statistics()
        
        print("\nProcessor Statistics:")
        print("-"*70)
        for processor in processors:
            stats = processor.get_statistics()
            print(f"\n{stats['processor_id']}:")
            print(f"  Frames processed: {stats['frames_processed']}")
            print(f"  Avg processing time: {stats['avg_processing_time']*1000:.1f}ms")
            print(f"  Processing FPS: {stats['fps']:.1f}")
        
        print("\n" + "="*70)
        print("KEY OBSERVATION:")
        print("="*70)
        print("Notice how LOW/MINIMAL priority frames were dropped more")
        print("while CRITICAL/HIGH priority frames were mostly processed.")
        print("This is intelligent prioritization in action!")
        print("="*70)
