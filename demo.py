"""
ByteTrack Integration Demo

Demonstrates:
1. Single frame tracking
2. Multi-frame tracking with ID persistence
3. Occlusion handling
4. Track confirmation
5. Performance metrics
6. Integration with detection pipeline
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.tracking.bytetrack import ByteTracker
from ai.logging import setup_logger

logger = setup_logger('tracking.demo')


def demo_basic_tracking():
    """Demo 1: Basic single-frame tracking"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Single-Frame Tracking")
    print("="*70)
    
    tracker = ByteTracker(frame_rate=30)
    
    # Simulate 5 detections [x1, y1, x2, y2, confidence]
    detections = np.array([
        [100, 100, 150, 180, 0.95],  # Person 1
        [200, 150, 250, 250, 0.92],  # Person 2
        [400, 200, 450, 300, 0.88],  # Person 3
        [50, 300, 100, 400, 0.85],   # Person 4
        [500, 400, 550, 500, 0.82],  # Person 5
    ])
    
    logger.info(f"Processing {len(detections)} detections")
    tracked_objects = tracker.update(detections)
    
    logger.info(f"Tracked objects: {len(tracked_objects)}")
    for obj in tracked_objects:
        logger.info(f"  ID {obj['track_id']}: conf={obj['confidence']:.2f}, "
                   f"pos=({obj['position']['x']:.0f}, {obj['position']['y']:.0f})")


def demo_multi_frame_tracking():
    """Demo 2: Multi-frame tracking with ID persistence"""
    print("\n" + "="*70)
    print("DEMO 2: Multi-Frame Tracking (ID Persistence)")
    print("="*70)
    
    tracker = ByteTracker(frame_rate=30)
    
    # Simulate 5 frames
    frames_data = [
        # Frame 1: 3 people
        np.array([
            [100, 100, 150, 180, 0.95],
            [200, 150, 250, 250, 0.92],
            [400, 200, 450, 300, 0.88],
        ]),
        # Frame 2: Same people, moved slightly
        np.array([
            [102, 102, 152, 182, 0.94],  # Person 1 moved right/down
            [202, 152, 252, 252, 0.91],  # Person 2 moved right/down
            [402, 202, 452, 302, 0.87],  # Person 3 moved right/down
        ]),
        # Frame 3: Person 2 occluded, new person appears
        np.array([
            [104, 104, 154, 184, 0.93],  # Person 1 moved more
            # Person 2 not detected (occluded)
            [404, 204, 454, 304, 0.86],  # Person 3 moved more
            [300, 300, 350, 400, 0.90],  # NEW person
        ]),
        # Frame 4: Person 2 reappears, Person 3 no detection
        np.array([
            [106, 106, 156, 186, 0.92],  # Person 1
            [204, 154, 254, 254, 0.90],  # Person 2 (back from occlusion)
            # Person 3 not detected
            [302, 302, 352, 402, 0.89],  # NEW person (or Person 4)
        ]),
        # Frame 5: All people visible again
        np.array([
            [108, 108, 158, 188, 0.91],  # Person 1
            [206, 156, 256, 256, 0.89],  # Person 2
            [406, 206, 456, 306, 0.85],  # Person 3 (back from occlusion)
            [304, 304, 354, 404, 0.88],  # NEW person
        ]),
    ]
    
    for frame_idx, detections in enumerate(frames_data):
        logger.info(f"\nFrame {frame_idx + 1}:")
        logger.info(f"  Input detections: {len(detections)}")
        
        tracked_objects = tracker.update(detections)
        
        logger.info(f"  Tracked objects: {len(tracked_objects)}")
        for obj in tracked_objects:
            status = "✓ CONFIRMED" if obj['is_confirmed'] else "⊘ TENTATIVE"
            logger.info(f"    ID {obj['track_id']:2d} ({obj['hits']} hits) {status}: "
                       f"conf={obj['confidence']:.2f}")


def demo_occlusion_handling():
    """Demo 3: Occlusion and re-detection handling"""
    print("\n" + "="*70)
    print("DEMO 3: Occlusion & Re-Detection Handling")
    print("="*70)
    
    tracker = ByteTracker(frame_rate=30)
    
    # Person walks across frame, gets occluded, then reappears
    frames_data = [
        # Frames 1-3: Person 1 walks left to right
        np.array([[100, 200, 150, 300, 0.95]]),
        np.array([[150, 200, 200, 300, 0.94]]),
        np.array([[200, 200, 250, 300, 0.93]]),
        # Frames 4-6: Person occluded (behind obstacle)
        np.array([]),  # No detection
        np.array([]),  # No detection
        np.array([]),  # No detection
        # Frames 7-10: Person reappears on other side
        np.array([[300, 200, 350, 300, 0.92]]),
        np.array([[350, 200, 400, 300, 0.91]]),
        np.array([[400, 200, 450, 300, 0.90]]),
        np.array([[450, 200, 500, 300, 0.89]]),
    ]
    
    logger.info("Scenario: Person walks across frame, gets occluded, reappears")
    
    for frame_idx, detections in enumerate(frames_data):
        status = "Occluded" if len(detections) == 0 else "Visible"
        logger.info(f"\nFrame {frame_idx + 1} ({status}):")
        
        tracked_objects = tracker.update(detections)
        
        if len(tracked_objects) > 0:
            for obj in tracked_objects:
                logger.info(f"  Track {obj['track_id']}: "
                           f"x={obj['position']['x']:.0f}, "
                           f"hits={obj['hits']}")
        else:
            logger.info("  No tracked objects (still monitoring lost tracks)")
        
        stats = tracker.get_statistics()
        logger.info(f"  Stats: Active={stats['active_tracks']}, "
                   f"Confirmed={stats['confirmed_tracks']}, "
                   f"Tentative={stats['tentative_tracks']}")


def demo_crowded_scene():
    """Demo 4: Crowded scene with many people"""
    print("\n" + "="*70)
    print("DEMO 4: Crowded Scene Tracking")
    print("="*70)
    
    tracker = ByteTracker(frame_rate=30)
    
    # Simulate crowded scene (20 people)
    np.random.seed(42)
    
    for frame_idx in range(10):
        # Generate random detections in crowded area
        num_people = np.random.randint(15, 25)
        detections = []
        
        for _ in range(num_people):
            x1 = np.random.uniform(50, 1200)
            y1 = np.random.uniform(50, 600)
            w = np.random.uniform(40, 100)
            h = np.random.uniform(80, 200)
            conf = np.random.uniform(0.8, 0.99)
            
            detections.append([x1, y1, x1 + w, y1 + h, conf])
        
        detections = np.array(detections)
        tracked_objects = tracker.update(detections)
        
        logger.info(f"\nFrame {frame_idx + 1}:")
        logger.info(f"  Input: {len(detections)} detections")
        logger.info(f"  Output: {len(tracked_objects)} tracked objects")
        
        stats = tracker.get_statistics()
        logger.info(f"  Active: {stats['active_tracks']}, "
                   f"Confirmed: {stats['confirmed_tracks']}, "
                   f"Total created: {stats['total_tracks_created']}")


def demo_performance_metrics():
    """Demo 5: Performance metrics"""
    print("\n" + "="*70)
    print("DEMO 5: Performance Metrics")
    print("="*70)
    
    tracker = ByteTracker(frame_rate=30)
    
    import time
    
    # Run 100 frames with varying number of objects
    logger.info("Running 100 frames with 50-100 objects...")
    
    start_time = time.time()
    frame_times = []
    
    for frame_idx in range(100):
        num_objects = np.random.randint(50, 100)
        detections = []
        
        for _ in range(num_objects):
            x1 = np.random.uniform(0, 1280)
            y1 = np.random.uniform(0, 720)
            w = np.random.uniform(30, 100)
            h = np.random.uniform(60, 200)
            conf = np.random.uniform(0.7, 0.99)
            
            detections.append([x1, y1, x1 + w, y1 + h, conf])
        
        detections = np.array(detections)
        
        frame_start = time.time()
        tracker.update(detections)
        frame_end = time.time()
        
        frame_times.append((frame_end - frame_start) * 1000)  # ms
    
    total_time = time.time() - start_time
    
    logger.info(f"\n100 frames completed in {total_time:.2f}s")
    logger.info(f"Average frame time: {np.mean(frame_times):.2f}ms")
    logger.info(f"Min frame time: {np.min(frame_times):.2f}ms")
    logger.info(f"Max frame time: {np.max(frame_times):.2f}ms")
    logger.info(f"Average FPS: {100/total_time:.1f}")
    
    stats = tracker.get_statistics()
    logger.info(f"\nFinal Statistics:")
    logger.info(f"  Total detections processed: {stats['total_detections']}")
    logger.info(f"  Total tracks created: {stats['total_tracks_created']}")
    logger.info(f"  Active tracks: {stats['active_tracks']}")
    logger.info(f"  Confirmed tracks: {stats['confirmed_tracks']}")


def demo_track_lifecycle():
    """Demo 6: Track lifecycle (creation, confirmation, deletion)"""
    print("\n" + "="*70)
    print("DEMO 6: Track Lifecycle")
    print("="*70)
    
    tracker = ByteTracker(frame_rate=30)
    
    logger.info("Scenario: Track creation, confirmation, and deletion")
    logger.info("- Tentative: 0-2 hits")
    logger.info("- Confirmed: 3+ hits")
    logger.info("- Deleted: 30+ frames without detection")
    
    # Create a track that will be confirmed
    for frame in range(5):
        if frame < 3:
            # Create new track
            detections = np.array([[100, 100, 150, 200, 0.9]])
        elif frame < 15:
            # Track confirmed after 3 frames
            detections = np.array([[100 + frame*2, 100, 150 + frame*2, 200, 0.9]])
        else:
            # Track disappears
            detections = np.array([])
        
        tracked = tracker.update(detections)
        
        if len(tracked) > 0:
            obj = tracked[0]
            status = "CONFIRMED" if obj['is_confirmed'] else "TENTATIVE"
            logger.info(f"Frame {frame + 1}: Track {obj['track_id']} - "
                       f"{status} (hits={obj['hits']})")
        else:
            logger.info(f"Frame {frame + 1}: No confirmed tracks (or track deleted)")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("ByteTrack Multi-Object Tracking - Demonstrations")
    print("Day 13: ByteTrack Integration")
    print("="*70)
    
    try:
        demo_basic_tracking()
        demo_multi_frame_tracking()
        demo_occlusion_handling()
        demo_crowded_scene()
        demo_performance_metrics()
        demo_track_lifecycle()
        
        print("\n" + "="*70)
        print("[SUCCESS] All demos completed successfully!")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
