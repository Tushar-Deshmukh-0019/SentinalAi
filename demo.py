"""
Person Detection Demo

This demonstrates the person detection system in action.

Real-world simulation:
- Processes surveillance camera feed
- Detects persons
- Shows confidence levels
- Displays performance metrics

Run this to understand how Layer 0 works.
"""

import cv2
import sys
from pathlib import Path

from detector import PersonDetector
from config import PersonDetectionConfig


def demo_image(image_path: str):
    """
    Demo: Detect persons in a single image.
    
    Scenario: Analyzing a snapshot from surveillance camera.
    """
    print("\n" + "="*60)
    print("DEMO 1: Single Image Detection")
    print("="*60)
    
    # Initialize detector with default config
    config = PersonDetectionConfig(
        confidence_threshold=0.45,
        device='cuda'  # Change to 'cpu' if no GPU
    )
    detector = PersonDetector(config)
    
    print(f"\nAnalyzing image: {image_path}")
    
    # Detect
    detections, visualized = detector.detect(image_path, visualize=True)
    
    # Results
    print(f"\n{'─'*60}")
    print(f"DETECTION RESULTS:")
    print(f"{'─'*60}")
    print(f"Total persons detected: {len(detections)}")
    
    if len(detections) > 0:
        print("\nDetailed breakdown:")
        for i, det in enumerate(detections, 1):
            confidence_level = (
                "HIGH" if det.confidence > 0.8 
                else "MEDIUM" if det.confidence > 0.6 
                else "LOW"
            )
            
            print(f"\n  Person #{i}:")
            print(f"    Confidence: {det.confidence:.2%} ({confidence_level})")
            print(f"    Location: {det.center}")
            print(f"    Bounding Box: {det.bbox}")
            print(f"    Size: {det.area} pixels²")
            
            # Simulated threat assessment
            if det.confidence > 0.7:
                print(f"    ✓ Valid detection - proceed to behavior analysis")
            else:
                print(f"    ⚠ Low confidence - requires verification")
    else:
        print("\n  No persons detected in frame.")
        print("  Possible reasons:")
        print("    - Empty surveillance zone")
        print("    - Objects mistaken for persons filtered out")
        print("    - Persons too distant/small")
    
    # Performance
    stats = detector.get_performance_stats()
    print(f"\n{'─'*60}")
    print(f"PERFORMANCE METRICS:")
    print(f"{'─'*60}")
    print(f"  Inference Time: {stats['avg_inference_time_ms']:.1f}ms")
    print(f"  Processing Speed: {stats['avg_fps']:.1f} FPS")
    
    if stats['avg_fps'] >= 30:
        print(f"  ✓ Real-time capable (>30 FPS)")
    else:
        print(f"  ⚠ Below real-time threshold")
    
    # Show visualization
    if visualized is not None:
        cv2.imshow("Person Detection - Press any key to continue", visualized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def demo_webcam():
    """
    Demo: Real-time person detection from webcam.
    
    Scenario: Live surveillance feed monitoring.
    Simulates what happens at an actual surveillance station.
    """
    print("\n" + "="*60)
    print("DEMO 2: Real-Time Webcam Detection")
    print("="*60)
    print("\nSimulating live surveillance feed...")
    print("Press 'q' to quit, 's' to save screenshot")
    
    # Initialize detector
    config = PersonDetectionConfig(
        confidence_threshold=0.45,
        device='cuda',
        skip_frames=0  # Process every frame
    )
    detector = PersonDetector(config)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("\n❌ Error: Could not access webcam")
        print("This demo requires a connected camera.")
        return
    
    print("\n✓ Camera connected - starting surveillance...")
    
    frame_count = 0
    alert_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n❌ Failed to read from camera")
            break
        
        frame_count += 1
        
        # Detect persons
        detections, visualized = detector.detect(frame, visualize=True)
        
        # Alert logic (simulated)
        if len(detections) > 0:
            alert_count += 1
            
            # In real system, this would:
            # 1. Log to database
            # 2. Start tracking
            # 3. Begin behavior analysis
            # 4. Check against patrol schedule
            # 5. Calculate threat score
            
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"\n[ALERT] Frame {frame_count}: {len(detections)} person(s) detected")
                for det in detections:
                    print(f"  - Confidence: {det.confidence:.2%} at {det.center}")
        
        # Display
        if visualized is not None:
            cv2.imshow("SentinelAI - Live Surveillance", visualized)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"sentinel_capture_{frame_count}.jpg"
            cv2.imwrite(filename, visualized)
            print(f"\n📸 Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    stats = detector.get_performance_stats()
    print(f"\n{'='*60}")
    print(f"SESSION SUMMARY:")
    print(f"{'='*60}")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with detections: {alert_count}")
    print(f"Detection rate: {alert_count/frame_count*100:.1f}%")
    print(f"Average FPS: {stats['avg_fps']:.1f}")
    print(f"Average inference time: {stats['avg_inference_time_ms']:.1f}ms")


def demo_video(video_path: str):
    """
    Demo: Analyze recorded surveillance footage.
    
    Scenario: Post-incident analysis or testing on recorded footage.
    """
    print("\n" + "="*60)
    print("DEMO 3: Video File Analysis")
    print("="*60)
    
    # Initialize detector
    config = PersonDetectionConfig(
        confidence_threshold=0.45,
        device='cuda'
    )
    detector = PersonDetector(config)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\n❌ Error: Could not open video file: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\nVideo information:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {total_frames/fps:.1f} seconds")
    
    print("\nAnalyzing... (press 'q' to stop)")
    
    detection_timeline = []
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # Detect
        detections, visualized = detector.detect(frame, visualize=True)
        
        # Record timeline
        detection_timeline.append({
            'frame': frame_num,
            'timestamp': frame_num / fps,
            'count': len(detections),
            'detections': detections
        })
        
        # Display progress
        if frame_num % 30 == 0:
            progress = frame_num / total_frames * 100
            print(f"  Progress: {progress:.1f}% - Frame {frame_num}/{total_frames}")
        
        # Show frame
        if visualized is not None:
            cv2.imshow("Video Analysis", visualized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Analyze timeline
    print(f"\n{'='*60}")
    print(f"ANALYSIS RESULTS:")
    print(f"{'='*60}")
    
    frames_with_persons = sum(1 for entry in detection_timeline if entry['count'] > 0)
    max_persons = max(entry['count'] for entry in detection_timeline)
    
    print(f"Frames analyzed: {len(detection_timeline)}")
    print(f"Frames with persons: {frames_with_persons} ({frames_with_persons/len(detection_timeline)*100:.1f}%)")
    print(f"Maximum persons in single frame: {max_persons}")
    
    # Key moments
    print(f"\nKey moments (person detected):")
    for entry in detection_timeline[:10]:  # Show first 10
        if entry['count'] > 0:
            print(f"  Time {entry['timestamp']:.1f}s: {entry['count']} person(s)")


def main():
    """Main demo selector."""
    print("\n" + "="*60)
    print("SentinelAI - Person Detection System Demo")
    print("="*60)
    print("\nThis demonstrates Layer 0: Person Detection")
    print("The foundation of the multi-layer threat assessment system.")
    
    print("\nAvailable demos:")
    print("  1. Single image detection")
    print("  2. Real-time webcam detection")
    print("  3. Video file analysis")
    print("  4. Batch processing (multiple images)")
    
    choice = input("\nSelect demo (1-4) or 'q' to quit: ").strip()
    
    if choice == '1':
        image_path = input("Enter image path (or press Enter for default): ").strip()
        if not image_path:
            print("\n⚠ No test image specified.")
            print("Please provide an image path or download test data.")
            return
        demo_image(image_path)
    
    elif choice == '2':
        demo_webcam()
    
    elif choice == '3':
        video_path = input("Enter video path: ").strip()
        if not video_path:
            print("\n⚠ No video file specified.")
            return
        demo_video(video_path)
    
    elif choice == '4':
        print("\n⚠ Batch processing demo not yet implemented.")
        print("This will be added in Day 2 improvements.")
    
    elif choice.lower() == 'q':
        print("\nExiting demo.")
    
    else:
        print("\n❌ Invalid choice.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
