"""
Vehicle Detection Demo

Demonstrates vehicle detection in action.

Real-world simulation:
- Processes surveillance camera feed
- Detects and classifies vehicles
- Shows tactical information
- Simulates threat assessment

Run this to understand how vehicle detection integrates with the system.
"""

import cv2
import sys
from pathlib import Path

from detector import VehicleDetector
from config import VehicleDetectionConfig
from classifier import VehicleType


def demo_combined_detection(image_path: str):
    """
    Demo: Combined person + vehicle detection.
    
    Scenario: The real power - detecting BOTH persons and vehicles.
    This is how the system correlates vehicles with occupants.
    """
    print("\n" + "="*60)
    print("DEMO: Combined Person + Vehicle Detection")
    print("="*60)
    print("\nThis demonstrates how vehicle and person detection work together.")
    print("In real surveillance, this correlation is CRITICAL.")
    
    # Import person detector
    try:
        sys.path.append(str(Path(__file__).parent.parent / 'person'))
        from detector import PersonDetector
    except ImportError:
        print("\n⚠ Person detector not available. Install Day 1 module first.")
        return
    
    # Initialize both detectors
    vehicle_config = VehicleDetectionConfig(
        confidence_threshold=0.50,
        device='cuda'
    )
    vehicle_detector = VehicleDetector(vehicle_config)
    person_detector = PersonDetector()
    
    print(f"\nAnalyzing image: {image_path}")
    
    # Detect vehicles
    vehicles, _ = vehicle_detector.detect(image_path)
    
    # Detect persons
    persons, _ = person_detector.detect(image_path)
    
    # Results
    print(f"\n{'─'*60}")
    print(f"DETECTION RESULTS:")
    print(f"{'─'*60}")
    print(f"Vehicles: {len(vehicles)}")
    print(f"Persons:  {len(persons)}")
    
    # Detailed vehicle info
    if len(vehicles) > 0:
        print(f"\nVehicle Details:")
        for i, vehicle in enumerate(vehicles, 1):
            print(f"\n  Vehicle #{i}:")
            print(f"    {vehicle.tactical_summary}")
            print(f"    Location: {vehicle.center}")
            print(f"    Size: {vehicle.width}x{vehicle.height} pixels")
            print(f"    Area: {vehicle.area} pixels²")
            
            if vehicle.has_license_plate_region:
                print(f"    License Plate: Detected ✓")
            else:
                print(f"    License Plate: Not visible")
            
            # Expected occupants
            min_occ, max_occ = vehicle.vehicle_type.typical_occupants
            print(f"    Expected occupants: {min_occ}-{max_occ}")
    
    # Tactical correlation
    print(f"\n{'─'*60}")
    print(f"TACTICAL CORRELATION:")
    print(f"{'─'*60}")
    
    if len(vehicles) == 0 and len(persons) == 0:
        print("  ✓ No activity detected")
    elif len(vehicles) == 0 and len(persons) > 0:
        print(f"  ⚠ {len(persons)} person(s) WITHOUT vehicle")
        print(f"    - Pedestrian activity")
        print(f"    - Possible infiltration on foot")
        print(f"    - Requires behavior analysis")
    elif len(vehicles) > 0 and len(persons) == 0:
        print(f"  ⚠ {len(vehicles)} vehicle(s) WITHOUT visible persons")
        print(f"    - Possible empty/unmanned vehicle")
        print(f"    - Persons may be inside (not visible)")
        print(f"    - Monitor for occupant exit")
    else:
        # Both vehicles and persons detected
        print(f"  {len(vehicles)} vehicle(s) + {len(persons)} person(s)")
        
        for vehicle in vehicles:
            min_occ, max_occ = vehicle.vehicle_type.typical_occupants
            
            if len(persons) < min_occ:
                print(f"\n  ⚠ SUSPICIOUS: {vehicle.vehicle_type.display_name} with only {len(persons)} person(s)")
                print(f"    Expected: {min_occ}-{max_occ}")
                print(f"    Possible: Persons inside vehicle, or incomplete detection")
            elif len(persons) > max_occ:
                print(f"\n  🚨 ALERT: {vehicle.vehicle_type.display_name} with {len(persons)} person(s)")
                print(f"    Expected: {min_occ}-{max_occ}")
                print(f"    Unusual occupant count - investigate")
            else:
                print(f"\n  ✓ Normal: {vehicle.vehicle_type.display_name} with {len(persons)} person(s)")
                print(f"    Within expected range: {min_occ}-{max_occ}")
    
    # Combined visualization
    frame = cv2.imread(image_path)
    if frame is not None:
        # Draw vehicles
        _, viz = vehicle_detector.detect(image_path, visualize=True)
        
        # Draw persons on same frame
        for person in persons:
            x1, y1, x2, y2 = person.bbox
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                viz, f"Person {person.confidence:.0%}", 
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        cv2.imshow("Combined Detection - Press any key", viz)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def demo_vehicle_image(image_path: str):
    """
    Demo: Detect vehicles in a single image.
    
    Scenario: Analyzing a snapshot from surveillance camera.
    """
    print("\n" + "="*60)
    print("DEMO: Single Image Vehicle Detection")
    print("="*60)
    
    config = VehicleDetectionConfig(
        confidence_threshold=0.50,
        device='cuda'
    )
    detector = VehicleDetector(config)
    
    print(f"\nAnalyzing image: {image_path}")
    
    # Detect
    detections, visualized = detector.detect(image_path, visualize=True)
    
    # Results
    print(f"\n{'─'*60}")
    print(f"DETECTION RESULTS:")
    print(f"{'─'*60}")
    print(f"Total vehicles detected: {len(detections)}")
    
    if len(detections) > 0:
        print("\nDetailed breakdown:")
        for i, det in enumerate(detections, 1):
            print(f"\n  Vehicle #{i}:")
            print(f"    {det.tactical_summary}")
            print(f"    Type: {det.vehicle_type.display_name}")
            print(f"    Size: {det.vehicle_size.name}")
            print(f"    Confidence: {det.confidence:.2%}")
            print(f"    Location: {det.center}")
            print(f"    Dimensions: {det.width}x{det.height} pixels")
            print(f"    Area: {det.area} pixels²")
            
            if det.has_license_plate_region:
                print(f"    License Plate: Region detected ✓")
            else:
                print(f"    License Plate: Not visible")
            
            # Tactical assessment
            threat = det.characteristics.base_threat_level
            if threat > 70:
                print(f"    ⚠ HIGH THREAT: Investigate immediately")
            elif threat > 40:
                print(f"    ⚠ MODERATE: Monitor closely")
            else:
                print(f"    ✓ LOW THREAT: Routine monitoring")
    else:
        print("\n  No vehicles detected in frame.")
    
    # Performance
    stats = detector.get_performance_stats()
    print(f"\n{'─'*60}")
    print(f"PERFORMANCE METRICS:")
    print(f"{'─'*60}")
    print(f"  Inference Time: {stats['avg_inference_time_ms']:.1f}ms")
    print(f"  Processing Speed: {stats['avg_fps']:.1f} FPS")
    
    # Show visualization
    if visualized is not None:
        cv2.imshow("Vehicle Detection - Press any key", visualized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def demo_webcam():
    """
    Demo: Real-time vehicle detection from webcam.
    
    Scenario: Live surveillance feed monitoring.
    """
    print("\n" + "="*60)
    print("DEMO: Real-Time Vehicle Detection")
    print("="*60)
    print("\nSimulating live surveillance feed...")
    print("Press 'q' to quit, 's' to save screenshot")
    
    config = VehicleDetectionConfig(
        confidence_threshold=0.50,
        device='cuda',
        skip_frames=0
    )
    detector = VehicleDetector(config)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("\n❌ Error: Could not access webcam")
        return
    
    print("\n✓ Camera connected - starting surveillance...")
    
    frame_count = 0
    detection_log = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect vehicles
        detections, visualized = detector.detect(frame, visualize=True)
        
        if len(detections) > 0:
            timestamp = frame_count / 30  # Approximate seconds
            detection_log.append({
                'frame': frame_count,
                'time': timestamp,
                'vehicles': len(detections),
                'types': [d.vehicle_type.display_name for d in detections]
            })
            
            if frame_count % 30 == 0:
                print(f"\n[ALERT] Frame {frame_count}: {len(detections)} vehicle(s)")
                for det in detections:
                    print(f"  - {det.tactical_summary}")
        
        # Display
        if visualized is not None:
            cv2.imshow("SentinelAI - Vehicle Surveillance", visualized)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"sentinel_vehicle_{frame_count}.jpg"
            cv2.imwrite(filename, visualized)
            print(f"\n📸 Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Summary
    stats = detector.get_performance_stats()
    print(f"\n{'='*60}")
    print(f"SESSION SUMMARY:")
    print(f"{'='*60}")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with vehicles: {len(detection_log)}")
    print(f"Average FPS: {stats['avg_fps']:.1f}")


def demo_scenario():
    """
    Demo: Simulate a real border surveillance scenario.
    
    This demonstrates the tactical decision-making process.
    """
    print("\n" + "="*60)
    print("DEMO: Border Surveillance Scenario")
    print("="*60)
    
    print("""
SCENARIO:
---------
Location: Border Access Road, Sector 5
Time: 03:14 AM
Weather: Clear, low visibility (night)
Expected Activity: None (no scheduled patrol)

Camera #8 triggers motion alert...
    """)
    
    image_path = input("Enter image/video path to analyze: ").strip()
    if not image_path or not Path(image_path).exists():
        print("\n⚠ No valid path provided.")
        return
    
    print("\n🔍 Analyzing surveillance feed...")
    print("─" * 60)
    
    # Initialize detectors
    vehicle_config = VehicleDetectionConfig(
        confidence_threshold=0.45,  # Lower for night
        low_light_boost=True,
        device='cuda'
    )
    vehicle_detector = VehicleDetector(vehicle_config)
    
    # Detect
    detections, viz = vehicle_detector.detect(image_path, visualize=True)
    
    print(f"\n📊 ANALYSIS RESULTS:")
    print("─" * 60)
    print(f"Vehicles Detected: {len(detections)}")
    
    if len(detections) == 0:
        print("\n✓ No vehicles detected")
        print("  Assessment: Area clear")
        print("  Action: Continue routine monitoring")
    else:
        total_threat = 0
        for i, det in enumerate(detections, 1):
            print(f"\nVehicle #{i}:")
            print(f"  {det.tactical_summary}")
            print(f"  Location: {det.center}")
            total_threat += det.characteristics.base_threat_level
        
        avg_threat = total_threat / len(detections)
        
        print(f"\n{'─'*60}")
        print(f"THREAT ASSESSMENT:")
        print(f"{'─'*60}")
        print(f"Average Threat Score: {avg_threat:.0f}/100")
        
        # Decision matrix
        print(f"\nEVIDENCE SUMMARY:")
        print(f"  [+20] Unauthorized vehicle(s) at night")
        print(f"  [+15] No scheduled patrol")
        print(f"  [+10] Multiple vehicles" if len(detections) > 1 else "  [+05] Single vehicle")
        
        # Recommendation
        if avg_threat > 70:
            print(f"\n🚨 RECOMMENDATION: IMMEDIATE RESPONSE")
            print(f"  - Alert rapid response team")
            print(f"  - Activate additional cameras")
            print(f"  - Initiate vehicle tracking")
            print(f"  - Request operator review")
        elif avg_threat > 40:
            print(f"\n⚠ RECOMMENDATION: ELEVATED MONITORING")
            print(f"  - Continue tracking")
            print(f"  - Check authorization database")
            print(f"  - Alert operator for review")
        else:
            print(f"\n✓ RECOMMENDATION: ROUTINE MONITORING")
            print(f"  - Log incident")
            print(f"  - Continue observation")
        
        # Show visualization
        if viz is not None:
            cv2.imshow("Tactical Assessment", viz)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    """Main demo selector."""
    print("\n" + "="*60)
    print("SentinelAI - Vehicle Detection System Demo")
    print("="*60)
    print("\nThis demonstrates Layer 0: Vehicle Detection")
    print("Works alongside Person Detection for complete surveillance.")
    
    print("\nAvailable demos:")
    print("  1. Single image detection")
    print("  2. Real-time webcam detection")
    print("  3. Combined person + vehicle detection")
    print("  4. Border surveillance scenario")
    
    choice = input("\nSelect demo (1-4) or 'q' to quit: ").strip()
    
    if choice == '1':
        image_path = input("Enter image path: ").strip()
        if not image_path:
            print("\n⚠ No image specified.")
            print("Try with: https://ultralytics.com/images/bus.jpg")
            return
        demo_vehicle_image(image_path)
    
    elif choice == '2':
        demo_webcam()
    
    elif choice == '3':
        image_path = input("Enter image path: ").strip()
        if not image_path:
            print("\n⚠ No image specified.")
            return
        demo_combined_detection(image_path)
    
    elif choice == '4':
        demo_scenario()
    
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
