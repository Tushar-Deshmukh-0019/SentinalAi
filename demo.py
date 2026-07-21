"""
Animal Detection Demo

Demonstrates the false-positive filtering system.

The REAL power: Combining person + vehicle + animal detection
to create an intelligent surveillance system that doesn't cry wolf.
"""

import cv2
import sys
from pathlib import Path

from detector import AnimalDetector
from config import AnimalDetectionConfig


def demo_triple_detection(image_path: str):
    """
    Demo: Combined Person + Vehicle + Animal Detection.
    
    This is the COMPLETE Layer 0 system in action.
    Shows how all three detectors work together.
    """
    print("\n" + "="*70)
    print("DEMO: COMPLETE LAYER 0 - PERSON + VEHICLE + ANIMAL DETECTION")
    print("="*70)
    print("\nThis demonstrates the full false-positive filtering system.")
    print("The holy grail of surveillance: ONLY alert on real threats.\n")
    
    # Import other detectors
    try:
        sys.path.append(str(Path(__file__).parent.parent / 'person'))
        sys.path.append(str(Path(__file__).parent.parent / 'vehicle'))
        from detector import PersonDetector
        from detector import VehicleDetector as VehicleDetectorImport
    except ImportError:
        print("\n⚠ Person/Vehicle detectors not available.")
        print("Install Day 1 and Day 2 modules first.")
        return
    
    # Initialize all detectors
    print("Loading detection systems...")
    person_detector = PersonDetector()
    vehicle_detector = VehicleDetectorImport()
    animal_detector = AnimalDetector()
    print("✓ All systems loaded\n")
    
    print(f"Analyzing: {image_path}\n")
    
    # Run all detections
    print("Running detection pipeline...")
    persons_raw, _ = person_detector.detect(image_path)
    vehicles, _ = vehicle_detector.detect(image_path)
    animals, _ = animal_detector.detect(image_path)
    
    print(f"  Raw detections:")
    print(f"    Persons (raw):  {len(persons_raw)}")
    print(f"    Vehicles:       {len(vehicles)}")
    print(f"    Animals:        {len(animals)}\n")
    
    # CRITICAL: Resolve person/animal conflicts
    print("Resolving person/animal conflicts...")
    persons_filtered, animals_used = animal_detector.resolve_conflict_with_person(
        animals, persons_raw
    )
    
    print(f"  After conflict resolution:")
    print(f"    Persons (confirmed): {len(persons_filtered)}")
    print(f"    Animals (filtered):  {len(animals_used)}\n")
    
    # Final assessment
    print("=" * 70)
    print("FINAL SURVEILLANCE ASSESSMENT")
    print("=" * 70)
    
    total_alerts = len(persons_filtered) + len(vehicles)
    filtered_count = len(persons_raw) - len(persons_filtered)
    
    print(f"\nTotal Alerts: {total_alerts}")
    print(f"False Positives Filtered: {filtered_count}")
    
    if total_alerts == 0:
        print("\n✓ NO THREATS DETECTED")
        print("  - All activity identified as wildlife")
        print("  - No operator alert required")
        print("  - System maintaining watch")
    else:
        print(f"\n⚠ {total_alerts} CONFIRMED DETECTION(S)")
        
        if len(persons_filtered) > 0:
            print(f"\n  PERSONS ({len(persons_filtered)}):")
            for i, person in enumerate(persons_filtered, 1):
                print(f"    #{i}: Confidence {person.confidence:.0%}, Location {person.center}")
        
        if len(vehicles) > 0:
            print(f"\n  VEHICLES ({len(vehicles)}):")
            for i, vehicle in enumerate(vehicles, 1):
                print(f"    #{i}: {vehicle.tactical_summary}")
    
    if len(animals) > 0:
        print(f"\n  WILDLIFE DETECTED ({len(animals)}):")
        for i, animal in enumerate(animals, 1):
            status = "FILTERED" if animal.should_filter else "LOGGED"
            print(f"    #{i}: [{status}] {animal.summary}")
    
    # Tactical recommendations
    print(f"\n{'=' * 70}")
    print("TACTICAL RECOMMENDATIONS")
    print("=" * 70)
    
    if total_alerts == 0 and len(animals) > 0:
        print("\n✓ CONTINUE ROUTINE MONITORING")
        print("  Reason: Only wildlife detected")
        print("  Action: Log wildlife activity")
        print("  Operator: No alert needed")
    elif total_alerts > 0:
        print("\n⚠ OPERATOR ATTENTION REQUIRED")
        print(f"  Confirmed threats: {total_alerts}")
        if len(persons_filtered) > 0 and len(vehicles) > 0:
            print(f"  Profile: {len(persons_filtered)} person(s) with {len(vehicles)} vehicle(s)")
            print("  Priority: HIGH - Person-vehicle correlation")
        elif len(persons_filtered) > 0:
            print(f"  Profile: {len(persons_filtered)} person(s) on foot")
            print("  Priority: MODERATE - No vehicle detected")
        elif len(vehicles) > 0:
            print(f"  Profile: {len(vehicles)} vehicle(s) without visible persons")
            print("  Priority: MODERATE - Monitor for occupants")
    
    # Show visualization
    frame = cv2.imread(image_path)
    if frame is not None:
        # Draw all detections on one frame
        _, person_viz = person_detector.detect(image_path, visualize=True)
        
        # Draw animals
        for animal in animals:
            x1, y1, x2, y2 = animal.bbox
            color = (0, 255, 0) if animal.should_filter else (0, 165, 255)
            cv2.rectangle(person_viz, (x1, y1), (x2, y2), color, 2)
            label = f"{animal.animal_type.display_name}"
            if animal.should_filter:
                label += " [FILTERED]"
            cv2.putText(
                person_viz, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        # Draw vehicles
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle.bbox
            cv2.rectangle(person_viz, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(
                person_viz, f"Vehicle: {vehicle.vehicle_type.display_name}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2
            )
        
        cv2.imshow("Complete Layer 0 Detection - Press any key", person_viz)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def demo_animal_only(image_path: str):
    """Demo: Animal detection only."""
    print("\n" + "="*60)
    print("DEMO: Animal Detection")
    print("="*60)
    
    config = AnimalDetectionConfig(
        confidence_threshold=0.40,
        device='cuda'
    )
    detector = AnimalDetector(config)
    
    print(f"\nAnalyzing: {image_path}\n")
    
    detections, visualized = detector.detect(image_path, visualize=True)
    
    print(f"{'─'*60}")
    print(f"DETECTION RESULTS:")
    print(f"{'─'*60}")
    print(f"Total animals detected: {len(detections)}")
    
    if len(detections) > 0:
        filtered = [d for d in detections if d.should_filter]
        alerts = [d for d in detections if not d.should_filter]
        
        print(f"  Filtered (no alert): {len(filtered)}")
        print(f"  Alerts (operator notified): {len(alerts)}")
        
        print(f"\nDetailed breakdown:")
        for i, det in enumerate(detections, 1):
            print(f"\n  Animal #{i}:")
            print(f"    {det.summary}")
            print(f"    Type: {det.animal_type.display_name}")
            print(f"    Size: {det.animal_size.name}")
            print(f"    Confidence: {det.confidence:.2%}")
            print(f"    Location: {det.center}")
            print(f"    Threat Level: {det.characteristics.threat_level.name}")
            
            if det.should_filter:
                print(f"    ✓ FILTERED: {det.characteristics.filter_reason}")
            else:
                print(f"    ⚠ ALERT: {det.characteristics.threat_level.alert_message}")
    else:
        print("\n  No animals detected.")
    
    # Statistics
    stats = detector.get_statistics()
    print(f"\n{'─'*60}")
    print(f"STATISTICS:")
    print(f"{'─'*60}")
    print(f"  Filter Rate: {stats['filter_rate']:.1f}%")
    print(f"  Animal Counts: {stats['animal_counts']}")
    
    if visualized is not None:
        cv2.imshow("Animal Detection - Press any key", visualized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def demo_false_positive_scenario():
    """
    Demo: Simulated false positive scenario.
    
    Shows the before/after of animal detection filtering.
    """
    print("\n" + "="*70)
    print("DEMO: FALSE POSITIVE FILTERING SCENARIO")
    print("="*70)
    
    print("""
SCENARIO:
---------
Location: Border Perimeter, Sector 3
Time: 02:34 AM
Weather: Clear night
Expected Activity: None (no scheduled patrol)

Motion sensor triggered on Camera #11...
    """)
    
    image_path = input("Enter image path to analyze: ").strip()
    if not image_path or not Path(image_path).exists():
        print("\n⚠ No valid path provided.")
        return
    
    # Without animal detection
    print("\n" + "─"*70)
    print("WITHOUT ANIMAL DETECTION (Day 1-2 only):")
    print("─"*70)
    
    try:
        sys.path.append(str(Path(__file__).parent.parent / 'person'))
        from detector import PersonDetector
        person_detector = PersonDetector()
        
        persons_raw, _ = person_detector.detect(image_path)
        
        print(f"\nPerson Detection: {len(persons_raw)} detection(s)")
        if len(persons_raw) > 0:
            for person in persons_raw:
                print(f"  - Confidence: {person.confidence:.0%}")
                if person.confidence > 0.45:
                    print(f"    → ALERT OPERATOR ⚠")
                    print(f"    → Operator woken up at 2:34 AM")
        
        print(f"\n❌ PROBLEM:")
        print(f"  - Low confidence detection (could be animal)")
        print(f"  - No way to verify if person or animal")
        print(f"  - Must alert operator to be safe")
        print(f"  - Results in false alarm fatigue")
    except:
        print("Person detector not available")
    
    # With animal detection
    print("\n" + "─"*70)
    print("WITH ANIMAL DETECTION (Day 1-3 complete):")
    print("─"*70)
    
    animal_detector = AnimalDetector()
    animals, _ = animal_detector.detect(image_path, time_of_day="night")
    
    print(f"\nAnimal Detection: {len(animals)} detection(s)")
    if len(animals) > 0:
        for animal in animals:
            print(f"  - {animal.animal_type.display_name}: {animal.confidence:.0%}")
            if animal.should_filter:
                print(f"    → FILTERED ✓ (wildlife)")
    
    # Conflict resolution
    if len(persons_raw) > 0 and len(animals) > 0:
        persons_filtered, _ = animal_detector.resolve_conflict_with_person(
            animals, persons_raw
        )
        
        print(f"\nConflict Resolution:")
        print(f"  Raw person detections: {len(persons_raw)}")
        print(f"  After animal filtering: {len(persons_filtered)}")
        print(f"  False positives prevented: {len(persons_raw) - len(persons_filtered)}")
        
        if len(persons_filtered) == 0:
            print(f"\n✓ RESULT: No alert needed")
            print(f"  - Identified as wildlife")
            print(f"  - Operator continues sleeping")
            print(f"  - System credibility maintained")
            print(f"\n✓ SYSTEM WORKING AS DESIGNED")
        else:
            print(f"\n⚠ RESULT: Alert operator")
            print(f"  - Confirmed human presence")
            print(f"  - Animal explanation ruled out")
            print(f"  - Legitimate alert")


def demo_wildlife_statistics():
    """
    Demo: Wildlife activity logging and statistics.
    
    Shows how the system builds wildlife activity intelligence.
    """
    print("\n" + "="*60)
    print("DEMO: Wildlife Activity Intelligence")
    print("="*60)
    
    print("""
This demo shows how animal detection provides operational intelligence
beyond just filtering false positives.

Wildlife patterns reveal:
- Migration routes
- Active times
- Seasonal changes
- Environmental conditions
    """)
    
    detector = AnimalDetector()
    
    # Simulated data (in production, this comes from database)
    print("\nSample Wildlife Log (Last 24 hours):")
    print("─" * 60)
    
    sample_logs = [
        ("02:15 AM", "Deer", 3, "Sector 3 - Forest Edge"),
        ("02:34 AM", "Deer", 1, "Sector 3 - Forest Edge"),
        ("03:47 AM", "Bear", 1, "Sector 5 - North Perimeter"),
        ("06:12 AM", "Bird", 12, "Multiple Sectors"),
        ("18:45 PM", "Dog", 1, "Sector 1 - With patrol"),
        ("21:33 PM", "Deer", 2, "Sector 3 - Forest Edge"),
        ("23:58 PM", "Coyote", 1, "Sector 4 - Open Field"),
    ]
    
    for time, animal, count, location in sample_logs:
        print(f"{time:>10} | {animal:>8} x{count} | {location}")
    
    print(f"\n{'─' * 60}")
    print("INTELLIGENCE ANALYSIS:")
    print("─" * 60)
    
    print("""
Key Findings:
  1. Deer activity concentrated in Sector 3 (forest edge)
     → Known migration path through this sector
     → Expected pattern
  
  2. Peak activity 02:00-04:00 AM
     → Nocturnal wildlife most active
     → Adjust person detection sensitivity
  
  3. Bear sighting in Sector 5 (03:47 AM)
     → Logged for safety awareness
     → Alert patrol units
  
  4. Dog with patrol (18:45 PM)
     → K-9 unit, expected
     → Correctly not filtered
  
Operational Recommendations:
  - Expect higher false trigger rate 02:00-06:00 AM (wildlife active)
  - Sector 3 requires careful person/animal distinction
  - Consider additional lighting in high-traffic wildlife areas
  - Brief patrols on bear presence in Sector 5
    """)


def main():
    """Main demo selector."""
    print("\n" + "="*70)
    print("SentinelAI - Animal Detection & False Positive Filtering")
    print("="*70)
    print("\nLayer 0 is now COMPLETE: Person + Vehicle + Animal")
    print("The system can now distinguish threats from wildlife.")
    
    print("\nAvailable demos:")
    print("  1. Animal detection only")
    print("  2. Complete Layer 0 (Person + Vehicle + Animal)")
    print("  3. False positive filtering scenario")
    print("  4. Wildlife activity intelligence")
    
    choice = input("\nSelect demo (1-4) or 'q' to quit: ").strip()
    
    if choice == '1':
        image_path = input("Enter image path: ").strip()
        if not image_path:
            print("\n⚠ No image specified.")
            return
        demo_animal_only(image_path)
    
    elif choice == '2':
        image_path = input("Enter image path: ").strip()
        if not image_path:
            print("\n⚠ No image specified.")
            return
        demo_triple_detection(image_path)
    
    elif choice == '3':
        demo_false_positive_scenario()
    
    elif choice == '4':
        demo_wildlife_statistics()
    
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
