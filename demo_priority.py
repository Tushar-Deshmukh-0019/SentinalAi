"""
Priority Queue Demo

Demonstrates the critical difference between:
- Random processing (Day 5 only)
- Priority-based processing (Day 6)

This demo shows WHY priority matters for real-world threat detection.
"""

import time
import random
from typing import List
from dataclasses import dataclass


@dataclass
class MockFrame:
    """Mock frame for demonstration."""
    camera_id: str
    camera_name: str
    priority: int
    timestamp: float
    frame_number: int
    threat_level: str = "NONE"


def simulate_random_processing():
    """Simulate processing WITHOUT priority (Day 5 only)."""
    
    print("\n" + "="*70)
    print("SCENARIO 1: RANDOM PROCESSING (No Priority)")
    print("="*70)
    print("\nBorder Post @ 2:47 AM - Suspicious movement detected!")
    print("10 cameras running, can only process 3 frames/second\n")
    
    # Simulate 10 cameras with different priorities
    cameras = [
        ("main_gate", "Main Gate", 10, "CRITICAL - PERSON APPROACHING"),
        ("checkpoint", "Security Checkpoint", 10, "CRITICAL - VEHICLE STOPPED"),
        ("north_fence", "North Fence", 8, "HIGH - Movement detected"),
        ("south_fence", "South Fence", 8, "HIGH - All clear"),
        ("east_road", "East Approach Road", 5, "MEDIUM - Empty road"),
        ("west_road", "West Approach Road", 5, "MEDIUM - Empty road"),
        ("parking", "Parking Lot", 3, "LOW - Parked vehicles only"),
        ("back_door", "Back Entrance", 5, "MEDIUM - Closed"),
        ("interior", "Interior Hall", 2, "LOW - Empty"),
        ("equipment", "Equipment Room", 1, "MINIMAL - No activity")
    ]
    
    # Create frames in random order (how they actually arrive)
    frames = []
    for i, (cam_id, name, priority, threat) in enumerate(cameras):
        frames.append(MockFrame(
            camera_id=cam_id,
            camera_name=name,
            priority=priority,
            timestamp=time.time(),
            frame_number=i,
            threat_level=threat
        ))
    
    # Shuffle to simulate random arrival
    random.shuffle(frames)
    
    print("Frame Arrival Order (random):")
    print("-"*70)
    for i, frame in enumerate(frames):
        print(f"{i+1}. {frame.camera_name:25} (Priority: {frame.priority:2}) - {frame.threat_level}")
    
    print("\n" + "="*70)
    print("PROCESSING (Can only handle 3 frames in first second):")
    print("="*70)
    
    # Process first 3 frames
    processing_time = 0
    for i in range(min(3, len(frames))):
        frame = frames[i]
        print(f"\n[{processing_time}ms] Processing {frame.camera_name}...")
        print(f"  Priority: {frame.priority}")
        print(f"  Status: {frame.threat_level}")
        
        if "CRITICAL" in frame.threat_level and "PERSON" in frame.threat_level:
            print("  ⚠️  THREAT DETECTED AT MAIN GATE!")
            print(f"  Detection time: {processing_time}ms after movement started")
        
        processing_time += 333  # 3 FPS = 333ms per frame
    
    print("\n" + "="*70)
    print("RESULT:")
    print("="*70)
    
    # Check if critical threat was in first 3
    critical_frame = next((f for f in frames if "PERSON" in f.threat_level), None)
    if critical_frame:
        position = frames.index(critical_frame) + 1
        if position <= 3:
            print(f"✓ Critical threat detected in position {position}")
            print(f"  Detection delay: {(position-1) * 333}ms")
        else:
            print(f"❌ CRITICAL THREAT MISSED!")
            print(f"  Main Gate threat was in position {position}")
            print(f"  Would be detected after {(position-1) * 333}ms")
            print(f"  By then, intruder may have already entered! ⚠️")
    
    return frames


def simulate_priority_processing():
    """Simulate processing WITH priority queue (Day 6)."""
    
    print("\n\n" + "="*70)
    print("SCENARIO 2: PRIORITY-BASED PROCESSING (With Day 6)")
    print("="*70)
    print("\nSAME SCENARIO - But now with intelligent priority queue\n")
    
    # Same cameras
    cameras = [
        ("main_gate", "Main Gate", 10, "CRITICAL - PERSON APPROACHING"),
        ("checkpoint", "Security Checkpoint", 10, "CRITICAL - VEHICLE STOPPED"),
        ("north_fence", "North Fence", 8, "HIGH - Movement detected"),
        ("south_fence", "South Fence", 8, "HIGH - All clear"),
        ("east_road", "East Approach Road", 5, "MEDIUM - Empty road"),
        ("west_road", "West Approach Road", 5, "MEDIUM - Empty road"),
        ("parking", "Parking Lot", 3, "LOW - Parked vehicles only"),
        ("back_door", "Back Entrance", 5, "MEDIUM - Closed"),
        ("interior", "Interior Hall", 2, "LOW - Empty"),
        ("equipment", "Equipment Room", 1, "MINIMAL - No activity")
    ]
    
    # Create frames
    frames = []
    for i, (cam_id, name, priority, threat) in enumerate(cameras):
        frames.append(MockFrame(
            camera_id=cam_id,
            camera_name=name,
            priority=priority,
            timestamp=time.time(),
            frame_number=i,
            threat_level=threat
        ))
    
    # Shuffle to show they arrive randomly
    random.shuffle(frames)
    
    print("Frame Arrival Order (random):")
    print("-"*70)
    for i, frame in enumerate(frames):
        print(f"{i+1}. {frame.camera_name:25} (Priority: {frame.priority:2}) - {frame.threat_level}")
    
    # But NOW we sort by priority before processing
    frames_sorted = sorted(frames, key=lambda f: (-f.priority, f.timestamp))
    
    print("\n" + "="*70)
    print("PRIORITY QUEUE REORDERS FRAMES:")
    print("="*70)
    print("\nProcessing Order (priority-sorted):")
    print("-"*70)
    for i, frame in enumerate(frames_sorted):
        print(f"{i+1}. {frame.camera_name:25} (Priority: {frame.priority:2}) - {frame.threat_level}")
    
    print("\n" + "="*70)
    print("PROCESSING (Can only handle 3 frames in first second):")
    print("="*70)
    
    # Process first 3 frames (now they're prioritized!)
    processing_time = 0
    for i in range(min(3, len(frames_sorted))):
        frame = frames_sorted[i]
        print(f"\n[{processing_time}ms] Processing {frame.camera_name}...")
        print(f"  Priority: {frame.priority}")
        print(f"  Status: {frame.threat_level}")
        
        if "CRITICAL" in frame.threat_level and "PERSON" in frame.threat_level:
            print("  🚨 THREAT DETECTED AT MAIN GATE!")
            print(f"  Detection time: {processing_time}ms after movement started")
            print("  ✓ FAST RESPONSE POSSIBLE!")
        
        processing_time += 333
    
    print("\n" + "="*70)
    print("RESULT:")
    print("="*70)
    
    # Check position of critical threat
    critical_frame = next((f for f in frames_sorted if "PERSON" in f.threat_level), None)
    if critical_frame:
        position = frames_sorted.index(critical_frame) + 1
        print(f"✓ Critical threat detected in position {position}")
        print(f"  Detection delay: {(position-1) * 333}ms")
        print(f"  Alert sent while intruder still at entrance ✓")
        print(f"  Response team can intercept before entry ✓")


def compare_drop_rates():
    """Compare frame drop rates with and without priority."""
    
    print("\n\n" + "="*70)
    print("SCENARIO 3: BUFFER OVERFLOW - What Gets Dropped?")
    print("="*70)
    print("\nSystem overload: 300 frames/second incoming, can only process 60")
    print("240 frames MUST be dropped. Which ones?\n")
    
    # Simulate frame arrivals
    cameras = {
        "Main Gate": (10, "CRITICAL"),
        "Checkpoint": (10, "CRITICAL"),
        "North Fence": (8, "HIGH"),
        "South Fence": (8, "HIGH"),
        "East Road": (5, "MEDIUM"),
        "West Road": (5, "MEDIUM"),
        "Parking": (3, "LOW"),
        "Interior": (2, "LOW"),
        "Equipment": (1, "MINIMAL")
    }
    
    print("WITHOUT Priority (Random Dropping):")
    print("-"*70)
    print("When buffer is full, drop ANY frame randomly")
    print("\nExpected drops:")
    total_frames = 300
    drop_count = 240
    for name, (priority, level) in cameras.items():
        # Random dropping = proportional to camera count
        expected_drop = drop_count / len(cameras)
        print(f"  {name:20} ({level:8}): ~{expected_drop:.0f} frames dropped")
    
    print(f"\n❌ RESULT: Critical cameras lose ~{drop_count/len(cameras):.0f} frames")
    print("   This means Main Gate threats could be MISSED!")
    
    print("\n" + "="*70)
    print("WITH Priority (Intelligent Dropping):")
    print("-"*70)
    print("When buffer is full, drop LOW priority first")
    print("\nExpected drops:")
    
    # Simulate intelligent dropping
    drops = {
        "Main Gate": 2,       # CRITICAL: Almost never dropped
        "Checkpoint": 2,      # CRITICAL: Almost never dropped
        "North Fence": 15,    # HIGH: Some drops under load
        "South Fence": 15,    # HIGH: Some drops under load
        "East Road": 35,      # MEDIUM: Moderate drops
        "West Road": 35,      # MEDIUM: Moderate drops
        "Parking": 55,        # LOW: Heavy drops
        "Interior": 45,       # LOW: Heavy drops
        "Equipment": 36       # MINIMAL: Very heavy drops
    }
    
    for name, (priority, level) in cameras.items():
        dropped = drops[name]
        print(f"  {name:20} ({level:8}): ~{dropped} frames dropped")
    
    print(f"\n✓ RESULT: Critical cameras lose only ~2 frames")
    print("   Main Gate threats are DETECTED with high reliability!")
    print("   Non-critical cameras gracefully degrade")
    
    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("="*70)
    print("Priority-based dropping ensures CRITICAL threats are never missed")
    print("even when the system is under heavy load.")
    print("\nThis is the difference between a reliable system and a useless one.")


def main():
    """Run all demonstrations."""
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  FRAME BUFFER & PRIORITY QUEUE DEMONSTRATION".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█" + "  Why Priority Matters in Real-World Surveillance".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # Scenario 1: Random processing
    simulate_random_processing()
    
    # Scenario 2: Priority processing
    simulate_priority_processing()
    
    # Scenario 3: Drop rate comparison
    compare_drop_rates()
    
    # Final summary
    print("\n\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  CONCLUSION".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    print("\nDay 5 (Camera Feed Manager):")
    print("  ✓ Can ingest 300 frames/second from 10 cameras")
    print("  ✓ Multi-threaded, thread-safe, automatic reconnection")
    print("  ❌ BUT: Processes frames in random order")
    print("  ❌ Critical threats can be missed due to queue position")
    
    print("\nDay 6 (Frame Buffer & Priority Queue):")
    print("  ✓ Same 300 frames/second ingestion")
    print("  ✓ BUT: Intelligent priority-based processing")
    print("  ✓ Critical cameras (Main Gate) processed FIRST")
    print("  ✓ Low-priority cameras (Parking) processed LAST")
    print("  ✓ Under load: Drop unimportant frames, keep critical ones")
    print("  ✓ RESULT: Threats detected faster, response time improved")
    
    print("\nReal-World Impact:")
    print("  • Border surveillance: Detect infiltration attempts immediately")
    print("  • Airport security: Process checkpoint cameras first")
    print("  • Critical infrastructure: Never miss main entrance threats")
    print("  • Emergency response: Fastest possible alert times")
    
    print("\n" + "█"*70)
    print("\nThis is not theoretical. This is how real defense systems work.")
    print("Priority saves lives.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
