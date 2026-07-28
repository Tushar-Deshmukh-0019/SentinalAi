"""
Detection Pipeline Orchestrator

The "brain" that coordinates all detection components into a unified system.

Why This Matters:
=================

Individual detectors are like having sensors without a brain:
- Person detector: "I see a person"
- Vehicle detector: "I see a vehicle"
- Object detector: "I see a backpack"

But the critical questions remain unanswered:
- Is the person IN the vehicle?
- Does the backpack BELONG to the person?
- Is this person EXPECTED here?
- What's the THREAT LEVEL?

The orchestrator provides the intelligence layer that:
1. Runs all detectors efficiently
2. Correlates results (person + backpack = ownership)
3. Resolves conflicts (person vs. animal ambiguity)
4. Calculates threat scores
5. Generates alerts
6. Produces actionable intelligence

This is the difference between raw data and intelligence.

Real-World Impact:
==================

Border Post Scenario:
- Frame arrives: Person, vehicle, backpack detected
- WITHOUT orchestrator: 3 separate detections, no context
- WITH orchestrator: "Unknown individual with large backpack 
  exiting unauthorized vehicle at main gate. Not matching 
  patrol schedule. Threat score: 85/100. ALERT OPERATOR."

This is production-grade surveillance intelligence.
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np

from .camera_feed_manager import Frame
from .frame_buffer import FrameBuffer
from ..detection.person import PersonDetector, PersonDetection
from ..detection.vehicle import VehicleDetector, VehicleDetection
from ..detection.animal import AnimalDetector, AnimalDetection
from ..detection.object import ObjectDetector, ObjectDetection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat level classification."""
    NONE = "none"           # No threats detected
    LOW = "low"             # Minor anomaly, log only
    MODERATE = "moderate"   # Unusual activity, monitor
    HIGH = "high"           # Suspicious, alert operator
    CRITICAL = "critical"   # Immediate threat, emergency response


@dataclass
class DetectionResult:
    """
    Unified detection result with all information.
    
    This is what the orchestrator produces - complete intelligence.
    """
    
    # Source information
    camera_id: str
    frame_number: int
    timestamp: float
    
    # Detection results
    persons: List[PersonDetection] = field(default_factory=list)
    vehicles: List[VehicleDetection] = field(default_factory=list)
    animals: List[AnimalDetection] = field(default_factory=list)
    objects: List[ObjectDetection] = field(default_factory=list)
    
    # Intelligence layer
    threat_score: float = 0.0
    """Threat score (0-100). Higher = more suspicious."""
    
    threat_level: ThreatLevel = ThreatLevel.NONE
    """Classified threat level."""
    
    explanation: str = ""
    """Human-readable explanation of threat score."""
    
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    """List of alerts to send to operators."""
    
    # Correlation information
    person_vehicle_associations: List[Tuple[int, int]] = field(default_factory=list)
    """List of (person_idx, vehicle_idx) associations."""
    
    person_object_associations: List[Tuple[int, int]] = field(default_factory=list)
    """List of (person_idx, object_idx) associations."""
    
    # Performance metrics
    processing_time_ms: float = 0.0
    """Total processing time in milliseconds."""
    
    detector_times: Dict[str, float] = field(default_factory=dict)
    """Individual detector timings."""


class DetectionOrchestrator:
    """
    Coordinates all detection components into a unified intelligence system.
    
    Responsibilities:
    =================
    
    1. Detector Coordination
       - Run all 4 detectors on each frame
       - Parallel execution where possible
       - Performance optimization
    
    2. Conflict Resolution
       - Person vs. Animal ambiguity (Day 3)
       - Multiple overlapping detections
       - Confidence-based arbitration
    
    3. Result Correlation
       - Person-Vehicle associations (who's in which vehicle?)
       - Person-Object associations (who owns which object?)
       - Spatial proximity analysis
    
    4. Threat Assessment (Preliminary Layer 9)
       - Calculate threat scores
       - Classify threat levels
       - Generate explanations
       - Determine alert priority
    
    5. Alert Generation
       - Create operator alerts
       - Priority-based routing
       - Explanation attachment
    
    This is the "brain" that turns detections into intelligence.
    """
    
    def __init__(
        self,
        person_detector: Optional[PersonDetector] = None,
        vehicle_detector: Optional[VehicleDetector] = None,
        animal_detector: Optional[AnimalDetector] = None,
        object_detector: Optional[ObjectDetector] = None,
        enable_parallel: bool = True
    ):
        """
        Initialize detection orchestrator.
        
        Args:
            person_detector: Person detection module (Day 1)
            vehicle_detector: Vehicle detection module (Day 2)
            animal_detector: Animal detection module (Day 3)
            object_detector: Object detection module (Day 4)
            enable_parallel: Run detectors in parallel (faster but more GPU memory)
        """
        # Initialize detectors (create default if not provided)
        self.person_detector = person_detector or PersonDetector()
        self.vehicle_detector = vehicle_detector or VehicleDetector()
        self.animal_detector = animal_detector or AnimalDetector()
        self.object_detector = object_detector or ObjectDetector()
        
        self.enable_parallel = enable_parallel
        
        # Statistics
        self.frames_processed = 0
        self.total_processing_time = 0.0
        self.threat_counts = {level: 0 for level in ThreatLevel}
        
        logger.info("Detection Orchestrator initialized")
        logger.info(f"  Parallel processing: {enable_parallel}")
    
    def process_frame(self, frame: Frame) -> DetectionResult:
        """
        Process a single frame through all detection pipelines.
        
        This is the main entry point - one frame in, complete intelligence out.
        
        Args:
            frame: Frame to process
            
        Returns:
            Complete detection result with threat assessment
        """
        start_time = time.time()
        
        # Initialize result
        result = DetectionResult(
            camera_id=frame.camera_id,
            frame_number=frame.frame_number,
            timestamp=frame.timestamp
        )
        
        # Step 1: Run all detectors
        if self.enable_parallel:
            result = self._run_detectors_parallel(frame, result)
        else:
            result = self._run_detectors_sequential(frame, result)
        
        # Step 2: Resolve conflicts (person vs. animal)
        result = self._resolve_conflicts(result)
        
        # Step 3: Correlate results
        result = self._correlate_detections(result)
        
        # Step 4: Calculate threat score (preliminary Layer 9)
        result = self._calculate_threat_score(result)
        
        # Step 5: Generate alerts
        result = self._generate_alerts(result)
        
        # Update statistics
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        result.processing_time_ms = processing_time
        
        self.frames_processed += 1
        self.total_processing_time += processing_time
        self.threat_counts[result.threat_level] += 1
        
        return result
    
    def _run_detectors_sequential(self, frame: Frame, result: DetectionResult) -> DetectionResult:
        """
        Run all detectors sequentially (slower but less memory).
        
        Order matters:
        1. Animal first (for conflict resolution)
        2. Person second (checked against animals)
        3. Vehicle third (for person-vehicle correlation)
        4. Object last (for person-object correlation)
        """
        # Animal detection
        t0 = time.time()
        result.animals = self.animal_detector.detect(frame.image)
        result.detector_times['animal'] = (time.time() - t0) * 1000
        
        # Person detection
        t0 = time.time()
        result.persons = self.person_detector.detect(frame.image)
        result.detector_times['person'] = (time.time() - t0) * 1000
        
        # Vehicle detection
        t0 = time.time()
        result.vehicles = self.vehicle_detector.detect(frame.image)
        result.detector_times['vehicle'] = (time.time() - t0) * 1000
        
        # Object detection
        t0 = time.time()
        result.objects = self.object_detector.detect(frame.image)
        result.detector_times['object'] = (time.time() - t0) * 1000
        
        return result
    
    def _run_detectors_parallel(self, frame: Frame, result: DetectionResult) -> DetectionResult:
        """
        Run all detectors in parallel (faster with multiple GPUs or CPU cores).
        
        Note: This requires sufficient GPU memory to run 4 models simultaneously.
        In production, you'd use a GPU with 16GB+ VRAM or distribute across GPUs.
        """
        # For now, we'll use threading (good for CPU, limited for single GPU)
        # In production, consider:
        # - Multiple GPUs with model parallelism
        # - Batched inference
        # - Model optimization (quantization, TensorRT)
        
        results_dict = {}
        times_dict = {}
        threads = []
        
        def run_animal():
            t0 = time.time()
            results_dict['animals'] = self.animal_detector.detect(frame.image)
            times_dict['animal'] = (time.time() - t0) * 1000
        
        def run_person():
            t0 = time.time()
            results_dict['persons'] = self.person_detector.detect(frame.image)
            times_dict['person'] = (time.time() - t0) * 1000
        
        def run_vehicle():
            t0 = time.time()
            results_dict['vehicles'] = self.vehicle_detector.detect(frame.image)
            times_dict['vehicle'] = (time.time() - t0) * 1000
        
        def run_object():
            t0 = time.time()
            results_dict['objects'] = self.object_detector.detect(frame.image)
            times_dict['object'] = (time.time() - t0) * 1000
        
        # Start all detectors
        threads = [
            threading.Thread(target=run_animal),
            threading.Thread(target=run_person),
            threading.Thread(target=run_vehicle),
            threading.Thread(target=run_object)
        ]
        
        for thread in threads:
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        result.animals = results_dict.get('animals', [])
        result.persons = results_dict.get('persons', [])
        result.vehicles = results_dict.get('vehicles', [])
        result.objects = results_dict.get('objects', [])
        result.detector_times = times_dict
        
        return result
    
    def _resolve_conflicts(self, result: DetectionResult) -> DetectionResult:
        """
        Resolve conflicts between detectors.
        
        Main conflict: Person vs. Animal (Day 3 issue)
        
        Scenario:
        - Person detector: "Maybe person" (confidence: 0.52)
        - Animal detector: "Definitely deer" (confidence: 0.94)
        
        Resolution:
        - If confidence difference > 0.20, trust higher confidence
        - Remove the lower confidence detection
        - Prevents deer from triggering person alerts ✓
        """
        if not result.persons or not result.animals:
            return result  # No conflict possible
        
        persons_to_keep = []
        
        for person in result.persons:
            # Check if this "person" overlaps with any animal
            is_actually_animal = False
            
            for animal in result.animals:
                # Calculate IoU (Intersection over Union)
                iou = self._calculate_iou(person.bbox, animal.bbox)
                
                if iou > 0.3:  # Significant overlap
                    # Confidence difference
                    conf_diff = animal.confidence - person.confidence
                    
                    if conf_diff > 0.20:
                        # Animal detector is much more confident
                        # This is probably an animal, not a person
                        is_actually_animal = True
                        logger.debug(
                            f"Conflict resolved: Person (conf={person.confidence:.2f}) "
                            f"vs {animal.species} (conf={animal.confidence:.2f}) "
                            f"→ Classified as {animal.species}"
                        )
                        break
            
            if not is_actually_animal:
                persons_to_keep.append(person)
        
        # Update result
        removed_count = len(result.persons) - len(persons_to_keep)
        if removed_count > 0:
            logger.info(f"Conflict resolution: Removed {removed_count} false person detections")
        
        result.persons = persons_to_keep
        
        return result
    
    def _correlate_detections(self, result: DetectionResult) -> DetectionResult:
        """
        Correlate detections to understand relationships.
        
        Correlations:
        =============
        
        1. Person-Vehicle Associations
           - Is person inside vehicle?
           - Is person next to vehicle (driver/passenger)?
           - Spatial proximity analysis
        
        2. Person-Object Associations
           - Does person "own" this object?
           - Is object near person?
           - Abandoned object detection (Day 4)
        
        Why This Matters:
        =================
        
        Without correlation:
          "Person detected. Vehicle detected. Backpack detected."
          No context, no intelligence ❌
        
        With correlation:
          "Person carrying large backpack, exiting vehicle at main gate."
          Complete context, actionable intelligence ✓
        """
        # Person-Vehicle associations
        for p_idx, person in enumerate(result.persons):
            for v_idx, vehicle in enumerate(result.vehicles):
                # Check if person is near/in vehicle
                if self._is_person_in_vehicle(person, vehicle):
                    result.person_vehicle_associations.append((p_idx, v_idx))
        
        # Person-Object associations
        for p_idx, person in enumerate(result.persons):
            for o_idx, obj in enumerate(result.objects):
                # Check if object belongs to person
                if self._is_object_owned_by_person(person, obj):
                    result.person_object_associations.append((p_idx, o_idx))
        
        return result
    
    def _calculate_threat_score(self, result: DetectionResult) -> DetectionResult:
        """
        Calculate preliminary threat score (Layer 9 preview).
        
        This is a simplified version. Full Layer 9 (Days 69-75) will include:
        - Patrol schedule matching (Layer 1)
        - Directional analysis (Layer 2)
        - Behavior analysis (Layer 3)
        - Team correlation (Layer 4)
        - GPS tracking (Layer 5)
        - Vehicle authorization (Layer 6)
        - Identity verification (Layer 7)
        - Weapon detection (Layer 8)
        
        For now, we use basic heuristics:
        - Presence of persons: +20 per person
        - Unknown vehicles: +15 per vehicle
        - Large objects (backpacks): +10 per object
        - Person with object: +15 (potential threat)
        - Multiple persons: +10 (coordination indicator)
        
        Score Ranges:
        - 0-20: NONE (normal activity)
        - 21-40: LOW (minor anomaly)
        - 41-60: MODERATE (unusual activity)
        - 61-80: HIGH (suspicious)
        - 81-100: CRITICAL (immediate threat)
        """
        score = 0.0
        explanations = []
        
        # Base scoring
        if result.persons:
            score += 20 * len(result.persons)
            explanations.append(f"{len(result.persons)} person(s) detected")
        
        if result.vehicles:
            score += 15 * len(result.vehicles)
            explanations.append(f"{len(result.vehicles)} vehicle(s) detected")
        
        if result.objects:
            score += 10 * len(result.objects)
            explanations.append(f"{len(result.objects)} object(s) detected")
        
        # Correlation scoring
        if result.person_object_associations:
            score += 15 * len(result.person_object_associations)
            explanations.append(f"Person carrying {len(result.person_object_associations)} object(s)")
        
        # Multiple persons (potential coordination)
        if len(result.persons) >= 2:
            score += 10
            explanations.append("Multiple individuals (coordination indicator)")
        
        # No animals (if animals present, less threatening)
        if result.animals:
            score *= 0.5  # Reduce score if wildlife present
            explanations.append(f"Wildlife detected ({len(result.animals)} animal(s))")
        
        # Cap score at 100
        score = min(score, 100.0)
        
        # Classify threat level
        if score <= 20:
            threat_level = ThreatLevel.NONE
        elif score <= 40:
            threat_level = ThreatLevel.LOW
        elif score <= 60:
            threat_level = ThreatLevel.MODERATE
        elif score <= 80:
            threat_level = ThreatLevel.HIGH
        else:
            threat_level = ThreatLevel.CRITICAL
        
        # Update result
        result.threat_score = score
        result.threat_level = threat_level
        result.explanation = "; ".join(explanations) if explanations else "No activity"
        
        return result
    
    def _generate_alerts(self, result: DetectionResult) -> DetectionResult:
        """
        Generate operator alerts based on threat level.
        
        Alert Priority:
        - CRITICAL: Immediate notification, audio alarm
        - HIGH: Priority notification
        - MODERATE: Standard notification
        - LOW: Log only, no notification
        - NONE: Log only
        """
        if result.threat_level == ThreatLevel.CRITICAL:
            result.alerts.append({
                'type': 'critical_threat',
                'priority': 10,
                'message': f"CRITICAL THREAT at {result.camera_id}",
                'details': result.explanation,
                'action': 'Deploy security team immediately'
            })
        
        elif result.threat_level == ThreatLevel.HIGH:
            result.alerts.append({
                'type': 'high_threat',
                'priority': 8,
                'message': f"Suspicious activity at {result.camera_id}",
                'details': result.explanation,
                'action': 'Review footage and assess'
            })
        
        elif result.threat_level == ThreatLevel.MODERATE:
            result.alerts.append({
                'type': 'moderate_activity',
                'priority': 5,
                'message': f"Unusual activity at {result.camera_id}",
                'details': result.explanation,
                'action': 'Monitor situation'
            })
        
        # LOW and NONE: No alerts, logging only
        
        return result
    
    # Helper methods
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                        bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0  # No intersection
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _is_person_in_vehicle(self, person: PersonDetection, 
                              vehicle: VehicleDetection) -> bool:
        """Check if person is inside or near vehicle."""
        # Simple proximity check (can be enhanced)
        iou = self._calculate_iou(person.bbox, vehicle.bbox)
        return iou > 0.1  # Person overlaps with vehicle
    
    def _is_object_owned_by_person(self, person: PersonDetection, 
                                    obj: ObjectDetection) -> bool:
        """Check if object belongs to person (proximity-based)."""
        # Calculate distance between centers
        p_center_x = (person.bbox[0] + person.bbox[2]) / 2
        p_center_y = (person.bbox[1] + person.bbox[3]) / 2
        
        o_center_x = (obj.bbox[0] + obj.bbox[2]) / 2
        o_center_y = (obj.bbox[1] + obj.bbox[3]) / 2
        
        distance = np.sqrt((p_center_x - o_center_x)**2 + 
                          (p_center_y - o_center_y)**2)
        
        # If object is within 100 pixels of person, consider it "owned"
        # In production, this would be more sophisticated (body pose, etc.)
        return distance < 100
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        avg_time = (
            self.total_processing_time / self.frames_processed
            if self.frames_processed > 0 else 0
        )
        
        return {
            'frames_processed': self.frames_processed,
            'avg_processing_time_ms': avg_time,
            'fps': 1000.0 / avg_time if avg_time > 0 else 0,
            'threat_counts': {
                level.value: count
                for level, count in self.threat_counts.items()
            }
        }
    
    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("DETECTION ORCHESTRATOR STATISTICS")
        print("="*70)
        print(f"Frames Processed: {stats['frames_processed']}")
        print(f"Avg Processing Time: {stats['avg_processing_time_ms']:.1f}ms")
        print(f"Processing FPS: {stats['fps']:.1f}")
        
        print("\nThreat Level Distribution:")
        print("-"*70)
        for level, count in stats['threat_counts'].items():
            pct = count / stats['frames_processed'] * 100 if stats['frames_processed'] > 0 else 0
            print(f"  {level.upper():10}: {count:6} ({pct:5.1f}%)")


# Example usage
if __name__ == "__main__":
    print("Detection Orchestrator initialized.")
    print("Run complete demo with: python ai/pipelines/demo_orchestrator.py")
