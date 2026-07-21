"""
Animal Detector - False Positive Filter

Real-world scenario this solves:
------------------------------------
02:34 AM - Motion detected in Sector 3

WITHOUT animal detection:
- Person detector: "Maybe person" (0.52 confidence)
- System: ALERT OPERATOR
- Operator wakes up, checks camera
- It's a deer
- False alarm #7 tonight
- Operator frustration increases

WITH animal detection:
- Person detector: "Maybe person" (0.52 confidence - LOW)
- Animal detector: "Deer" (0.94 confidence - HIGH)
- System: Wildlife detected - no alert
- Operator sleeps
- Trust in system maintained ✓

This module is the difference between a useful surveillance system
and an alarm system that gets ignored.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional
from dataclasses import dataclass
import time

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "ultralytics not installed. Install with: pip install ultralytics"
    )

from .config import AnimalDetectionConfig
from .classifier import AnimalType, AnimalSize, ThreatLevel, AnimalCharacteristics


@dataclass
class AnimalDetection:
    """
    Single animal detection result.
    
    Contains all information needed for filtering and logging.
    """
    
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    """Bounding box coordinates."""
    
    confidence: float  # 0.0 to 1.0
    """Detection confidence."""
    
    animal_type: AnimalType
    """Classified animal type."""
    
    animal_size: AnimalSize
    """Size classification."""
    
    characteristics: AnimalCharacteristics
    """Complete animal characteristics."""
    
    class_id: int = 0
    """COCO class ID."""
    
    area: int = 0
    """Bounding box area in pixels."""
    
    center: Tuple[int, int] = (0, 0)
    """Center point of bounding box."""
    
    def __post_init__(self):
        """Calculate derived fields."""
        x1, y1, x2, y2 = self.bbox
        self.area = (x2 - x1) * (y2 - y1)
        self.center = (
            (x1 + x2) // 2,
            (y1 + y2) // 2
        )
    
    @property
    def width(self) -> int:
        """Animal width in pixels."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        """Animal height in pixels."""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def should_filter(self) -> bool:
        """Whether this detection should be filtered out."""
        return self.characteristics.should_filter
    
    @property
    def summary(self) -> str:
        """One-line summary."""
        filter_status = "FILTERED" if self.should_filter else "ALERT"
        threat = self.characteristics.threat_level.name
        return (
            f"[{filter_status}] {self.animal_type.display_name} "
            f"({self.confidence:.0%}) - Threat: {threat}"
        )


class AnimalDetector:
    """
    Core animal detection system.
    
    Primary purpose: Reduce false positive alerts from wildlife.
    
    Usage:
        detector = AnimalDetector()
        detections = detector.detect(frame)
        
        filtered = [d for d in detections if d.should_filter]
        alerts = [d for d in detections if not d.should_filter]
        
        print(f"Filtered {len(filtered)} animals, alerting on {len(alerts)}")
    """
    
    def __init__(self, config: Optional[AnimalDetectionConfig] = None):
        """
        Initialize detector with configuration.
        
        Args:
            config: Detection configuration. If None, uses defaults.
        """
        self.config = config or AnimalDetectionConfig()
        
        # Load YOLOv8 model
        print(f"Loading YOLOv8{self.config.model_size} model for animal detection...")
        model_name = f"yolov8{self.config.model_size}.pt"
        
        try:
            self.model = YOLO(model_name)
            
            # Configure device
            if self.config.device == 'cuda':
                import torch
                if not torch.cuda.is_available():
                    print("Warning: CUDA not available, falling back to CPU")
                    self.config.device = 'cpu'
                    self.config.half_precision = False
            
            print(f"Model loaded successfully on {self.config.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
        
        # Build class filter
        self.detect_classes = self._build_class_filter()
        
        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.total_detections = 0
        self.total_filtered = 0
        
        # Statistics
        self.animal_counts = {}  # Track detection counts by type
    
    def _build_class_filter(self) -> List[int]:
        """Build list of COCO class IDs to detect based on config."""
        classes = []
        
        # Birds
        if self.config.detect_birds:
            classes.append(16)  # bird
        
        # Small animals
        if self.config.detect_small_animals:
            classes.extend([17, 18])  # cat, dog
        
        # Livestock
        if self.config.detect_livestock:
            classes.extend([19, 20, 21])  # horse, sheep, cow
        
        # Wildlife (always detect for filtering)
        classes.extend([22, 23, 24, 25])  # elephant, bear, zebra, giraffe
        
        # Remove duplicates
        return list(set(classes))
    
    def detect(
        self,
        source: Union[np.ndarray, str, Path],
        visualize: bool = False,
        time_of_day: str = "unknown"
    ) -> Tuple[List[AnimalDetection], Optional[np.ndarray]]:
        """
        Detect animals in image/frame.
        
        Args:
            source: Input image as numpy array, or path to image
            visualize: If True, return annotated image
            time_of_day: Time context ("day", "night", "twilight")
            
        Returns:
            (detections, annotated_image)
            - detections: List of AnimalDetection objects
            - annotated_image: Image with bounding boxes (if visualize=True)
        """
        self.frame_count += 1
        
        # Frame skipping
        if self.config.skip_frames > 0:
            if self.frame_count % (self.config.skip_frames + 1) != 0:
                return [], None
        
        # Preprocess
        if isinstance(source, np.ndarray):
            frame = source.copy()
            if self.config.low_light_boost:
                frame = self._enhance_low_light(frame)
        else:
            frame = cv2.imread(str(source))
            if frame is None:
                raise ValueError(f"Could not read image from {source}")
        
        # Run inference
        start_time = time.time()
        
        results = self.model.predict(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.input_size,
            device=self.config.device,
            half=self.config.half_precision,
            max_det=self.config.max_detections,
            classes=self.detect_classes,
            verbose=False
        )
        
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        
        # Extract detections
        detections = self._extract_detections(results[0], frame, time_of_day)
        
        # Update statistics
        self.total_detections += len(detections)
        self.total_filtered += sum(1 for d in detections if d.should_filter)
        
        for det in detections:
            animal_name = det.animal_type.name
            self.animal_counts[animal_name] = self.animal_counts.get(animal_name, 0) + 1
        
        # Visualization
        annotated = None
        if visualize:
            annotated = self._visualize_detections(frame, detections, inference_time)
        
        return detections, annotated
    
    def _extract_detections(
        self,
        result,
        frame: np.ndarray,
        time_of_day: str
    ) -> List[AnimalDetection]:
        """Extract animal detection objects from YOLO results."""
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            cls_id = int(cls_id)
            
            # Skip if not an animal class we're detecting
            if cls_id not in self.detect_classes:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            
            # Filter by minimum area
            if area < self.config.min_animal_area:
                continue
            
            # Classify animal
            try:
                animal_type = AnimalType.from_class_id(cls_id)
            except ValueError:
                continue
            
            # Determine size
            animal_size = self._classify_size(area, animal_type)
            
            # Calculate characteristics
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0.0
            
            # Check if expected in this zone
            in_expected_zone = False
            if self.config.expected_animals:
                in_expected_zone = animal_type in self.config.expected_animals
            
            characteristics = AnimalCharacteristics(
                animal_type=animal_type,
                size=animal_size,
                confidence=float(conf),
                bbox_area=area,
                aspect_ratio=aspect_ratio,
                time_of_day=time_of_day,
                in_expected_zone=in_expected_zone
            )
            
            detection = AnimalDetection(
                bbox=(x1, y1, x2, y2),
                confidence=float(conf),
                animal_type=animal_type,
                animal_size=animal_size,
                characteristics=characteristics,
                class_id=cls_id
            )
            
            detections.append(detection)
        
        return detections
    
    def _classify_size(
        self,
        area: int,
        animal_type: AnimalType
    ) -> AnimalSize:
        """Classify animal size based on area and type."""
        # Type-based classification
        if animal_type in [AnimalType.BIRD, AnimalType.CAT]:
            return AnimalSize.SMALL
        
        if animal_type in [AnimalType.BEAR, AnimalType.COW, AnimalType.HORSE]:
            return AnimalSize.LARGE
        
        # Area-based for others
        if area < self.config.size_threshold_small_medium:
            return AnimalSize.SMALL
        elif area < self.config.size_threshold_medium_large:
            return AnimalSize.MEDIUM
        else:
            return AnimalSize.LARGE
    
    def resolve_conflict_with_person(
        self,
        animal_detections: List[AnimalDetection],
        person_detections: List
    ) -> Tuple[List, List[AnimalDetection]]:
        """
        Resolve conflicts when both person and animal detected.
        
        This is the CRITICAL function for false positive reduction.
        
        Args:
            animal_detections: List of animal detections
            person_detections: List of person detections (from person detector)
            
        Returns:
            (filtered_persons, kept_animals)
            - filtered_persons: Person detections that were NOT animals
            - kept_animals: Animal detections that caused filtering
            
        Logic:
        1. For each person detection:
           - Check if overlaps with animal detection
           - Compare confidences
           - Decide: person or animal?
        2. If animal wins → filter out person detection
        3. If person wins → keep person, log animal
        """
        if not self.config.enable_conflict_resolution:
            return person_detections, animal_detections
        
        filtered_persons = []
        filtered_animals = []
        conflicts_resolved = []
        
        for person_det in person_detections:
            px1, py1, px2, py2 = person_det.bbox
            person_center = person_det.center
            person_conf = person_det.confidence
            
            # Check for overlapping animals
            overlapping_animals = []
            for animal_det in animal_detections:
                ax1, ay1, ax2, ay2 = animal_det.bbox
                
                # Check overlap using IoU
                inter_x1 = max(px1, ax1)
                inter_y1 = max(py1, ay1)
                inter_x2 = min(px2, ax2)
                inter_y2 = min(py2, ay2)
                
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    person_area = person_det.area
                    animal_area = animal_det.area
                    union_area = person_area + animal_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    
                    if iou > 0.3:  # Significant overlap
                        overlapping_animals.append((animal_det, iou))
            
            # Resolve conflict
            if overlapping_animals:
                # Get best matching animal
                best_animal, best_iou = max(overlapping_animals, key=lambda x: x[1])
                
                # Use conflict resolution
                decision, final_conf = best_animal.characteristics.get_conflict_resolution(
                    person_conf
                )
                
                if decision == "animal":
                    # Filter out person, keep animal
                    filtered_animals.append(best_animal)
                    conflicts_resolved.append({
                        'person_conf': person_conf,
                        'animal_conf': best_animal.confidence,
                        'animal_type': best_animal.animal_type.display_name,
                        'decision': 'animal',
                        'iou': best_iou
                    })
                else:
                    # Keep person, log animal
                    filtered_persons.append(person_det)
                    conflicts_resolved.append({
                        'person_conf': person_conf,
                        'animal_conf': best_animal.confidence,
                        'animal_type': best_animal.animal_type.display_name,
                        'decision': 'person',
                        'iou': best_iou
                    })
            else:
                # No conflict, keep person
                filtered_persons.append(person_det)
        
        # Log conflicts if any resolved
        if conflicts_resolved and self.config.log_filtered_detections:
            print(f"\n[CONFLICT RESOLUTION] Resolved {len(conflicts_resolved)} conflicts:")
            for conflict in conflicts_resolved:
                print(f"  Person:{conflict['person_conf']:.2f} vs "
                      f"{conflict['animal_type']}:{conflict['animal_conf']:.2f} "
                      f"(IoU:{conflict['iou']:.2f}) → {conflict['decision'].upper()}")
        
        return filtered_persons, filtered_animals
    
    def _enhance_low_light(self, frame: np.ndarray) -> np.ndarray:
        """Enhance low-light frames."""
        if len(frame.shape) == 3:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            return enhanced
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(frame)
    
    def _visualize_detections(
        self,
        frame: np.ndarray,
        detections: List[AnimalDetection],
        inference_time: float
    ) -> np.ndarray:
        """Draw bounding boxes and information on frame."""
        vis = frame.copy()
        
        # Color map
        color_map = {
            ThreatLevel.NONE: (0, 255, 0),      # Green (filtered)
            ThreatLevel.LOW: (0, 255, 255),     # Yellow
            ThreatLevel.MODERATE: (0, 165, 255),  # Orange
            ThreatLevel.HIGH: (0, 0, 255)       # Red
        }
        
        filtered_count = 0
        alert_count = 0
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            threat = det.characteristics.threat_level
            color = color_map[threat]
            
            # Thicker border for alerts, thinner for filtered
            thickness = 1 if det.should_filter else 2
            
            # Draw bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            # Draw center
            cv2.circle(vis, det.center, 3, color, -1)
            
            # Label
            label = f"{det.animal_type.display_name} {det.confidence:.0%}"
            if det.should_filter:
                label += " [F]"  # Filtered
                filtered_count += 1
            else:
                label += " [A]"  # Alert
                alert_count += 1
            
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Label background
            cv2.rectangle(
                vis,
                (x1, y1 - label_h - 8),
                (x1 + label_w + 8, y1),
                color,
                -1
            )
            
            cv2.putText(
                vis, label, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        # Draw statistics
        fps = 1.0 / inference_time if inference_time > 0 else 0
        stats = [
            f"Animals: {len(detections)}",
            f"Filtered: {filtered_count}",
            f"Alerts: {alert_count}",
            f"FPS: {fps:.1f}"
        ]
        
        y_offset = 30
        for stat in stats:
            color = (0, 255, 0) if "Filtered" in stat else (0, 255, 255)
            cv2.putText(
                vis, stat, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
            y_offset += 25
        
        return vis
    
    def get_statistics(self) -> Dict:
        """Get detection statistics."""
        return {
            'total_frames': self.frame_count,
            'total_detections': self.total_detections,
            'total_filtered': self.total_filtered,
            'filter_rate': (
                self.total_filtered / self.total_detections * 100
                if self.total_detections > 0 else 0
            ),
            'animal_counts': self.animal_counts,
            'avg_fps': (
                1.0 / (self.total_inference_time / self.frame_count)
                if self.frame_count > 0 else 0
            )
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.total_detections = 0
        self.total_filtered = 0
        self.animal_counts = {}
