"""
Object Detector - Security-Relevant Object Detection

Critical use cases:
1. Checkpoint screening (backpacks, bags)
2. Abandoned object detection (IED threat)
3. Weapon-like object identification
4. Person-object association tracking
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
    raise ImportError("ultralytics not installed. Install with: pip install ultralytics")

from .config import ObjectDetectionConfig
from .classifier import ObjectType, ObjectSize, RiskLevel, ObjectCharacteristics


@dataclass
class ObjectDetection:
    """Single object detection result."""
    
    bbox: Tuple[int, int, int, int]
    confidence: float
    object_type: ObjectType
    object_size: ObjectSize
    characteristics: ObjectCharacteristics
    class_id: int = 0
    area: int = 0
    center: Tuple[int, int] = (0, 0)
    
    def __post_init__(self):
        """Calculate derived fields."""
        x1, y1, x2, y2 = self.bbox
        self.area = (x2 - x1) * (y2 - y1)
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def summary(self) -> str:
        """One-line summary."""
        risk = self.characteristics.risk_level.name
        status = "ABANDONED" if self.characteristics.is_abandoned else "NORMAL"
        return (
            f"[{status}] {self.object_type.display_name} "
            f"({self.confidence:.0%}) - Risk: {risk}"
        )


class ObjectDetector:
    """
    Object detection system for security surveillance.
    
    Primary use cases:
    - Checkpoint screening
    - Abandoned object detection
    - Person-object association
    """
    
    def __init__(self, config: Optional[ObjectDetectionConfig] = None):
        """Initialize detector."""
        self.config = config or ObjectDetectionConfig()
        
        print(f"Loading YOLOv8{self.config.model_size} for object detection...")
        model_name = f"yolov8{self.config.model_size}.pt"
        
        try:
            self.model = YOLO(model_name)
            
            if self.config.device == 'cuda':
                import torch
                if not torch.cuda.is_available():
                    print("Warning: CUDA not available, falling back to CPU")
                    self.config.device = 'cpu'
                    self.config.half_precision = False
            
            print(f"Model loaded on {self.config.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
        
        self.detect_classes = self._build_class_filter()
        
        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0.0
        
        # Object tracking for abandoned detection
        self.tracked_objects = {}  # {object_id: {'last_seen': time, 'stationary_time': float}}
    
    def _build_class_filter(self) -> List[int]:
        """Build list of object classes to detect."""
        classes = []
        
        if self.config.detect_backpacks:
            classes.append(24)
        if self.config.detect_handbags:
            classes.append(26)
        if self.config.detect_suitcases:
            classes.append(28)
        if self.config.detect_sports_equipment:
            classes.extend([33, 35, 37, 39, 31, 32])
        if self.config.detect_small_items:
            classes.extend([25, 41, 73])
        
        return list(set(classes))
    
    def detect(
        self,
        source: Union[np.ndarray, str, Path],
        person_detections: Optional[List] = None,
        visualize: bool = False,
        time_of_day: str = "unknown"
    ) -> Tuple[List[ObjectDetection], Optional[np.ndarray]]:
        """
        Detect objects in image/frame.
        
        Args:
            source: Input image
            person_detections: List of person detections for association
            visualize: Whether to return annotated image
            time_of_day: Time context
            
        Returns:
            (detections, annotated_image)
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
        
        # Associate with persons if provided
        if person_detections and self.config.enable_person_association:
            detections = self._associate_with_persons(detections, person_detections)
        
        # Update temporal tracking
        detections = self._update_temporal_tracking(detections)
        
        # Visualization
        annotated = None
        if visualize:
            annotated = self._visualize_detections(frame, detections, inference_time)
        
        return detections, annotated
    
    def _extract_detections(
        self, result, frame: np.ndarray, time_of_day: str
    ) -> List[ObjectDetection]:
        """Extract object detections from YOLO results."""
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            cls_id = int(cls_id)
            
            if cls_id not in self.detect_classes:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            
            if area < self.config.min_object_area:
                continue
            
            try:
                object_type = ObjectType.from_class_id(cls_id)
            except ValueError:
                continue
            
            object_size = self._classify_size(area, object_type)
            
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0.0
            
            # Check if in restricted zone
            in_restricted = self._is_in_restricted_zone((x1, y1, x2, y2))
            
            characteristics = ObjectCharacteristics(
                object_type=object_type,
                size=object_size,
                confidence=float(conf),
                bbox_area=area,
                aspect_ratio=aspect_ratio,
                time_of_day=time_of_day,
                in_restricted_zone=in_restricted
            )
            
            detection = ObjectDetection(
                bbox=(x1, y1, x2, y2),
                confidence=float(conf),
                object_type=object_type,
                object_size=object_size,
                characteristics=characteristics,
                class_id=cls_id
            )
            
            detections.append(detection)
        
        return detections
    
    def _classify_size(self, area: int, object_type: ObjectType) -> ObjectSize:
        """Classify object size."""
        if area < self.config.size_threshold_small_medium:
            return ObjectSize.SMALL
        elif area < self.config.size_threshold_medium_large:
            return ObjectSize.MEDIUM
        else:
            return ObjectSize.LARGE
    
    def _associate_with_persons(
        self, objects: List[ObjectDetection], persons: List
    ) -> List[ObjectDetection]:
        """Associate objects with nearby persons."""
        for obj in objects:
            min_dist = float('inf')
            nearest_person = None
            
            for person in persons:
                dist = self._calculate_distance(obj.center, person.center)
                if dist < min_dist:
                    min_dist = dist
                    nearest_person = person
            
            if min_dist <= self.config.association_distance_threshold:
                obj.characteristics.near_person = True
                obj.characteristics.person_distance = int(min_dist)
            else:
                obj.characteristics.near_person = False
        
        return objects
    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance."""
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
    
    def _update_temporal_tracking(self, detections: List[ObjectDetection]) -> List[ObjectDetection]:
        """Update temporal tracking for abandoned object detection."""
        current_time = time.time()
        
        for det in detections:
            obj_key = f"{det.center[0]}_{det.center[1]}"  # Simplified tracking
            
            if obj_key in self.tracked_objects:
                # Update existing tracking
                track_info = self.tracked_objects[obj_key]
                time_delta = current_time - track_info['last_seen']
                track_info['stationary_time'] += time_delta
                track_info['last_seen'] = current_time
                
                det.characteristics.stationary_time = track_info['stationary_time']
                
                # Check abandoned threshold
                if (det.characteristics.stationary_time > self.config.abandoned_time_threshold
                    and not det.characteristics.near_person):
                    det.characteristics.is_abandoned = True
            else:
                # New object
                self.tracked_objects[obj_key] = {
                    'last_seen': current_time,
                    'stationary_time': 0.0
                }
        
        return detections
    
    def _is_in_restricted_zone(self, bbox: Tuple[int, int, int, int]) -> bool:
        """Check if object is in restricted zone."""
        if not self.config.restricted_zones:
            return False
        
        obj_center_x = (bbox[0] + bbox[2]) // 2
        obj_center_y = (bbox[1] + bbox[3]) // 2
        
        for zone in self.config.restricted_zones:
            zx1, zy1, zx2, zy2 = zone
            if zx1 <= obj_center_x <= zx2 and zy1 <= obj_center_y <= zy2:
                return True
        
        return False
    
    def _enhance_low_light(self, frame: np.ndarray) -> np.ndarray:
        """Enhance low-light frames."""
        if len(frame.shape) == 3:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(frame)
    
    def _visualize_detections(
        self, frame: np.ndarray, detections: List[ObjectDetection], inference_time: float
    ) -> np.ndarray:
        """Draw bounding boxes and information."""
        vis = frame.copy()
        
        color_map = {
            RiskLevel.NONE: (0, 255, 0),
            RiskLevel.LOW: (0, 255, 255),
            RiskLevel.MODERATE: (0, 165, 255),
            RiskLevel.HIGH: (0, 0, 255),
            RiskLevel.CRITICAL: (128, 0, 128)
        }
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            risk = det.characteristics.risk_level
            color = color_map[risk]
            
            thickness = 3 if det.characteristics.is_abandoned else 2
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(vis, det.center, 4, color, -1)
            
            label = f"{det.object_type.display_name} {det.confidence:.0%}"
            if det.characteristics.is_abandoned:
                label += f" [ABANDONED {int(det.characteristics.stationary_time)}s]"
            
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            cv2.rectangle(vis, (x1, y1 - label_h - 8), (x1 + label_w + 8, y1), color, -1)
            cv2.putText(vis, label, (x1 + 4, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw statistics
        fps = 1.0 / inference_time if inference_time > 0 else 0
        abandoned = sum(1 for d in detections if d.characteristics.is_abandoned)
        
        stats = [
            f"Objects: {len(detections)}",
            f"Abandoned: {abandoned}",
            f"FPS: {fps:.1f}"
        ]
        
        y_offset = 30
        for stat in stats:
            color = (0, 0, 255) if "Abandoned" in stat and abandoned > 0 else (0, 255, 255)
            cv2.putText(vis, stat, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        return vis
    
    def get_statistics(self) -> Dict:
        """Get detection statistics."""
        return {
            'total_frames': self.frame_count,
            'tracked_objects': len(self.tracked_objects),
            'avg_fps': (
                1.0 / (self.total_inference_time / self.frame_count)
                if self.frame_count > 0 else 0
            )
        }
