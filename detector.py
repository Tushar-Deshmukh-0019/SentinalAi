"""
Vehicle Detector - Core Detection Engine

Real-world scenario this handles:
-----------------------------------
Camera shows movement at 3:14 AM on access road.

Questions:
- Is there a vehicle?
- What type? (Car, truck, motorcycle, bus)
- What size? (Small, medium, large)
- How many vehicles?
- Are they moving or stationary?

This module provides the answers that feed into:
- Vehicle-person correlation (Day 56)
- Authorization checks (Day 53-55)
- Threat scoring (Day 69+)
- Tracking (Day 13+)
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

from .config import VehicleDetectionConfig
from .classifier import VehicleType, VehicleSize, VehicleCharacteristics


@dataclass
class VehicleDetection:
    """
    Single vehicle detection result.
    
    Contains all information needed for downstream analysis.
    """
    
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    """Bounding box coordinates."""
    
    confidence: float  # 0.0 to 1.0
    """Detection confidence."""
    
    vehicle_type: VehicleType
    """Classified vehicle type."""
    
    vehicle_size: VehicleSize
    """Size classification (small/medium/large)."""
    
    characteristics: VehicleCharacteristics
    """Complete vehicle characteristics for tactical analysis."""
    
    class_id: int = 0
    """COCO class ID."""
    
    area: int = 0
    """Bounding box area in pixels."""
    
    center: Tuple[int, int] = (0, 0)
    """Center point of bounding box."""
    
    has_license_plate_region: bool = False
    """Whether a license plate region was detected."""
    
    license_plate_bbox: Optional[Tuple[int, int, int, int]] = None
    """License plate region bounding box (if detected)."""
    
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
        """Vehicle width in pixels."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        """Vehicle height in pixels."""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def aspect_ratio(self) -> float:
        """Width/height ratio."""
        return self.width / self.height if self.height > 0 else 0.0
    
    @property
    def tactical_summary(self) -> str:
        """
        One-line tactical summary.
        
        Example: "Medium Car (87% conf) - Threat Level: 35/100"
        """
        threat = self.characteristics.base_threat_level
        return (
            f"{self.characteristics.description} "
            f"({self.confidence:.0%} conf) - "
            f"Threat Level: {threat}/100"
        )


class VehicleDetector:
    """
    Core vehicle detection system.
    
    Usage:
        detector = VehicleDetector()
        detections = detector.detect(frame)
        
        for vehicle in detections:
            print(f"Detected: {vehicle.tactical_summary}")
    """
    
    def __init__(self, config: Optional[VehicleDetectionConfig] = None):
        """
        Initialize detector with configuration.
        
        Args:
            config: Detection configuration. If None, uses defaults.
        """
        self.config = config or VehicleDetectionConfig()
        
        # Load YOLOv8 model
        print(f"Loading YOLOv8{self.config.model_size} model for vehicle detection...")
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
        
        # Build class filter based on config
        self.detect_classes = self._build_class_filter()
        
        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.skipped_frames = 0
    
    def _build_class_filter(self) -> List[int]:
        """
        Build list of COCO class IDs to detect based on config.
        
        Returns:
            List of class IDs to detect
        """
        classes = []
        
        # Always detect cars (class 2)
        classes.append(2)
        
        # Motorcycles (class 3)
        if self.config.detect_motorcycles:
            classes.append(3)
        
        # Buses and trucks (classes 5, 7)
        if self.config.detect_large_vehicles:
            classes.extend([5, 7])
        
        return classes
    
    def detect(
        self, 
        source: Union[np.ndarray, str, Path],
        visualize: bool = False
    ) -> Tuple[List[VehicleDetection], Optional[np.ndarray]]:
        """
        Detect vehicles in image/frame.
        
        Args:
            source: Input image as numpy array, or path to image/video
            visualize: If True, return annotated image
            
        Returns:
            (detections, annotated_image)
            - detections: List of VehicleDetection objects
            - annotated_image: Image with bounding boxes (if visualize=True)
            
        Example:
            detections, viz = detector.detect(frame, visualize=True)
            
            if len(detections) > 0:
                for vehicle in detections:
                    print(f"ALERT: {vehicle.tactical_summary}")
                    print(f"  Location: {vehicle.center}")
                    print(f"  Size: {vehicle.width}x{vehicle.height}")
        """
        self.frame_count += 1
        
        # Frame skipping for performance
        if self.config.skip_frames > 0:
            if self.frame_count % (self.config.skip_frames + 1) != 0:
                self.skipped_frames += 1
                return [], None
        
        # Preprocess if needed
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
            classes=self.detect_classes,  # Only detect configured vehicle types
            verbose=False
        )
        
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        
        # Extract detections
        detections = self._extract_detections(results[0], frame)
        
        # Detect license plates (if enabled)
        if self.config.license_plate_detection:
            detections = self._detect_license_plates(frame, detections)
        
        # Visualization
        annotated = None
        if visualize:
            annotated = self._visualize_detections(frame, detections, inference_time)
        
        return detections, annotated
    
    def _extract_detections(
        self, 
        result, 
        frame: np.ndarray
    ) -> List[VehicleDetection]:
        """
        Extract vehicle detection objects from YOLO results.
        
        Filters:
        - Minimum area threshold
        - Maximum area threshold (if set)
        - Vehicle type filter (based on config)
        
        Classifies:
        - Vehicle type (car, truck, motorcycle, bus)
        - Vehicle size (small, medium, large)
        """
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            cls_id = int(cls_id)
            
            # Skip if not a vehicle class we're detecting
            if cls_id not in self.detect_classes:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            
            # Filter by area
            if area < self.config.min_vehicle_area:
                continue
            
            if self.config.max_vehicle_area is not None:
                if area > self.config.max_vehicle_area:
                    continue
            
            # Classify vehicle
            try:
                vehicle_type = VehicleType.from_class_id(cls_id)
            except ValueError:
                continue
            
            # Determine size
            vehicle_size = self._classify_size(
                area, 
                vehicle_type,
                (x2 - x1, y2 - y1)
            )
            
            # Calculate characteristics
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0.0
            
            # Check if oversized for type
            is_oversized = self._is_oversized(area, vehicle_type, vehicle_size)
            
            characteristics = VehicleCharacteristics(
                vehicle_type=vehicle_type,
                size=vehicle_size,
                confidence=float(conf),
                bbox_area=area,
                aspect_ratio=aspect_ratio,
                is_oversized=is_oversized
            )
            
            detection = VehicleDetection(
                bbox=(x1, y1, x2, y2),
                confidence=float(conf),
                vehicle_type=vehicle_type,
                vehicle_size=vehicle_size,
                characteristics=characteristics,
                class_id=cls_id
            )
            
            detections.append(detection)
        
        return detections
    
    def _classify_size(
        self, 
        area: int, 
        vehicle_type: VehicleType,
        dimensions: Tuple[int, int]
    ) -> VehicleSize:
        """
        Classify vehicle size based on bounding box area and type.
        
        Args:
            area: Bounding box area in pixels²
            vehicle_type: Type of vehicle
            dimensions: (width, height) in pixels
            
        Returns:
            VehicleSize classification
            
        Logic:
        - Motorcycles always SMALL (even if detection box large)
        - Buses always LARGE
        - Cars/Trucks classified by area thresholds
        """
        # Type-based classification
        if vehicle_type == VehicleType.MOTORCYCLE:
            return VehicleSize.SMALL
        
        if vehicle_type == VehicleType.BUS:
            return VehicleSize.LARGE
        
        # Area-based classification for cars and trucks
        if area < self.config.size_threshold_small_medium:
            return VehicleSize.SMALL
        elif area < self.config.size_threshold_medium_large:
            return VehicleSize.MEDIUM
        else:
            return VehicleSize.LARGE
    
    def _is_oversized(
        self, 
        area: int, 
        vehicle_type: VehicleType,
        vehicle_size: VehicleSize
    ) -> bool:
        """
        Check if vehicle is unusually large for its type.
        
        Oversized vehicles are tactically significant:
        - Modified vehicle (cargo added)
        - Detection artifact
        - Special purpose vehicle
        
        Returns True if vehicle is >150% expected size for type.
        """
        expected_max = {
            (VehicleType.CAR, VehicleSize.SMALL): 20000,
            (VehicleType.CAR, VehicleSize.MEDIUM): 45000,
            (VehicleType.CAR, VehicleSize.LARGE): 60000,
            (VehicleType.TRUCK, VehicleSize.MEDIUM): 50000,
            (VehicleType.TRUCK, VehicleSize.LARGE): 80000,
            (VehicleType.MOTORCYCLE, VehicleSize.SMALL): 10000,
            (VehicleType.BUS, VehicleSize.LARGE): 100000,
        }
        
        key = (vehicle_type, vehicle_size)
        if key in expected_max:
            return area > expected_max[key] * 1.5
        
        return False
    
    def _detect_license_plates(
        self, 
        frame: np.ndarray, 
        detections: List[VehicleDetection]
    ) -> List[VehicleDetection]:
        """
        Detect license plate regions within vehicle bounding boxes.
        
        Note: This is simplified region detection, not OCR.
        Full OCR will be added in Day 40+.
        
        Strategy:
        - Search bottom 1/3 of vehicle bbox (plates usually at bottom)
        - Look for rectangular regions with high edge density
        - Typical aspect ratio: 2:1 to 5:1 (wider than tall)
        
        This is a placeholder implementation.
        Production would use dedicated license plate detection model.
        """
        # For now, simple heuristic-based detection
        # In production, use specialized model (e.g., YOLOv8 trained on plates)
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Search region: bottom 40% of vehicle
            search_y1 = y1 + int((y2 - y1) * 0.6)
            search_region = frame[search_y1:y2, x1:x2]
            
            if search_region.size == 0:
                continue
            
            # Simple edge detection
            gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            
            # Look for rectangular contours
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 200:  # Too small
                    continue
                
                # Get bounding rectangle
                px, py, pw, ph = cv2.boundingRect(contour)
                aspect_ratio = pw / ph if ph > 0 else 0
                
                # Check if aspect ratio matches license plate (2:1 to 5:1)
                if 2.0 <= aspect_ratio <= 5.0:
                    # Convert to full frame coordinates
                    plate_x1 = x1 + px
                    plate_y1 = search_y1 + py
                    plate_x2 = plate_x1 + pw
                    plate_y2 = plate_y1 + ph
                    
                    detection.has_license_plate_region = True
                    detection.license_plate_bbox = (
                        plate_x1, plate_y1, plate_x2, plate_y2
                    )
                    break  # Found one, that's enough
        
        return detections
    
    def _enhance_low_light(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance low-light frames.
        
        Same technique as person detection.
        """
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
        detections: List[VehicleDetection],
        inference_time: float
    ) -> np.ndarray:
        """
        Draw bounding boxes and information on frame.
        
        Color coding:
        - Blue: Cars
        - Red: Trucks
        - Yellow: Motorcycles
        - Green: Buses
        """
        vis = frame.copy()
        
        # Color map for vehicle types
        color_map = {
            VehicleType.CAR: (255, 100, 0),        # Blue
            VehicleType.TRUCK: (0, 0, 255),        # Red
            VehicleType.MOTORCYCLE: (0, 255, 255), # Yellow
            VehicleType.BUS: (0, 255, 0)           # Green
        }
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = color_map.get(det.vehicle_type, (255, 255, 255))
            
            # Draw vehicle bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(vis, det.center, 5, color, -1)
            
            # Draw license plate region if detected
            if det.has_license_plate_region and det.license_plate_bbox:
                px1, py1, px2, py2 = det.license_plate_bbox
                cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 255, 0), 1)
                cv2.putText(
                    vis, "PLATE", (px1, py1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
                )
            
            # Draw label
            label = f"{det.vehicle_type.display_name} {det.confidence:.0%}"
            size_indicator = det.vehicle_size.name[0]  # S/M/L
            label += f" [{size_indicator}]"
            
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Label background
            cv2.rectangle(
                vis, 
                (x1, y1 - label_h - 10), 
                (x1 + label_w + 10, y1),
                color, 
                -1
            )
            
            cv2.putText(
                vis, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            
            # Threat level indicator
            threat = det.characteristics.base_threat_level
            threat_color = (
                (0, 255, 0) if threat < 40 else
                (0, 255, 255) if threat < 70 else
                (0, 0, 255)
            )
            cv2.putText(
                vis, f"T:{threat}", (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, threat_color, 2
            )
        
        # Draw statistics
        fps = 1.0 / inference_time if inference_time > 0 else 0
        stats = [
            f"Vehicles: {len(detections)}",
            f"FPS: {fps:.1f}",
            f"Inference: {inference_time*1000:.1f}ms"
        ]
        
        # Count by type
        type_counts = {}
        for det in detections:
            type_name = det.vehicle_type.display_name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        if type_counts:
            stats.append("---")
            for vtype, count in type_counts.items():
                stats.append(f"{vtype}: {count}")
        
        y_offset = 30
        for stat in stats:
            cv2.putText(
                vis, stat, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )
            y_offset += 25
        
        return vis
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        avg_time = (
            self.total_inference_time / self.frame_count 
            if self.frame_count > 0 else 0
        )
        
        return {
            'avg_fps': 1.0 / avg_time if avg_time > 0 else 0,
            'avg_inference_time_ms': avg_time * 1000,
            'total_frames': self.frame_count,
            'skipped_frames': self.skipped_frames
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.skipped_frames = 0
