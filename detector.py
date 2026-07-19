"""
Person Detector - Core Detection Engine

This is the heart of Layer 0.

Real-world scenario this handles:
-----------------------------------
Camera feed shows movement at 2:47 AM in Sector 7.

Question: Is there a person?

This module answers:
- YES/NO (is person present?)
- WHERE (bounding box coordinates)
- HOW CONFIDENT (0-100% confidence)
- HOW MANY (multiple persons?)

Edge cases handled:
- Person partially hidden behind tree
- Multiple people overlapping
- Poor lighting conditions
- Fast-moving subjects
- Distant subjects (small in frame)
- Animals that might look person-like from distance
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

from .config import PersonDetectionConfig


@dataclass
class Detection:
    """
    Single person detection result.
    
    In a real surveillance system, each detection becomes a record
    that flows through the entire analysis pipeline.
    """
    
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    """
    Bounding box coordinates.
    Example: (100, 150, 200, 400)
    - Person's top-left corner at pixel (100, 150)
    - Person's bottom-right at pixel (200, 400)
    """
    
    confidence: float  # 0.0 to 1.0
    """
    Model's confidence in this detection.
    
    Real interpretation:
    - 0.95+: Almost certain (clear view, good lighting)
    - 0.70-0.95: High confidence (normal conditions)
    - 0.50-0.70: Medium confidence (partial occlusion, distance)
    - 0.45-0.50: Low confidence (poor conditions, needs verification)
    - <0.45: Rejected (too uncertain)
    """
    
    class_id: int = 0  # COCO dataset: 0 = person
    """
    Class identifier. In COCO dataset (what YOLOv8 uses):
    - 0 = person
    - 1 = bicycle
    - 2 = car
    - etc.
    
    We filter for class_id == 0 only.
    """
    
    area: int = 0
    """
    Bounding box area in pixels.
    Used for filtering very small detections.
    """
    
    center: Tuple[int, int] = (0, 0)
    """
    Center point of bounding box.
    Used for:
    - Zone violation detection
    - Distance calculations
    - Tracking correlation
    """
    
    def __post_init__(self):
        """Calculate derived fields."""
        x1, y1, x2, y2 = self.bbox
        self.area = (x2 - x1) * (y2 - y1)
        self.center = (
            (x1 + x2) // 2,
            (y1 + y2) // 2
        )


class PersonDetector:
    """
    Core person detection system.
    
    Usage:
        detector = PersonDetector()
        detections = detector.detect(frame)
        
    Real deployment:
        - Runs continuously on surveillance feeds
        - Processes 30-60 frames per second
        - Feeds results to tracking and behavior analysis modules
    """
    
    def __init__(self, config: Optional[PersonDetectionConfig] = None):
        """
        Initialize detector with configuration.
        
        Args:
            config: Detection configuration. If None, uses defaults.
        """
        self.config = config or PersonDetectionConfig()
        
        # Load YOLOv8 model
        print(f"Loading YOLOv8{self.config.model_size} model...")
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
        
        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.skipped_frames = 0
    
    def detect(
        self, 
        source: Union[np.ndarray, str, Path],
        visualize: bool = False
    ) -> Tuple[List[Detection], Optional[np.ndarray]]:
        """
        Detect persons in image/frame.
        
        Args:
            source: Input image as numpy array, or path to image/video
            visualize: If True, return annotated image
            
        Returns:
            (detections, annotated_image)
            - detections: List of Detection objects
            - annotated_image: Image with bounding boxes (if visualize=True)
            
        Real-world example:
            # From camera feed
            ret, frame = camera.read()
            detections, viz = detector.detect(frame, visualize=True)
            
            if len(detections) > 0:
                print(f"ALERT: {len(detections)} person(s) detected")
                for det in detections:
                    print(f"  Confidence: {det.confidence:.2%}")
                    print(f"  Location: {det.center}")
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
            classes=[0],  # Only detect persons (class 0 in COCO)
            verbose=False
        )
        
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        
        # Extract detections
        detections = self._extract_detections(results[0])
        
        # Visualization
        annotated = None
        if visualize:
            annotated = self._visualize_detections(frame, detections, inference_time)
        
        return detections, annotated
    
    def _extract_detections(self, result) -> List[Detection]:
        """
        Extract detection objects from YOLO results.
        
        Filters:
        - Minimum area threshold (removes tiny false positives)
        - Confidence threshold (already applied by YOLO)
        - Person class only (class_id == 0)
        """
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            if cls_id != 0:  # Skip non-person detections
                continue
            
            x1, y1, x2, y2 = map(int, box)
            
            detection = Detection(
                bbox=(x1, y1, x2, y2),
                confidence=float(conf),
                class_id=int(cls_id)
            )
            
            # Filter by minimum area
            if detection.area < self.config.min_detection_area:
                continue
            
            detections.append(detection)
        
        return detections
    
    def _enhance_low_light(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance low-light frames for better detection.
        
        Real scenario:
        - Night patrol cameras
        - Foggy/rainy conditions
        - Indoor areas with poor lighting
        
        Techniques:
        - CLAHE (Contrast Limited Adaptive Histogram Equalization)
        - Gamma correction
        """
        if len(frame.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            return enhanced
        else:
            # Grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(frame)
    
    def _visualize_detections(
        self, 
        frame: np.ndarray, 
        detections: List[Detection],
        inference_time: float
    ) -> np.ndarray:
        """
        Draw bounding boxes and information on frame.
        
        Color coding:
        - Green: High confidence (>0.8)
        - Yellow: Medium confidence (0.6-0.8)
        - Orange: Low confidence (0.45-0.6)
        """
        vis = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Color based on confidence
            if det.confidence > 0.8:
                color = (0, 255, 0)  # Green
                label_bg = (0, 200, 0)
            elif det.confidence > 0.6:
                color = (0, 255, 255)  # Yellow
                label_bg = (0, 200, 200)
            else:
                color = (0, 165, 255)  # Orange
                label_bg = (0, 130, 200)
            
            # Draw bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(vis, det.center, 5, color, -1)
            
            # Draw label
            label = f"Person {det.confidence:.2%}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(
                vis, 
                (x1, y1 - label_h - 10), 
                (x1 + label_w + 10, y1),
                label_bg, 
                -1
            )
            
            cv2.putText(
                vis, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
        
        # Draw statistics
        fps = 1.0 / inference_time if inference_time > 0 else 0
        stats = [
            f"Detections: {len(detections)}",
            f"FPS: {fps:.1f}",
            f"Inference: {inference_time*1000:.1f}ms"
        ]
        
        y_offset = 30
        for stat in stats:
            cv2.putText(
                vis, stat, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            y_offset += 30
        
        return vis
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with:
            - avg_fps: Average frames per second
            - avg_inference_time: Average inference time in ms
            - total_frames: Total frames processed
            - skipped_frames: Frames skipped for performance
        """
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
