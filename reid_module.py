import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum

class ReIDConfidence(Enum):
    HIGH = 0.9
    MEDIUM = 0.7
    LOW = 0.5
    REJECT = 0.0

@dataclass
class GaitFeatures:
    stride_length: float
    gait_speed: float
    vertical_frequency: float
    lateral_frequency: float
    posture_angle: float
    confidence: float = 0.0

class AppearanceMatcher:
    def compute_cosine_similarity(self, feat1, feat2):
        if len(feat1) == 0 or len(feat2) == 0:
            return 0.0
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-6)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-6)
        return float(np.clip(np.dot(feat1_norm, feat2_norm), 0, 1))

class GaitMatcher:
    def extract_gait_features(self, detections, camera_id):
        if len(detections) < 3:
            return None
        centers, heights = [], []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
            heights.append(y2 - y1)
        strides = [abs(centers[i][0] - centers[i-1][0]) for i in range(1, len(centers))]
        stride_length = float(np.mean(strides)) if strides else 0.0
        speed = stride_length / (np.mean(heights) + 1e-6)
        vertical_diff = np.std([c[1] for c in centers])
        lateral_diff = np.std([c[0] for c in centers])
        widths = [x2 - x1 for x1, y1, x2, y2 in [det.bbox for det in detections]]
        width_trend = float(np.mean(np.diff(widths))) if len(widths) > 1 else 0.0
        posture_angle = np.clip(width_trend * 10, -45, 45)
        return GaitFeatures(stride_length, speed, vertical_diff, lateral_diff, posture_angle, 0.8)
    
    def match_gait_features(self, gait1, gait2, camera_transition=False):
        if not gait1 or not gait2:
            return 0.0
        tolerance = 0.5 if camera_transition else 0.3
        scores = []
        for g1_val, g2_val in [(gait1.stride_length, gait2.stride_length), (gait1.gait_speed, gait2.gait_speed), (gait1.vertical_frequency, gait2.vertical_frequency)]:
            max_val = max(abs(g1_val), abs(g2_val)) + 1e-6
            diff = abs(g1_val - g2_val) / max_val
            scores.append(1.0 - min(diff, tolerance) / tolerance)
        return float(np.mean(scores))

class TemporalValidator:
    def validate_temporal(self, track1_last_frame, track2_first_frame, gap_tolerance=30):
        gap = track2_first_frame - track1_last_frame
        if gap <= 0: return 0.0
        if gap <= gap_tolerance: return 1.0
        if gap <= 300: return 1.0 - (gap - gap_tolerance) / (300 - gap_tolerance)
        return 0.0
