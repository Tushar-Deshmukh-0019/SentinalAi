
import numpy as np
from dataclasses import dataclass
from typing import List
from enum import Enum

class BehaviorType(Enum):
    STEADY_WALK = 'steady_walk'
    NORMAL_VARIATION = 'normal_variation'
    ERRATIC = 'erratic'
    LOITERING = 'loitering'
    STATIONARY = 'stationary'

@dataclass
class BehaviorProfile:
    behavior_type: BehaviorType
    confidence: float
    movement_variance: float
    average_speed: float
    max_speed: float

class BehaviorAnalyzer:
    def analyze_movement(self, speeds: List[float]) -> BehaviorProfile:
        if not speeds:
            return BehaviorProfile(BehaviorType.STATIONARY, 1.0, 0, 0, 0)
        
        avg_speed = np.mean(speeds)
        std_speed = np.std(speeds)
        max_speed = np.max(speeds)
        
        variance_ratio = std_speed / (avg_speed + 1e-6)
        
        if max_speed < 5:
            btype = BehaviorType.STATIONARY
        elif variance_ratio < 0.2:
            btype = BehaviorType.STEADY_WALK
        elif variance_ratio < 0.5:
            btype = BehaviorType.NORMAL_VARIATION
        else:
            btype = BehaviorType.ERRATIC
        
        confidence = 1.0 - min(0.3, variance_ratio)
        
        return BehaviorProfile(btype, confidence, variance_ratio, avg_speed, max_speed)
