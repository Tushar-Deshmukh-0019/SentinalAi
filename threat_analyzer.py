
import numpy as np
from dataclasses import dataclass
from enum import Enum

class ThreatLevel(Enum):
    LOW = 0.0
    MEDIUM = 0.5
    HIGH = 1.0

@dataclass
class ThreatScore:
    loitering_score: float
    erratic_score: float
    behavior_score: float
    final_score: float
    threat_level: ThreatLevel

class ThreatAnalyzer:
    def score_threat(self, loitering_frames: int, speed_variance: float, behavior_type: str) -> ThreatScore:
        loitering = min(1.0, loitering_frames / 300)
        erratic = min(1.0, speed_variance)
        
        behavior_map = {
            'steady_walk': 0.1,
            'normal_variation': 0.3,
            'erratic': 0.7,
            'loitering': 0.8,
            'stationary': 0.5
        }
        behavior_score = behavior_map.get(behavior_type, 0.5)
        
        final = (0.4 * loitering) + (0.3 * erratic) + (0.3 * behavior_score)
        
        if final < 0.3:
            level = ThreatLevel.LOW
        elif final < 0.6:
            level = ThreatLevel.MEDIUM
        else:
            level = ThreatLevel.HIGH
        
        return ThreatScore(loitering, erratic, behavior_score, final, level)
