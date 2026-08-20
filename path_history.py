
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from datetime import datetime

@dataclass
class PathPoint:
    frame: int
    position: Tuple[int, int]
    velocity: Tuple[float, float]
    timestamp: float

@dataclass
class PathHistory:
    global_id: str
    path_points: List[PathPoint] = field(default_factory=list)
    total_distance: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    
    def add_point(self, point: PathPoint):
        if self.path_points:
            prev = self.path_points[-1]
            dist = np.sqrt((point.position[0]-prev.position[0])**2 + 
                          (point.position[1]-prev.position[1])**2)
            self.total_distance += dist
        self.path_points.append(point)
    
    def get_trajectory(self) -> List[Tuple[int, int]]:
        return [p.position for p in self.path_points]
    
    def get_duration_frames(self) -> int:
        if not self.path_points:
            return 0
        return self.path_points[-1].frame - self.path_points[0].frame
