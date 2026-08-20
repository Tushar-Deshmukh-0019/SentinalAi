import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class VelocityVector:
    vx: float
    vy: float
    speed: float
    direction: float
    confidence: float

class VelocityCalculator:
    def calc_velocity_from_boxes(self, boxes, timestamps):
        if len(boxes) < 2:
            return None
        centers = []
        for x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            centers.append((cx, cy))
        velocities = []
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i-1][0]
            dy = centers[i][1] - centers[i-1][1]
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                vx = dx / dt
                vy = dy / dt
                velocities.append((vx, vy))
        if not velocities:
            return None
        avg_vx = float(np.mean([v[0] for v in velocities]))
        avg_vy = float(np.mean([v[1] for v in velocities]))
        speed = np.sqrt(avg_vx**2 + avg_vy**2)
        direction = np.degrees(np.arctan2(avg_vy, avg_vx))
        if len(velocities) > 2:
            consistency = 1.0 / (1.0 + np.std([v[0] for v in velocities]) + np.std([v[1] for v in velocities]))
        else:
            consistency = 0.8
        return VelocityVector(avg_vx, avg_vy, float(speed), float(direction), float(consistency))

class TrajectoryPredictor:
    def predict_next_position(self, current_pos, velocity, frames_ahead=30):
        x, y = current_pos
        next_x = int(x + velocity.vx * frames_ahead)
        next_y = int(y + velocity.vy * frames_ahead)
        return (next_x, next_y)
    
    def predict_trajectory(self, start_pos, velocity, num_steps=10):
        path = []
        for i in range(num_steps):
            x = int(start_pos[0] + velocity.vx * i)
            y = int(start_pos[1] + velocity.vy * i)
            path.append((x, y))
        return path
