from __future__ import annotations

import math
from typing import Any, Tuple


def to_xy(value: Any) -> Tuple[float, float]:
    if isinstance(value, dict):
        return float(value.get("x", 0.0)), float(value.get("y", 0.0))
    return float(getattr(value, "x", 0.0)), float(getattr(value, "y", 0.0))


def normalize_angle_diff(target_angle: float, current_angle: float) -> float:
    diff = target_angle - current_angle
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


def heading_to_angle_deg(from_x: float, from_y: float, to_x: float, to_y: float) -> float:
    return math.degrees(math.atan2(to_y - from_y, to_x - from_x)) % 360


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)
