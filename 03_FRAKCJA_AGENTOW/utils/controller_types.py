from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


GridCell = Tuple[int, int]


@dataclass
class NavigatorOutput:
    mode: str
    target_world: Tuple[float, float]
    waypoint_world: Tuple[float, float]
    desired_heading_deg: float
    heading_error_deg: float
    distance_to_waypoint: float
    distance_to_target: float
    path_cells: List[GridCell]
    avoid_obstacle: bool
    seen_enemy: bool
    seen_powerup: bool


@dataclass
class MotionOutput:
    move_speed: float
    heading_rotation_angle: float
    stuck_ticks: int


@dataclass
class WeaponOutput:
    barrel_rotation_angle: float
    ammo_to_load: str
    should_fire: bool
    fire_score: float
    aim_error_deg: float
    target_visible: bool


@dataclass
class AgentAction:
    move_speed: float
    heading_rotation_angle: float
    barrel_rotation_angle: float
    ammo_to_load: str
    should_fire: bool
