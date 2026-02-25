from __future__ import annotations
from typing import List, Tuple

AMMO_RANGE = {
    "HEAVY": 25.0,
    "LIGHT": 50.0,
    "LONG_DISTANCE": 100.0,
}

STATIC_CORRIDOR_CHECKPOINTS: List[Tuple[float, float]] = [
    (1, 18),
    (1, 17),
    (0, 16),
    (0, 9),
    (1, 8),
    (0.9, 7.1),
    (1.5, 6.5),
    (2.1, 6.1),
    (3, 5),
    (4, 4),
    (9, 4),
]

STATIC_CORRIDOR_CHECKPOINTS = [
    (x * 10 + 5, y * 10 + 5) for x, y in STATIC_CORRIDOR_CHECKPOINTS
]
STATIC_CORRIDOR_CHECKPOINTS += [
    (200 - x, y) for x, y in STATIC_CORRIDOR_CHECKPOINTS[::-1]
]


def get_firing_range(ammo_loaded: str | None) -> float:
    if not ammo_loaded:
        return AMMO_RANGE["LIGHT"]
    key = str(ammo_loaded).upper()
    return AMMO_RANGE.get(key, AMMO_RANGE["LIGHT"])


def build_checkpoints_to_enemy(
    team: int,
    start_x: float,
    start_y: float,
    map_width: float = 200.0,
    map_height: float = 200.0,
    num_checkpoints: int = 6,
    map_filename: str = "advanced_road_trees.csv",
) -> List[Tuple[float, float]]:
    _ = (start_x, start_y, map_width, map_height, num_checkpoints, map_filename)
    if team == 2:
        return list(reversed(STATIC_CORRIDOR_CHECKPOINTS))
    return list(STATIC_CORRIDOR_CHECKPOINTS)


def lane_offset_checkpoint(
    tank_id: str, point: Tuple[float, float]
) -> Tuple[float, float]:
    lane = (sum(ord(ch) for ch in str(tank_id)) % 3) - 1
    return point[0], point[1] + 4.0 * lane
