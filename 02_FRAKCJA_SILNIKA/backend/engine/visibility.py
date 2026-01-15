"""
System widzenia czołgów

Przeszkody póki co nie zasłaniają innych obiektów.
"""

import math
from typing import List, Union

from ..structures import Position, ObstacleUnion, TerrainUnion, PowerUpData
from ..tank.heavy_tank import HeavyTank
from ..tank.light_tank import LightTank
from ..tank.sniper_tank import SniperTank
from ..tank.sensor_data import TankSensorData, SeenTank

TankUnion = Union[LightTank, HeavyTank, SniperTank]


def normalize_angle(angle: float) -> float:
    """Normalizuje kąt do zakresu [-180, 180]."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def calculate_distance(pos1: Position, pos2: Position) -> float:
    """Oblicza odległość euklidesową."""
    dx = pos2.x - pos1.x
    dy = pos2.y - pos1.y
    return math.sqrt(dx * dx + dy * dy)


def calculate_angle_to_target(from_pos: Position, to_pos: Position) -> float:
    """
    Oblicza kąt (w stopniach) od pozycji źródłowej do celu.

    Args:
        from_pos: Pozycja źródłowa
        to_pos: Pozycja docelowa

    Returns:
        Kąt w stopniach (0° = wschód, 90° = północ)
    """
    dx = to_pos.x - from_pos.x
    dy = to_pos.y - from_pos.y
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def is_in_vision_cone(
    tank_heading: float,
    tank_barrel: float,
    vision_angle: float,
    angle_to_target: float
) -> bool:
    """
    Sprawdza, czy cel znajduje się w stożku widzenia czołgu.

    Args:
        tank_heading: Kąt kadłuba czołgu (stopnie)
        tank_barrel: Kąt lufy względem kadłuba (stopnie)
        vision_angle: Kąt widzenia czołgu (stopnie)
        angle_to_target: Kąt do celu (stopnie)

    Returns:
        True jeśli cel jest w polu widzenia
    """
    view_direction = normalize_angle(tank_heading + tank_barrel)
    angle_diff = abs(normalize_angle(angle_to_target - view_direction))
    return angle_diff <= vision_angle / 2.0


def check_visibility(
    tank: TankUnion,
    all_tanks: List[TankUnion],
    obstacles: List[ObstacleUnion],
    terrains: List[TerrainUnion],
    powerups: List[PowerUpData]
) -> TankSensorData:
    """
    Wykrywa wszystkie obiekty w polu widzenia czołgu.

    Args:
        tank: Czołg obserwujący
        all_tanks: Lista wszystkich czołgów na mapie
        obstacles: Lista przeszkód
        terrains: Lista terenów
        powerups: Lista powerupów

    Returns:
        TankSensorData zawierający wszystkie wykryte obiekty
    """
    seen_tanks: List[SeenTank] = []
    seen_powerups: List[PowerUpData] = []
    seen_obstacles: List[ObstacleUnion] = []
    seen_terrains: List[TerrainUnion] = []

    origin = tank.position

    # =========================
    # CZOŁGI
    # =========================
    for other_tank in all_tanks:
        if other_tank.id == tank.id: # type: ignore
            continue
        if other_tank.hp <= 0:
            continue

        distance = calculate_distance(origin, other_tank.position)
        if distance > tank._vision_range:
            continue

        angle_to_target = calculate_angle_to_target(origin, other_tank.position) # type: ignore
        if not is_in_vision_cone(
            tank.heading,
            tank.barrel_angle,
            tank._vision_angle,
            angle_to_target
        ):
            continue

        seen_tanks.append(
            SeenTank(
                id=other_tank.id, # type: ignore
                team=other_tank.team, # type: ignore
                tank_type=other_tank.tank_type, # type: ignore
                position=other_tank.position, # type: ignore
                is_damaged=other_tank.hp < 0.3 * other_tank._max_hp, # type: ignore
                heading=other_tank.heading, # type: ignore
                barrel_angle=other_tank.barrel_angle, # type: ignore
                distance=distance
            )
        )

    # =========================
    # POWERUPY
    # =========================
    for powerup in powerups:
        distance = calculate_distance(origin, powerup.position)
        if distance > tank._vision_range:
            continue

        angle_to_target = calculate_angle_to_target(origin, powerup.position)
        if is_in_vision_cone(
            tank.heading,
            tank.barrel_angle,
            tank._vision_angle,
            angle_to_target
        ):
            seen_powerups.append(powerup)

    # =========================
    # PRZESZKODY
    # =========================
    for obstacle in obstacles:
        if not obstacle.is_alive:
            continue

        distance = calculate_distance(origin, obstacle.position)
        if distance > tank._vision_range:
            continue

        angle_to_target = calculate_angle_to_target(origin, obstacle.position)
        if is_in_vision_cone(
            tank.heading,
            tank.barrel_angle,
            tank._vision_angle,
            angle_to_target
        ):
            seen_obstacles.append(obstacle)

    # =========================
    # TERENY
    # =========================
    for terrain in terrains:
        distance = calculate_distance(origin, terrain.position)
        if distance > tank._vision_range:
            continue

        angle_to_target = calculate_angle_to_target(origin, terrain.position)
        if is_in_vision_cone(
            tank.heading,
            tank.barrel_angle,
            tank._vision_angle,
            angle_to_target
        ):
            seen_terrains.append(terrain)

    return TankSensorData(
        seen_tanks=seen_tanks,
        seen_powerups=seen_powerups,
        seen_obstacles=seen_obstacles,
        seen_terrains=seen_terrains
    )