"""
System widzenia czołgów

Przeszkody blokują widoczność (chyba że są przezierne jak AntiTankSpike).
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
    return math.hypot(dx, dy)


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


def check_segment_aabb_intersection(
    start: Position,
    end: Position,
    box_center: Position,
    box_size: List[int]
) -> bool:
    """Sprawdza czy odcinek przecina prostokąt (AABB)."""
    half_w = box_size[0] / 2.0
    half_h = box_size[1] / 2.0
    
    min_x = box_center.x - half_w
    max_x = box_center.x + half_w
    min_y = box_center.y - half_h
    max_y = box_center.y + half_h

    dx = end.x - start.x
    dy = end.y - start.y

    if abs(dx) < 1e-9:
        if start.x < min_x or start.x > max_x:
            return False
        t_min_x = float('-inf')
        t_max_x = float('inf')
    else:
        t1 = (min_x - start.x) / dx
        t2 = (max_x - start.x) / dx
        t_min_x = min(t1, t2)
        t_max_x = max(t1, t2)

    if abs(dy) < 1e-9:
        if start.y < min_y or start.y > max_y:
            return False
        t_min_y = float('-inf')
        t_max_y = float('inf')
    else:
        t1 = (min_y - start.y) / dy
        t2 = (max_y - start.y) / dy
        t_min_y = min(t1, t2)
        t_max_y = max(t1, t2)

    t_enter = max(t_min_x, t_min_y)
    t_exit = min(t_max_x, t_max_y)

    if t_enter > t_exit:
        return False
        
    if t_exit < 0:
        return False
        
    if t_enter > 1:
        return False
        
    return True


def is_line_of_sight_blocked(
    start_pos: Position,
    end_pos: Position,
    obstacles: List[ObstacleUnion],
    ignore_id: Union[str, None] = None
) -> bool:
    """Sprawdza czy linia widzenia jest zablokowana przez nieprzezierne przeszkody."""
    for obstacle in obstacles:
        if not obstacle.is_alive:
            continue
            
        if ignore_id and obstacle._id == ignore_id:
            continue

        # Sprawdzenie czy przeszkoda jest przezierna
        # W final_api.py konkretne klasy (Wall, Tree, AntiTankSpike) definiują obstacle_type
        if hasattr(obstacle, "obstacle_type") and obstacle.obstacle_type.value.get("see_through", False):
            continue
            
        pos = getattr(obstacle, "_position", getattr(obstacle, "position", None))
        size = getattr(obstacle, "_size", getattr(obstacle, "size", None))
        
        if pos and size and check_segment_aabb_intersection(start_pos, end_pos, pos, size):
            return True
            
    return False


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
        if other_tank._id == tank._id:
            continue
        if other_tank.hp <= 0:
            continue

        distance = calculate_distance(origin, other_tank.position)
        if distance > tank._vision_range:
            continue

        angle_to_target = calculate_angle_to_target(origin, other_tank.position)
        if not is_in_vision_cone(
            tank.heading,
            tank.barrel_angle,
            tank._vision_angle,
            angle_to_target
        ):
            continue

        if is_line_of_sight_blocked(origin, other_tank.position, obstacles):
            continue

        seen_tanks.append(
            SeenTank(
                id=other_tank._id,
                team=other_tank._team,
                tank_type=other_tank._tank_type,
                position=other_tank.position,
                is_damaged=other_tank.hp < 0.3 * other_tank._max_hp,
                heading=other_tank.heading,
                barrel_angle=other_tank.barrel_angle,
                distance=distance
            )
        )

    # =========================
    # POWERUPY
    # =========================
    for powerup in powerups:
        distance = calculate_distance(origin, powerup._position)
        if distance > tank._vision_range:
            continue

        angle_to_target = calculate_angle_to_target(origin, powerup._position)
        if is_in_vision_cone(
            tank.heading,
            tank.barrel_angle,
            tank._vision_angle,
            angle_to_target
        ):
            if not is_line_of_sight_blocked(origin, powerup._position, obstacles):
                seen_powerups.append(powerup)

    # =========================
    # PRZESZKODY
    # =========================
    for obstacle in obstacles:
        if not obstacle.is_alive:
            continue

        distance = calculate_distance(origin, obstacle._position)
        if distance > tank._vision_range:
            continue

        angle_to_target = calculate_angle_to_target(origin, obstacle._position)
        if is_in_vision_cone(
            tank.heading,
            tank.barrel_angle,
            tank._vision_angle,
            angle_to_target
        ):
            if not is_line_of_sight_blocked(origin, obstacle._position, obstacles, ignore_id=obstacle._id):
                seen_obstacles.append(obstacle)

    # =========================
    # TERENY
    # =========================
    for terrain in terrains:
        distance = calculate_distance(origin, terrain._position)
        if distance > tank._vision_range:
            continue

        angle_to_target = calculate_angle_to_target(origin, terrain._position)
        if is_in_vision_cone(
            tank.heading,
            tank.barrel_angle,
            tank._vision_angle,
            angle_to_target
        ):
            if not is_line_of_sight_blocked(origin, terrain._position, obstacles):
                seen_terrains.append(terrain)

    return TankSensorData(
        seen_tanks=seen_tanks,
        seen_powerups=seen_powerups,
        seen_obstacles=seen_obstacles,
        seen_terrains=seen_terrains
    )