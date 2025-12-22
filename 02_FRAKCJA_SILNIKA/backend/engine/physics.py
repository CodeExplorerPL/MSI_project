"""
Fizyka gry - Ruch, kolizje, strzały, interakcje z otoczeniem
"""

import math
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

from _01_DOKUMENTACJA.final_api import (
    TankUnion, Position, MapInfo,
    ObstacleUnion, TerrainUnion, PowerUpData,
    AmmoType, PowerUpType, ActionCommand
)


# ============================================================
# KOLIZJE – STRUKTURY
# ============================================================

class CollisionType(Enum):
    """Typy kolizji w grze."""
    NONE = "none"
    TANK_TANK_MOVING = "tank_tank_moving"
    TANK_TANK_STATIC = "tank_tank_static"
    TANK_WALL = "tank_wall"
    TANK_TREE = "tank_tree"
    TANK_SPIKE = "tank_spike"
    TANK_BOUNDARY = "tank_boundary"


@dataclass
class CollisionResult:
    """Wynik sprawdzenia kolizji."""
    has_collision: bool
    collision_type: CollisionType
    damage_to_tank1: int = 0
    damage_to_tank2: int = 0
    obstacle_destroyed: Optional[str] = None


@dataclass
class ProjectileHit:
    """Informacja o trafieniu pocisku."""
    hit_tank_id: Optional[str] = None
    hit_obstacle_id: Optional[str] = None
    damage_dealt: int = 0
    hit_position: Optional[Position] = None


# ============================================================
# NARZĘDZIA
# ============================================================

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


def rectangles_overlap(
    pos1: Position, size1: List[int],
    pos2: Position, size2: List[int]
) -> bool:
    """
    Sprawdza, czy dwa prostokąty (AABB) nachodzą na siebie.
    Pozycje to środki prostokątów.
    """
    half_w1, half_h1 = size1[0] / 2.0, size1[1] / 2.0
    half_w2, half_h2 = size2[0] / 2.0, size2[1] / 2.0

    return not (
        pos1.x + half_w1 <= pos2.x - half_w2 or
        pos1.x - half_w1 >= pos2.x + half_w2 or
        pos1.y + half_h1 <= pos2.y - half_h2 or
        pos1.y - half_h1 >= pos2.y + half_h2
    )


# ============================================================
# SYSTEM RUCHU
# ============================================================

def get_terrain_at_position(
    position: Position,
    terrains: List[TerrainUnion]
) -> Optional[TerrainUnion]:
    """
    Znajduje teren na danej pozycji.
    Zwraca pierwszy teren, którego bounding box zawiera pozycję.
    """
    for terrain in terrains:
        if rectangles_overlap(position, [1, 1], terrain._position, terrain._size):
            return terrain
    return None


def rotate_heading(tank: TankUnion, delta_heading: float) -> None:
    """
    Obraca kadłub czołgu o delta (API: zmiana kąta).
    """
    delta = max(
        -tank._heading_spin_rate,
        min(delta_heading, tank._heading_spin_rate)
    )
    tank.heading = normalize_angle(tank.heading + delta)


def rotate_barrel(tank: TankUnion, delta_barrel: float) -> None:
    """
    Obraca lufę czołgu o delta (API: zmiana kąta).
    """
    delta = max(
        -tank._barrel_spin_rate,
        min(delta_barrel, tank._barrel_spin_rate)
    )
    tank.barrel_angle = normalize_angle(tank.barrel_angle + delta)


def move_tank(
    tank: TankUnion,
    desired_speed: float,
    terrains: List[TerrainUnion],
    delta_time: float
) -> Tuple[Position, int]:
    """
    Przesuwa czołg zgodnie z jego prędkością i modyfikatorami terenu.
    """
    speed = max(-tank._top_speed, min(desired_speed, tank._top_speed))
    tank.move_speed = speed

    terrain = get_terrain_at_position(tank.position, terrains)
    modifier = terrain._movement_speed_modifier if terrain else 1.0
    damage = terrain._deal_damage if terrain else 0

    effective_speed = speed * modifier
    heading_rad = math.radians(tank.heading)

    new_position = Position(
        tank.position.x + math.cos(heading_rad) * effective_speed * delta_time,
        tank.position.y + math.sin(heading_rad) * effective_speed * delta_time
    )

    return new_position, damage


# ============================================================
# SYSTEM KOLIZJI
# ============================================================

def check_tank_boundary_collision(
    tank: TankUnion,
    map_size: List[int]
) -> bool:
    """Sprawdza, czy czołg wychodzi poza granice mapy."""
    half_w, half_h = tank._size[0] / 2, tank._size[1] / 2
    return (
        tank.position.x - half_w < 0 or
        tank.position.x + half_w > map_size[0] or
        tank.position.y - half_h < 0 or
        tank.position.y + half_h > map_size[1]
    )


def check_tank_obstacle_collision(
    tank: TankUnion,
    obstacles: List[ObstacleUnion]
) -> Optional[ObstacleUnion]:
    """Sprawdza kolizję czołgu z przeszkodami."""
    for obstacle in obstacles:
        if not obstacle.is_alive:
            continue
        if rectangles_overlap(tank.position, tank._size, obstacle._position, obstacle._size):
            return obstacle
    return None


def check_tank_tank_collision(
    tank1: TankUnion,
    tank2: TankUnion
) -> bool:
    """Sprawdza kolizję między dwoma czołgami."""
    if tank1._id == tank2._id:
        return False
    return rectangles_overlap(
        tank1.position, tank1._size,
        tank2.position, tank2._size
    )


# ============================================================
# SYSTEM STRZAŁÓW I RELOADU
# ============================================================

def update_reload(tank: TankUnion) -> None:
    if tank.current_reload_progress > 0:
        tank.current_reload_progress -= 1


def try_load_ammo(tank: TankUnion, ammo: Optional[AmmoType]) -> None:
    if ammo is None:
        return
    if ammo not in tank.ammo:
        return
    if tank.ammo[ammo].count <= 0:
        return
    tank.ammo_loaded = ammo
    tank.current_reload_progress = ammo.value["ReloadTime"]


def can_fire(tank: TankUnion) -> bool:
    return (
        tank.ammo_loaded is not None and
        tank.current_reload_progress == 0 and
        tank.ammo[tank.ammo_loaded].count > 0
    )


def fire_projectile(
    tank: TankUnion,
    all_tanks: List[TankUnion],
    obstacles: List[ObstacleUnion]
) -> Optional[ProjectileHit]:
    """
    Wykonuje strzał z czołgu.
    """
    if not can_fire(tank):
        return None

    ammo = tank.ammo_loaded
    tank.ammo[ammo].count -= 1
    tank.current_reload_progress = ammo.value["ReloadTime"]

    damage = abs(ammo.value["Value"])
    if tank.is_overcharged:
        damage *= 2
        tank.is_overcharged = False

    shoot_direction = normalize_angle(tank.heading + tank.barrel_angle)

    closest_hit_distance = ammo.value["Range"]
    hit = None

    for target in all_tanks:
        if target._id == tank._id:
            continue

        dist = calculate_distance(tank.position, target.position)
        if dist > closest_hit_distance:
            continue

        angle = math.degrees(math.atan2(
            target.position.y - tank.position.y,
            target.position.x - tank.position.x
        ))

        if abs(normalize_angle(angle - shoot_direction)) <= 5:
            closest_hit_distance = dist
            hit = ProjectileHit(
                hit_tank_id=target._id,
                damage_dealt=damage,
                hit_position=target.position
            )

    for obstacle in obstacles:
        if not obstacle.is_alive:
            continue

        dist = calculate_distance(tank.position, obstacle._position)
        if dist < closest_hit_distance:
            angle = math.degrees(math.atan2(
                obstacle._position.y - tank.position.y,
                obstacle._position.x - tank.position.x
            ))

            if abs(normalize_angle(angle - shoot_direction)) <= 5:
                if obstacle.is_destructible:
                    obstacle.is_alive = False
                    return ProjectileHit(hit_obstacle_id=obstacle._id)
                return None

    return hit


# ============================================================
# OBRAŻENIA
# ============================================================

def apply_damage(tank: TankUnion, damage: int) -> bool:
    """Zadaje obrażenia czołgowi (najpierw shield, potem HP)."""
    remaining = damage

    if tank.shield > 0:
        absorbed = min(tank.shield, remaining)
        tank.shield -= absorbed
        remaining -= absorbed

    tank.hp -= remaining
    return tank.hp <= 0


# ============================================================
# POWERUPY
# ============================================================

def check_powerup_pickup(
    tank: TankUnion,
    powerups: List[PowerUpData]
) -> Optional[PowerUpData]:
    """Sprawdza, czy czołg jest na powerupie i może go podnieść."""
    for powerup in powerups:
        if rectangles_overlap(
            tank.position, tank._size,
            powerup._position, powerup._size
        ):
            return powerup
    return None


def apply_powerup(tank: TankUnion, powerup: PowerUpData) -> None:
    """Aplikuje efekt powerupu na czołg."""
    ptype = powerup._powerup_type

    if ptype == PowerUpType.MEDKIT:
        tank.hp = min(tank.hp + powerup.value, tank._max_hp)

    elif ptype == PowerUpType.SHIELD:
        tank.shield = min(tank.shield + powerup.value, tank._max_shield)

    elif ptype == PowerUpType.OVERCHARGE:
        tank.is_overcharged = True

    else:
        ammo = AmmoType[ptype.value["AmmoType"]]
        tank.ammo[ammo].count = min(
            tank.ammo[ammo].count + powerup.value,
            tank._max_ammo[ammo]
        )


# ============================================================
# FUNKCJA GŁÓWNA – TICK
# ============================================================

def process_physics_tick(
    all_tanks: List[TankUnion],
    actions: Dict[str, ActionCommand],
    map_info: MapInfo,
    delta_time: float
) -> Dict[str, list]:
    """
    Przetwarza jedną turę fizyki gry.
    """
    results = {
        "collisions": [],
        "projectile_hits": [],
        "picked_powerups": [],
        "destroyed_tanks": [],
        "destroyed_obstacles": []
    }

    for tank in all_tanks:
        update_reload(tank)

    for tank in all_tanks:
        if tank.hp <= 0:
            continue

        action = actions.get(tank._id)
        if not action:
            continue

        rotate_heading(tank, action.heading_rotation_angle)
        rotate_barrel(tank, action.barrel_rotation_angle)
        try_load_ammo(tank, action.ammo_to_load)

    for tank in all_tanks:
        if tank.hp <= 0:
            continue

        action = actions.get(tank._id)
        if action and action.should_fire:
            hit = fire_projectile(tank, all_tanks, map_info.obstacle_list)
            if hit:
                results["projectile_hits"].append(hit)
                if hit.hit_tank_id:
                    results["destroyed_tanks"].append(hit.hit_tank_id)
                if hit.hit_obstacle_id:
                    results["destroyed_obstacles"].append(hit.hit_obstacle_id)

    for tank in all_tanks:
        if tank.hp <= 0:
            continue

        action = actions.get(tank._id)
        if not action or action.move_speed == 0:
            continue

        new_pos, dmg = move_tank(
            tank, action.move_speed,
            map_info.terrain_list, delta_time
        )
        tank.position = new_pos

        if dmg and apply_damage(tank, dmg):
            results["destroyed_tanks"].append(tank._id)

    for tank in all_tanks:
        powerup = check_powerup_pickup(tank, map_info.powerup_list)
        if powerup:
            apply_powerup(tank, powerup)
            map_info.powerup_list.remove(powerup)
            results["picked_powerups"].append(
                {"tank_id": tank._id, "powerup": powerup}
            )

    return results