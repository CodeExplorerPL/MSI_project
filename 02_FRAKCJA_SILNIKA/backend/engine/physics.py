"""
Fizyka gry - Ruch, kolizje, strzały, interakcje z otoczeniem
"""

import math
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from ..structures import (
    Position,
    MapInfo,
    ObstacleUnion,
    TerrainUnion,
    PowerUpData,
    PowerUpType,
    AmmoType,
    AmmoSlot,
    ObstacleType,
)
from ..tank.light_tank import LightTank
from ..tank.heavy_tank import HeavyTank
from ..tank.sniper_tank import SniperTank
import sys
import os

# Add parent directory to path for controller imports
_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _base_dir not in sys.path:
    sys.path.insert(0, _base_dir)

from controller.api import ActionCommand

TankUnion = Union[LightTank, HeavyTank, SniperTank]


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
    shooter_id: Optional[str] = None
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


def get_tank_size(tank: TankUnion) -> List[int]:
    """Zwraca rozmiar czołgu niezależnie od nazwy pola w strukturze."""
    if hasattr(tank, "size"):
        return tank.size
    return getattr(tank, "_size", [5, 5])


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
        terrain_position = getattr(terrain, "position", getattr(terrain, "_position", None))
        terrain_size = getattr(terrain, "size", getattr(terrain, "_size", [1, 1]))
        if terrain_position and rectangles_overlap(position, [1, 1], terrain_position, terrain_size):
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
    modifier = 1.0
    # Terrain damage is applied in process_physics_tick for the final position.
    damage = 0
    if terrain:
        modifier = getattr(terrain, "movement_speed_modifier", getattr(terrain, "_movement_speed_modifier", 1.0))
        damage = getattr(terrain, "deal_damage", getattr(terrain, "_deal_damage", 0))

    effective_speed = speed * modifier
    heading_rad = math.radians(tank.heading)

    new_position = Position(
        tank.position.x + math.cos(heading_rad) * effective_speed * delta_time,
        tank.position.y + math.sin(heading_rad) * effective_speed * delta_time
    )

    return new_position, damage


def _terrain_damage_at_position(
    position: Position,
    terrains: List[TerrainUnion]
) -> int:
    """Returns terrain damage dealt per tick at the given position."""
    terrain = get_terrain_at_position(position, terrains)
    if not terrain:
        return 0
    return int(getattr(terrain, "deal_damage", getattr(terrain, "_deal_damage", 0)) or 0)


# ============================================================
# SYSTEM KOLIZJI
# ============================================================

def check_tank_boundary_collision(
    tank: TankUnion,
    map_size: List[int]
) -> bool:
    """Sprawdza, czy czołg wychodzi poza granice mapy."""
    size = get_tank_size(tank)
    half_w, half_h = size[0] / 2, size[1] / 2
    return (
        tank.position.x - half_w < 0 or
        tank.position.x + half_w > map_size[0] or
        tank.position.y - half_h < 0 or
        tank.position.y + half_h > map_size[1]
    )


def _clamp_position_to_map(
    position: Position,
    size: List[int],
    map_size: Optional[List[int]]
) -> Position:
    if not map_size:
        return position
    half_w, half_h = size[0] / 2.0, size[1] / 2.0
    clamped_x = min(max(position.x, half_w), map_size[0] - half_w)
    clamped_y = min(max(position.y, half_h), map_size[1] - half_h)
    return Position(clamped_x, clamped_y)


def _candidate_has_collision(
    tank: TankUnion,
    candidate: Position,
    map_size: Optional[List[int]],
    obstacles: Optional[List[ObstacleUnion]]
) -> bool:
    original_pos = tank.position
    tank.position = candidate
    try:
        if map_size and check_tank_boundary_collision(tank, map_size):
            return True
        if obstacles and check_tank_obstacle_collision(tank, obstacles):
            return True
        return False
    finally:
        tank.position = original_pos


def resolve_tank_collision_position(
    tank: TankUnion,
    old_pos: Position,
    new_pos: Position,
    map_size: Optional[List[int]],
    obstacles: Optional[List[ObstacleUnion]],
    strong_recoil: bool = False
) -> Position:
    """
    Cofnij czołg bardziej niż 1 krok, by uniknąć ciągłego styku z przeszkodą.
    """
    size = get_tank_size(tank)
    dx = new_pos.x - old_pos.x
    dy = new_pos.y - old_pos.y
    dist = math.hypot(dx, dy)

    if dist == 0:
        return _clamp_position_to_map(old_pos, size, map_size)

    nx, ny = dx / dist, dy / dist
    
    if strong_recoil:
        # Użytkownik chce odbicie o "pół bloku" (np. 5 jednostek) dla ścian
        recoil_distance = 5.0
        candidates = (recoil_distance, recoil_distance * 0.5, 1.0, 0.1)
    else:
        # Oryginalna logika "miękkiego" cofnięcia dla drzew i kolców
        base_push = max(0.1, min(dist * 0.6, min(size) * 0.6))
        candidates = [base_push * m for m in (1.0, 1.5, 2.0, 3.0)]

    for bounce in candidates:
        candidate = Position(
            old_pos.x - nx * bounce,
            old_pos.y - ny * bounce
        )
        candidate = _clamp_position_to_map(candidate, size, map_size)
        
        # Sprawdź czy miejsce po odbiciu jest wolne
        if not _candidate_has_collision(tank, candidate, map_size, obstacles):
            return candidate

    return _clamp_position_to_map(old_pos, size, map_size)


def check_tank_obstacle_collision(
    tank: TankUnion,
    obstacles: List[ObstacleUnion]
) -> Optional[ObstacleUnion]:
    """Sprawdza kolizję czołgu z przeszkodami."""
    for obstacle in obstacles:
        if not obstacle.is_alive:
            continue
        obstacle_position = getattr(obstacle, "position", getattr(obstacle, "_position", None))
        obstacle_size = getattr(obstacle, "size", getattr(obstacle, "_size", [0, 0]))
        if obstacle_position and rectangles_overlap(tank.position, get_tank_size(tank), obstacle_position, obstacle_size):
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
        tank1.position, get_tank_size(tank1),
        tank2.position, get_tank_size(tank2)
    )


# ============================================================
# SYSTEM STRZAŁÓW I RELOADU
# ============================================================

def update_reload(tank: TankUnion, delta_time: float) -> None:
    tank.update_reload(delta_time)


def try_load_ammo(tank: TankUnion, ammo: Optional[AmmoType]) -> None:
    if ammo is None:
        return
    if ammo not in tank.ammo:
        return
    if tank.ammo[ammo].count <= 0:
        return
    tank.ammo_loaded = ammo


def can_fire(tank: TankUnion) -> bool:
    return tank.can_shoot()


def fire_projectile(
    tank: TankUnion,
    all_tanks: List[TankUnion],
    obstacles: List[ObstacleUnion]
) -> Optional[ProjectileHit]:
    """
    Wykonuje strzał z czołgu, znajdując najbliższy trafiony obiekt.
    """
    if not can_fire(tank):
        return None

    ammo = tank.ammo_loaded
    damage = tank.shoot()
    if damage is None:
        return None
    if tank.is_overcharged:
        damage *= 2
        tank.is_overcharged = False

    shoot_direction = normalize_angle(tank.heading + tank.barrel_angle)

    # Zasięg strzału - pobieramy z enum value dict
    ammo_range = ammo.value.get("Range", math.inf) if ammo else math.inf
    closest_hit_distance = ammo_range
    final_hit: Optional[ProjectileHit] = None

    # Sprawdź trafienia w inne czołgi
    for target in all_tanks:
        if target._id == tank._id or not target.is_alive():
            continue

        dist = calculate_distance(tank.position, target.position)
        if dist >= closest_hit_distance:
            continue

        angle_to_target = math.degrees(math.atan2(
            target.position.y - tank.position.y,
            target.position.x - tank.position.x
        ))

        if abs(normalize_angle(angle_to_target - shoot_direction)) <= 5:
            closest_hit_distance = dist
            final_hit = ProjectileHit(
                shooter_id=tank._id,
                hit_tank_id=target._id,
                hit_obstacle_id=None,
                damage_dealt=damage,
                hit_position=target.position
            )

    # Sprawdź trafienia w przeszkody
    for obstacle in obstacles:
        if not obstacle.is_alive:
            continue

        obstacle_pos = getattr(obstacle, "position", getattr(obstacle, "_position", None))
        if obstacle_pos is None:
            continue

        dist = calculate_distance(tank.position, obstacle_pos)
        if dist >= closest_hit_distance:
            continue
        
        angle_to_target = math.degrees(math.atan2(
                obstacle_pos.y - tank.position.y,
                obstacle_pos.x - tank.position.x
            ))

        if abs(normalize_angle(angle_to_target - shoot_direction)) <= 5:
            closest_hit_distance = dist
            final_hit = ProjectileHit(
                shooter_id=tank._id,
                hit_tank_id=None,
                hit_obstacle_id=getattr(obstacle, "id", getattr(obstacle, "_id", None)),
                damage_dealt=damage,
                hit_position=obstacle_pos
            )

    # Jeśli ostatecznie trafiono w przeszkodę, oznacz ją jako zniszczoną (jeśli to możliwe)
    if final_hit and final_hit.hit_obstacle_id:
        for obstacle in obstacles:
            obs_id = getattr(obstacle, "id", getattr(obstacle, "_id", None))
            if obs_id == final_hit.hit_obstacle_id:
                if obstacle.is_destructible:
                    obstacle.is_alive = False
                break

    return final_hit

# ============================================================
# OBRAŻENIA
# ============================================================

def apply_damage(tank: TankUnion, damage: int) -> bool:
    """Zadaje obrażenia czołgowi (najpierw shield, potem HP)."""
    tank.take_damage(damage)
    return not tank.is_alive()


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
            tank.position, get_tank_size(tank), # type: ignore
            powerup._position, powerup._size # type: ignore
        ):
            return powerup
    return None


def apply_powerup(tank: TankUnion, powerup: PowerUpData) -> None:
    """Aplikuje efekt powerupu na czołg."""
    ptype = powerup._powerup_type
    value = powerup.value

    if ptype == PowerUpType.MEDKIT:
        tank.hp = min(tank._max_hp, tank.hp + value)

    elif ptype == PowerUpType.SHIELD:
        tank.shield = min(tank._max_shield, tank.shield + value)

    elif ptype == PowerUpType.OVERCHARGE:
        tank.is_overcharged = True

    else:
        # powerup.ammo_type nie istnieje w strukturze PowerUpData
        ammo_name = ptype.value.get("AmmoType")
        if not ammo_name:
            return
        ammo_type = AmmoType[ammo_name]
        slot = tank.ammo.get(ammo_type)
        if slot is None:
            tank.ammo[ammo_type] = AmmoSlot(_ammo_type=ammo_type, count=0)
            slot = tank.ammo[ammo_type]
        slot.count += value

        max_ammo = getattr(tank, "_max_ammo", {}).get(ammo_type)
        if max_ammo:
            slot.count = min(slot.count, max_ammo)


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
        update_reload(tank, delta_time)

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
                    for target in all_tanks:
                        if target._id == hit.hit_tank_id:
                            if apply_damage(target, hit.damage_dealt):
                                results["destroyed_tanks"].append(target._id)
                if hit.hit_obstacle_id:
                    results["destroyed_obstacles"].append(hit.hit_obstacle_id)

    for tank in all_tanks:
        if tank.hp <= 0:
            continue

        action = actions.get(tank._id)
        if not action or action.move_speed == 0:
            continue

        old_pos = tank.position
        new_pos, _ = move_tank(
            tank, action.move_speed,
            map_info.terrain_list, delta_time
        )

        # Apply movement tentatively, then validate collisions.
        tank.position = new_pos

        # Boundary collision -> rollback (z dodatkowym cofnięciem)
        map_size = getattr(map_info, "size", getattr(map_info, "_size", None))
        if map_size and check_tank_boundary_collision(tank, map_size):
            tank.position = resolve_tank_collision_position(
                tank, old_pos, new_pos, map_size, map_info.obstacle_list
            )
            results["collisions"].append(
                {
                    "type": CollisionType.TANK_BOUNDARY.value,
                    "tank_id": tank._id,
                }
            )
            continue

        # Obstacle collision -> rollback
        hit_obstacle = check_tank_obstacle_collision(tank, map_info.obstacle_list)
        if hit_obstacle is not None:
            # Determine type early for recoil logic
            obstacle_type = getattr(hit_obstacle, "obstacle_type", getattr(hit_obstacle, "_obstacle_type", None))
            use_strong_recoil = (obstacle_type == ObstacleType.WALL)

            # Jeśli czołg już był w kolizji na starej pozycji (np. zespawnował się w ścianie),
            # nie naliczaj obrażeń co tick – obrażenia powinny być "za wejście" w przeszkodę.
            was_colliding_before_move = False
            for obstacle in map_info.obstacle_list:
                if not obstacle.is_alive:
                    continue
                obstacle_position = getattr(obstacle, "position", getattr(obstacle, "_position", None))
                obstacle_size = getattr(obstacle, "size", getattr(obstacle, "_size", [0, 0]))
                if obstacle_position and rectangles_overlap(old_pos, get_tank_size(tank), obstacle_position, obstacle_size):
                    was_colliding_before_move = True
                    break

            tank.position = resolve_tank_collision_position(
                tank, old_pos, new_pos, map_size, map_info.obstacle_list,
                strong_recoil=use_strong_recoil
            )
            
            collision_type = CollisionType.TANK_WALL
            if obstacle_type == ObstacleType.TREE:
                collision_type = CollisionType.TANK_TREE
                # Spec: Zderzenie czołgu z drzewem: -5 HP (drzewo jest niszczone)
                if not was_colliding_before_move:
                    if apply_damage(tank, 5):
                        results["destroyed_tanks"].append(tank._id)
                    if getattr(hit_obstacle, "is_destructible", False):
                        hit_obstacle.is_alive = False
                        results["destroyed_obstacles"].append(getattr(hit_obstacle, "id", getattr(hit_obstacle, "_id", None)))
            elif obstacle_type == ObstacleType.ANTI_TANK_SPIKE:
                collision_type = CollisionType.TANK_SPIKE
            elif obstacle_type == ObstacleType.WALL:
                collision_type = CollisionType.TANK_WALL
                # Spec: Zderzenie czołgu ze ścianą: -10 HP
                if not was_colliding_before_move:
                    if apply_damage(tank, 10):
                        results["destroyed_tanks"].append(tank._id)
            else:
                # Inne przeszkody: -5 HP
                if not was_colliding_before_move:
                    if apply_damage(tank, 5):
                        results["destroyed_tanks"].append(tank._id)

            results["collisions"].append(
                {
                    "type": collision_type.value,
                    "tank_id": tank._id,
                    "obstacle_id": getattr(hit_obstacle, "id", getattr(hit_obstacle, "_id", None)),
                }
            )
            continue

        # Tank-tank collision -> rollback (simple resolution)
        collided_with: Optional[str] = None
        for other in all_tanks:
            if other._id == tank._id or other.hp <= 0:
                continue
            if check_tank_tank_collision(tank, other):
                collided_with = other._id
                break

        if collided_with is not None:
            tank.position = old_pos
            results["collisions"].append(
                {
                    "type": CollisionType.TANK_TANK_MOVING.value,
                    "tank_id": tank._id,
                    "other_tank_id": collided_with,
                }
            )
            continue

    # Terrain damage (per tick) for all alive tanks based on final position.
    for tank in all_tanks:
        if tank.hp <= 0:
            continue
        dmg = _terrain_damage_at_position(tank.position, map_info.terrain_list)
        dmg *= 0.05
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