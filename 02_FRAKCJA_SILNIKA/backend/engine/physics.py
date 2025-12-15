"""
Fizyka gry - Ruch, kolizje, strzały, interakcje z otoczeniem
"""

import math
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

from _01_DOKUMENTACJA.final_api import (
    Tank, TankUnion, Position, MapInfo,
    ObstacleUnion, TerrainUnion, PowerUpData,
    AmmoType, PowerUpType, ActionCommand, Tree
)


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
    obstacle_destroyed: Optional[str] = None  # ID zniszczonej przeszkody


@dataclass
class ProjectileHit:
    """Informacja o trafieniu pocisku."""
    hit_tank_id: Optional[str] = None
    hit_obstacle_id: Optional[str] = None
    damage_dealt: int = 0
    hit_position: Optional[Position] = None


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
    
    # Granice prostokąta 1
    left1 = pos1.x - half_w1
    right1 = pos1.x + half_w1
    top1 = pos1.y - half_h1
    bottom1 = pos1.y + half_h1
    
    # Granice prostokąta 2
    left2 = pos2.x - half_w2
    right2 = pos2.x + half_w2
    top2 = pos2.y - half_h2
    bottom2 = pos2.y + half_h2
    
    # Sprawdź nakładanie
    return not (right1 <= left2 or left1 >= right2 or 
                bottom1 <= top2 or top1 >= bottom2)


# SYSTEM RUCHU CZOŁGÓW

def get_terrain_at_position(position: Position, terrains: List[TerrainUnion]) -> Optional[TerrainUnion]: 
    """
    Znajduje teren na danej pozycji.
    Zwraca pierwszy teren, którego bounding box zawiera pozycję.
    """
    for terrain in terrains:
        if rectangles_overlap(position, [1, 1], terrain.position, terrain.size):
            return terrain
    return None


def rotate_heading(tank: TankUnion, target_heading: float, delta_time: float) -> None:
    """
    Obraca kadłub czołgu w kierunku docelowego kąta.
    
    Args:
        tank: Czołg do obrócenia
        target_heading:  Docelowy kąt kadłuba (stopnie)
        delta_time:  Czas delta (sekundy)
    """
    max_rotation = tank._heading_spin_rate * delta_time
    
    # Oblicz różnicę kątową
    angle_diff = normalize_angle(target_heading - tank.heading)
    
    # Ogranicz rotację do maksymalnej prędkości
    if abs(angle_diff) <= max_rotation:
        tank.heading = target_heading
    else:
        tank.heading += max_rotation if angle_diff > 0 else -max_rotation
    
    tank.heading = normalize_angle(tank.heading)


def rotate_barrel(tank: TankUnion, target_barrel_angle: float, delta_time: float) -> None:
    """
    Obraca lufę czołgu względem kadłuba.
    
    Args:
        tank:  Czołg
        target_barrel_angle:  Docelowy kąt lufy względem kadłuba (stopnie)
        delta_time: Czas delta (sekundy)
    """
    max_rotation = tank._barrel_spin_rate * delta_time
    
    angle_diff = normalize_angle(target_barrel_angle - tank.barrel_angle)
    
    if abs(angle_diff) <= max_rotation:
        tank.barrel_angle = target_barrel_angle
    else:
        tank.barrel_angle += max_rotation if angle_diff > 0 else -max_rotation
    
    tank.barrel_angle = normalize_angle(tank.barrel_angle)


def move_tank(
    tank: TankUnion,
    desired_speed: float,
    terrains: List[TerrainUnion],
    delta_time:  float
) -> Tuple[Position, int]:
    """
    Przesuwa czołg zgodnie z jego prędkością i modyfikatorami terenu.
    
    Args:
        tank: Czołg do przesunięcia
        desired_speed:  Pożądana prędkość (może być ujemna dla cofania)
        terrains: Lista terenów na mapie
        delta_time:  Czas delta (sekundy)
    
    Returns:
        Tuple (nowa_pozycja, obrażenia_od_terenu)
    """
    # Ogranicz prędkość do limitów czołgu
    speed = max(-tank._top_speed, min(desired_speed, tank._top_speed))
    
    # Znajdź teren na obecnej pozycji
    current_terrain = get_terrain_at_position(tank.position, terrains)
    terrain_modifier = current_terrain.movement_speed_modifier if current_terrain else 1.0
    terrain_damage = current_terrain.deal_damage if current_terrain else 0
    
    # Oblicz efektywną prędkość
    effective_speed = speed * terrain_modifier
    
    # Oblicz nową pozycję
    heading_rad = math.radians(tank.heading)
    new_x = tank.position.x + math.cos(heading_rad) * effective_speed * delta_time
    new_y = tank.position.y + math.sin(heading_rad) * effective_speed * delta_time
    
    return Position(new_x, new_y), terrain_damage


# SYSTEM KOLIZJI

def check_tank_boundary_collision(tank: TankUnion, map_size: List[int]) -> bool:
    """Sprawdza, czy czołg wychodzi poza granice mapy."""
    half_w, half_h = tank.size[0] / 2.0, tank.size[1] / 2.0
    
    return (tank.position.x - half_w < 0 or
            tank.position.x + half_w > map_size[0] or
            tank.position.y - half_h < 0 or
            tank.position.y + half_h > map_size[1])


def check_tank_obstacle_collision(
    tank: TankUnion,
    obstacles: List[ObstacleUnion]
) -> Tuple[bool, Optional[ObstacleUnion]]: 
    """
    Sprawdza kolizję czołgu z przeszkodami.
    
    Returns:
        Tuple (czy_kolizja, przeszkoda_z_kolizji)
    """
    for obstacle in obstacles:
        if not obstacle.is_alive:
            continue
        
        if rectangles_overlap(tank.position, tank.size, obstacle.position, obstacle.size):
            return True, obstacle
    
    return False, None


def check_tank_tank_collision(
    tank1: TankUnion,
    tank2: TankUnion
) -> bool:
    """Sprawdza kolizję między dwoma czołgami."""
    if tank1._id == tank2._id:
        return False
    
    return rectangles_overlap(tank1.position, tank1.size, tank2.position, tank2.size)


def resolve_collision(
    tank:  TankUnion,
    previous_position: Position,
    collision_type: CollisionType,
    other_tank: Optional[TankUnion] = None,
    obstacle: Optional[ObstacleUnion] = None
) -> CollisionResult:
    """
    Rozwiązuje kolizję i zwraca informacje o obrażeniach.
    
    Args:
        tank:  Czołg, który ma kolizję
        previous_position: Poprzednia pozycja czołgu (przed ruchem)
        collision_type: Typ kolizji
        other_tank: Drugi czołg (jeśli kolizja tank-tank)
        obstacle: Przeszkoda (jeśli kolizja tank-obstacle)
    
    Returns:
        CollisionResult z informacjami o obrażeniach
    """
    result = CollisionResult(has_collision=True, collision_type=collision_type)
    
    # Cofnij pozycję czołgu
    tank.position = previous_position
    
    if collision_type == CollisionType.TANK_TANK_MOVING: 
        # Oba czołgi się poruszały
        result.damage_to_tank1 = 25
        result.damage_to_tank2 = 25
        if other_tank:
            other_tank.position = previous_position  # Cofnij też drugi czołg
    
    elif collision_type == CollisionType.TANK_TANK_STATIC:
        # Jeden czołg stał w miejscu
        result.damage_to_tank1 = 10  # Poruszający się
        result.damage_to_tank2 = 25  # Stojący
    
    elif collision_type == CollisionType.TANK_WALL:
        result.damage_to_tank1 = 25
    
    elif collision_type == CollisionType.TANK_TREE:
        result.damage_to_tank1 = 10
        if obstacle:
            result.obstacle_destroyed = obstacle.id
    
    elif collision_type == CollisionType.TANK_SPIKE:
        # AntiTankSpike - nie cofa pozycji, ale nie zadaje obrażeń przy kolizji
        tank.position = previous_position
    
    elif collision_type == CollisionType.TANK_BOUNDARY:
        result.damage_to_tank1 = 25
    
    return result


def check_all_collisions(
    tank: TankUnion,
    previous_position: Position,
    all_tanks: List[TankUnion],
    obstacles: List[ObstacleUnion],
    map_size: List[int],
    moving_tanks: set  # Set ID czołgów, które się poruszały w tej turze
) -> CollisionResult:
    """
    Sprawdza wszystkie możliwe kolizje dla czołgu.
    
    Args:
        tank: Czołg do sprawdzenia
        previous_position: Poprzednia pozycja czołgu
        all_tanks:  Wszystkie czołgi na mapie
        obstacles: Wszystkie przeszkody
        map_size: Rozmiar mapy [width, height]
        moving_tanks: Zbiór ID czołgów, które się poruszały
    
    Returns:
        CollisionResult lub None jeśli brak kolizji
    """
    # 1. Sprawdź granice mapy
    if check_tank_boundary_collision(tank, map_size):
        return resolve_collision(tank, previous_position, CollisionType.TANK_BOUNDARY)
    
    # 2. Sprawdź kolizje z innymi czołgami
    for other_tank in all_tanks:
        if check_tank_tank_collision(tank, other_tank):
            # Sprawdź, czy oba czołgi się poruszały
            if tank._id in moving_tanks and other_tank._id in moving_tanks:
                collision_type = CollisionType.TANK_TANK_MOVING
            else:
                collision_type = CollisionType.TANK_TANK_STATIC
            
            return resolve_collision(
                tank, previous_position, collision_type, other_tank=other_tank
            )
    
    # 3. Sprawdź kolizje z przeszkodami
    has_collision, obstacle = check_tank_obstacle_collision(tank, obstacles)
    if has_collision and obstacle:
        # Określ typ kolizji na podstawie typu przeszkody
        if obstacle.obstacle_type.name == "WALL":
            collision_type = CollisionType.TANK_WALL
        elif obstacle.obstacle_type.name == "TREE": 
            collision_type = CollisionType.TANK_TREE
        else:  # ANTI_TANK_SPIKE
            collision_type = CollisionType.TANK_SPIKE
        
        return resolve_collision(
            tank, previous_position, collision_type, obstacle=obstacle
        )
    
    # Brak kolizji
    return CollisionResult(has_collision=False, collision_type=CollisionType.NONE)


# SYSTEM STRZAŁÓW

def can_fire(tank: TankUnion, last_shot_time: Dict[str, float], current_time: float) -> bool:
    """
    Sprawdza, czy czołg może wystrzelić (czy minął czas przeładowania).
    
    Args:
        tank: Czołg
        last_shot_time:  Słownik {tank_id: czas_ostatniego_strzału}
        current_time: Obecny czas gry
    
    Returns: 
        True jeśli czołg może strzelać
    """
    if not tank.ammo_loaded:
        return False
    
    if tank.ammo[tank.ammo_loaded].count <= 0:
        return False
    
    reload_time = tank.ammo_loaded.value['ReloadTime']
    last_shot = last_shot_time.get(tank._id, -999)
    
    return (current_time - last_shot) >= reload_time


def fire_projectile(
    tank: TankUnion,
    all_tanks: List[TankUnion],
    obstacles: List[ObstacleUnion],
    current_time: float,
    last_shot_time: Dict[str, float]
) -> Optional[ProjectileHit]:
    """
    Wykonuje strzał z czołgu.
    
    Args:
        tank: Czołg strzelający
        all_tanks: Wszystkie czołgi na mapie
        obstacles: Wszystkie przeszkody
        current_time: Obecny czas gry
        last_shot_time: Słownik czasów ostatnich strzałów
    
    Returns:
        ProjectileHit z informacjami o trafieniu lub None jeśli nie trafiono
    """
    if not can_fire(tank, last_shot_time, current_time):
        return None
    
    ammo_type = tank.ammo_loaded
    ammo_range = ammo_type.value['Range']
    base_damage = abs(ammo_type.value['Value'])
    
    # Uwzględnij Overcharge
    if tank.is_overcharged:
        base_damage *= 2
        tank.is_overcharged = False
    
    # Zmniejsz ilość amunicji
    tank.ammo[ammo_type].count -= 1
    last_shot_time[tank._id] = current_time
    
    # Oblicz kierunek strzału
    shoot_direction = tank.heading + tank.barrel_angle
    shoot_rad = math.radians(shoot_direction)
    
    # Ray-casting - szukaj trafienia
    closest_hit_distance = float('inf')
    hit_result = None
    
    # 1. Sprawdź trafienia w czołgi
    for target_tank in all_tanks: 
        if target_tank._id == tank._id:
            continue
        
        distance = calculate_distance(tank.position, target_tank.position)
        
        # Poza zasięgiem
        if distance > ammo_range:
            continue
        
        # Sprawdź, czy cel jest w linii strzału (uproszczone)
        angle_to_target = math.degrees(math.atan2(
            target_tank.position.y - tank.position.y,
            target_tank.position.x - tank.position.x
        ))
        angle_diff = abs(normalize_angle(angle_to_target - shoot_direction))
        
        # Jeśli cel jest w stożku strzału (tolerancja 5°)
        if angle_diff <= 5 and distance < closest_hit_distance:
            closest_hit_distance = distance
            hit_result = ProjectileHit(
                hit_tank_id=target_tank._id,
                damage_dealt=base_damage,
                hit_position=target_tank.position
            )
    
    # 2. Sprawdź trafienia w przeszkody (mogą blokować strzał)
    for obstacle in obstacles: 
        if not obstacle.is_alive:
            continue
        
        distance = calculate_distance(tank.position, obstacle.position)
        
        if distance > ammo_range: 
            continue
        
        angle_to_obstacle = math.degrees(math.atan2(
            obstacle.position.y - tank.position.y,
            obstacle.position.x - tank.position.x
        ))
        angle_diff = abs(normalize_angle(angle_to_obstacle - shoot_direction))
        
        # Przeszkoda blokuje strzał jeśli jest bliżej niż trafiony czołg
        if angle_diff <= 5 and distance < closest_hit_distance:
            if obstacle.is_destructible:  # Tree
                return ProjectileHit(
                    hit_obstacle_id=obstacle.id,
                    damage_dealt=0,
                    hit_position=obstacle.position
                )
            else:  # Wall - zatrzymuje pocisk
                return ProjectileHit(hit_position=obstacle.position)
    
    return hit_result


def apply_damage(tank: TankUnion, damage: int) -> bool:
    """
    Zadaje obrażenia czołgowi (najpierw shield, potem HP).
    
    Args:
        tank: Czołg otrzymujący obrażenia
        damage: Ilość obrażeń
    
    Returns:
        True jeśli czołg został zniszczony (HP <= 0)
    """
    remaining_damage = damage
    
    # Najpierw odejmij od tarczy
    if tank.shield > 0:
        shield_absorb = min(tank.shield, remaining_damage)
        tank.shield -= shield_absorb
        remaining_damage -= shield_absorb
    
    # Reszta od HP
    tank.hp -= remaining_damage
    
    return tank.hp <= 0


# SYSTEM POWERUPÓW

def check_powerup_pickup(
    tank: TankUnion,
    powerups: List[PowerUpData]
) -> Optional[PowerUpData]:
    """
    Sprawdza, czy czołg jest na powerupie i może go podnieść.
    
    Args:
        tank: Czołg
        powerups: Lista powerupów na mapie
    
    Returns: 
        PowerUpData jeśli podniesiony, None w przeciwnym razie
    """
    for powerup in powerups: 
        if rectangles_overlap(tank.position, tank.size, powerup.position, powerup.size):
            return powerup
    return None


def apply_powerup(tank: TankUnion, powerup: PowerUpData) -> None:
    """
    Aplikuje efekt powerupu na czołg.
    
    Args:
        tank:  Czołg otrzymujący powerup
        powerup: Powerup do zaaplikowania
    """
    ptype = powerup.powerup_type
    
    if ptype == PowerUpType.MEDKIT:
        tank.hp = min(tank.hp + powerup.value, tank._max_hp)
    
    elif ptype == PowerUpType.SHIELD:
        tank.shield = min(tank.shield + powerup.value, tank._max_shield)
    
    elif ptype == PowerUpType.OVERCHARGE:
        tank.is_overcharged = True
    
    elif ptype in [PowerUpType.AMMO_HEAVY, PowerUpType.AMMO_LIGHT, PowerUpType.AMMO_LONG_DISTANCE]:
        # Określ typ amunicji
        ammo_type_name = ptype.value.get('AmmoType')
        if ammo_type_name == 'HEAVY':
            ammo_type = AmmoType.HEAVY
        elif ammo_type_name == 'LIGHT':
            ammo_type = AmmoType.LIGHT
        else: 
            ammo_type = AmmoType.LONG_DISTANCE
        
        # Dodaj amunicję (max to _max_ammo)
        current = tank.ammo[ammo_type].count
        max_ammo = tank._max_ammo[ammo_type]
        tank.ammo[ammo_type].count = min(current + powerup.value, max_ammo)


# FUNKCJA GŁÓWNA - PRZETWARZANIE TURY

def process_physics_tick(
    all_tanks: List[TankUnion],
    actions: Dict[str, ActionCommand],
    map_info: MapInfo,
    delta_time: float,
    current_time: float,
    last_shot_times: Dict[str, float]
) -> Dict[str, any]:
    """
    Przetwarza jedną turę fizyki gry.
    
    Args:
        all_tanks: Lista wszystkich czołgów
        actions: Słownik {tank_id: ActionCommand}
        map_info: Informacje o mapie
        delta_time:  Czas delta (sekundy)
        current_time: Obecny czas gry
        last_shot_times:  Słownik czasów ostatnich strzałów
    
    Returns:
        Słownik z wynikami tury (kolizje, trafienia, podniesione powerupy, etc.)
    """
    results = {
        'collisions': [],
        'projectile_hits': [],
        'picked_powerups': [],
        'destroyed_tanks': [],
        'destroyed_obstacles': []
    }
    
    # Śledź czołgi, które się poruszały
    moving_tanks = set()
    previous_positions = {}
    
    # 1. FAZA:  Rotacje kadłubów i luf
    for tank in all_tanks:
        if tank.hp <= 0:
            continue
        
        action = actions.get(tank._id)
        if not action:
            continue
        
        rotate_heading(tank, action.heading_rotation_angle, delta_time)
        rotate_barrel(tank, action.barrel_rotation_angle, delta_time)
    
    # 2. FAZA: Strzały
    for tank in all_tanks: 
        if tank.hp <= 0:
            continue
        
        action = actions.get(tank._id)
        if not action or not action.should_fire:
            continue
        
        hit = fire_projectile(tank, all_tanks, map_info.obstacle_list, current_time, last_shot_times)
        if hit:
            results['projectile_hits'].append(hit)
            
            # Aplikuj obrażenia
            if hit.hit_tank_id:
                target_tank = next((t for t in all_tanks if t._id == hit.hit_tank_id), None)
                if target_tank:
                    is_destroyed = apply_damage(target_tank, hit.damage_dealt)
                    if is_destroyed:
                        results['destroyed_tanks'].append(target_tank._id)
            
            # Zniszcz przeszkodę jeśli trafiono
            if hit.hit_obstacle_id:
                obstacle = next((o for o in map_info.obstacle_list if o.id == hit.hit_obstacle_id), None)
                if obstacle:
                    obstacle.is_alive = False
                    results['destroyed_obstacles'].append(obstacle.id)
    
    # 3. FAZA: Ruch czołgów
    for tank in all_tanks:
        if tank.hp <= 0:
            continue
        
        action = actions.get(tank._id)
        if not action or action.move_speed == 0:
            continue
        
        previous_positions[tank._id] = Position(tank.position.x, tank.position.y)
        moving_tanks.add(tank._id)
        
        # Przesuń czołg
        new_position, terrain_damage = move_tank(
            tank, action.move_speed, map_info.terrain_list, delta_time
        )
        tank.position = new_position
        
        # Aplikuj obrażenia od terenu
        if terrain_damage > 0:
            is_destroyed = apply_damage(tank, terrain_damage)
            if is_destroyed:
                results['destroyed_tanks'].append(tank._id)
    
    # 4. FAZA: Sprawdzenie kolizji
    for tank in all_tanks:
        if tank.hp <= 0:
            continue
        
        if tank._id not in moving_tanks:
            continue
        
        collision = check_all_collisions(
            tank,
            previous_positions[tank._id],
            all_tanks,
            map_info.obstacle_list,
            map_info.size,
            moving_tanks
        )
        
        if collision.has_collision:
            results['collisions'].append(collision)
            
            # Aplikuj obrażenia od kolizji
            if collision.damage_to_tank1 > 0:
                is_destroyed = apply_damage(tank, collision.damage_to_tank1)
                if is_destroyed:
                    results['destroyed_tanks'].append(tank._id)
            
            # Zniszcz przeszkodę jeśli kolizja z drzewem
            if collision.obstacle_destroyed: 
                obstacle = next((o for o in map_info.obstacle_list if o.id == collision.obstacle_destroyed), None)
                if obstacle:
                    obstacle.is_alive = False
                    results['destroyed_obstacles'].append(collision.obstacle_destroyed)
    
    # 5. FAZA: Zbieranie powerupów
    for tank in all_tanks:
        if tank.hp <= 0:
            continue
        
        powerup = check_powerup_pickup(tank, map_info.powerup_list)
        if powerup:
            apply_powerup(tank, powerup)
            map_info.powerup_list.remove(powerup)
            results['picked_powerups'].append({'tank_id': tank._id, 'powerup': powerup})
    
    return results