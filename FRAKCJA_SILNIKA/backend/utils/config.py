"""Game Configuration Module - Refactored to use existing structures"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Any

# Import existing structures
from ..structures.ammo import AmmoType


class TankType(Enum):
    """Typy czołgów z różnymi statystykami."""

    LIGHT = "light"
    HEAVY = "heavy"
    SNIPER = "sniper"


class TerrainType(Enum):
    """Typy terenu z modyfikatorami prędkości i obrażeń."""

    GRASS = {"speed_modifier": 1.0, "damage_per_tick": 0}
    ROAD = {"speed_modifier": 1.5, "damage_per_tick": 0}
    SWAMP = {"speed_modifier": 0.5, "damage_per_tick": 0}
    POTHOLE_ROAD = {"speed_modifier": 0.8, "damage_per_tick": -1}
    WATER = {"speed_modifier": 0.3, "damage_per_tick": -5}


class ObstacleType(Enum):
    """Typy przeszkód."""

    WALL = {"destructible": False, "blocks_vision": True, "blocks_shooting": True}
    TREE = {"destructible": True, "blocks_vision": True, "blocks_shooting": True}
    ANTI_TANK_SPIKE = {
        "destructible": False,
        "blocks_vision": False,
        "blocks_shooting": False,
    }


class PowerUpType(Enum):
    """Typy power-upów."""

    MEDKIT = {"heal_hp": 50}
    SHIELD = {"shield_boost": 20}
    OVERCHARGE = {"double_damage": True}
    AMMO_BOX_HEAVY = {"ammo_type": AmmoType.HEAVY, "refill_amount": 10}
    AMMO_BOX_LIGHT = {"ammo_type": AmmoType.LIGHT, "refill_amount": 20}
    AMMO_BOX_LONG_DISTANCE = {"ammo_type": AmmoType.LONG_DISTANCE, "refill_amount": 5}


@dataclass
class MapConfig:
    """Konfiguracja mapy."""

    width: int = 200
    height: int = 200
    available_layouts: int = 3
    obstacle_size: Tuple[int, int] = (10, 10)
    terrain_size: Tuple[int, int] = (10, 10)
    powerup_size: Tuple[int, int] = (2, 2)


@dataclass
class TankConfig:

    size: Tuple[int, int] = (5, 5)
    team_size: int = 5
    team_count: int = 2

    tank_stats: Dict[TankType, Dict[str, Any]] = field(
        default_factory=lambda: {
            TankType.LIGHT: {
                "max_hp": 80,
                "max_shield": 30,
                "top_speed": 5.0,
                "vision_range": 70.0,
                "vision_angle": 40.0,
                "barrel_spin_rate": 90.0,
                "heading_spin_rate": 70.0,
                "base_ammo": {
                    AmmoType.HEAVY: 1,
                    AmmoType.LIGHT: 15,
                    AmmoType.LONG_DISTANCE: 2,
                },
            },
            TankType.HEAVY: {
                "max_hp": 120,
                "max_shield": 80,
                "top_speed": 1.0,
                "vision_range": 40.0,
                "vision_angle": 60.0,
                "barrel_spin_rate": 70.0,
                "heading_spin_rate": 30.0,
                "base_ammo": {
                    AmmoType.HEAVY: 5,
                    AmmoType.LIGHT: 10,
                    AmmoType.LONG_DISTANCE: 2,
                },
            },
            TankType.SNIPER: {
                "max_hp": 40,
                "max_shield": 30,
                "top_speed": 3.0,
                "vision_range": 120.0,
                "vision_angle": 20.0,
                "barrel_spin_rate": 100.0,
                "heading_spin_rate": 45.0,
                "base_ammo": {
                    AmmoType.HEAVY: 1,
                    AmmoType.LIGHT: 5,
                    AmmoType.LONG_DISTANCE: 10,
                },
            },
        }
    )


@dataclass
class PowerUpConfig:
    """Konfiguracja power-upów."""

    spawn_start_tick: int = 50
    spawn_interval: int = 20
    max_powerups_on_map: int = 25
    min_distance_between_powerups: float = 20.0
    despawn_time: int = 1000  # ticki po których power-up znika jeśli nie został zebrany


@dataclass
class PhysicsConfig:
    """Konfiguracja fizyki i kolizji."""

    collision_damage: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            "tank_vs_stationary_tank": {"moving": -25, "stationary": -10},
            "tank_vs_moving_tank": {"both": -25},
            "tank_vs_tree": {"tank": -10},
            "tank_vs_wall": {"tank": -25},
        }
    )


@dataclass
class GameRules:
    """Zasady gry."""

    friendly_fire: bool = True
    sudden_death_tick: int = 100000
    sudden_death_damage_per_tick: int = -1
    win_condition: str = "team_annihilation"

    # Spawning rules
    ally_min_distance: float = 5.0
    enemy_min_distance: float = 50.0


@dataclass
class CoordinateSystem:
    """System współrzędnych zgodny z API."""

    origin: str = "bottom_left"  # (0,0) w lewym dolnym rogu
    x_direction: str = "right"  # X rośnie w prawo
    y_direction: str = "up"  # Y rośnie w górę
    angle_zero: str = "north"  # 0° to kierunek w górę (Północ)
    angle_direction: str = "clockwise"  # Kąty rosną zgodnie z ruchem wskazówek zegara
    position_reference: str = "center"  # Pozycja obiektów to ich środek geometryczny


@dataclass
class GameConfig:
    """Główna konfiguracja gry."""

    map_config: MapConfig = field(default_factory=MapConfig)
    tank_config: TankConfig = field(default_factory=TankConfig)
    powerup_config: PowerUpConfig = field(default_factory=PowerUpConfig)
    physics_config: PhysicsConfig = field(default_factory=PhysicsConfig)
    game_rules: GameRules = field(default_factory=GameRules)
    coordinate_system: CoordinateSystem = field(default_factory=CoordinateSystem)

    # Debugging and logging
    debug_mode: bool = False
    log_level: str = "INFO"
    save_game_replay: bool = True

    def validate_config(self) -> bool:
        """Walidacja konfiguracji gry."""
        if self.tank_config.team_size * self.tank_config.team_count > 20:
            raise ValueError("Zbyt dużo czołgów na mapie!")

        if self.map_config.width <= 0 or self.map_config.height <= 0:
            raise ValueError("Nieprawidłowy rozmiar mapy!")

        if self.powerup_config.spawn_interval <= 0:
            raise ValueError("Nieprawidłowy interwał spawnu power-upów!")

        return True

    def get_tank_spawn_positions(self) -> List[Tuple[float, float]]:
        """Generuje pozycje spawnu dla czołgów zgodnie z zasadami."""
        positions = []

        # Team 1 spawn positions (lewa strona mapy)
        team1_start_x = 50
        team1_positions = []
        for i in range(self.tank_config.team_size):
            x = team1_start_x + (i * 15)
            y = 50 + (i * 20)
            team1_positions.append((x, y))

        # Team 2 spawn positions (prawa strona mapy)
        team2_start_x = self.map_config.width - 100
        team2_positions = []
        for i in range(self.tank_config.team_size):
            x = team2_start_x + (i * 15)
            y = 50 + (i * 20)
            team2_positions.append((x, y))

        positions.extend(team1_positions)
        positions.extend(team2_positions)

        return positions


# Singleton instance
game_config = GameConfig()


# Helper functions using existing AmmoType structure
def get_ammo_damage(ammo_type: AmmoType) -> int:
    """Pobiera obrażenia dla danego typu amunicji."""
    return ammo_type.value_amount


def get_ammo_range(ammo_type: AmmoType) -> float:
    """Pobiera zasięg dla danego typu amunicji."""
    return ammo_type.range


def get_ammo_reload_time(ammo_type: AmmoType) -> float:
    """Pobiera czas przeładowania dla danego typu amunicji."""
    return ammo_type.reload_time


def get_terrain_speed_modifier(terrain_type: TerrainType) -> float:
    """Pobiera modyfikator prędkości dla danego typu terenu."""
    return terrain_type.value["speed_modifier"]


def get_terrain_damage(terrain_type: TerrainType) -> int:
    """Pobiera obrażenia na tick dla danego typu terenu."""
    return terrain_type.value["damage_per_tick"]


def get_tank_stats(tank_type: TankType) -> Dict:
    """Pobiera statystyki dla danego typu czołgu."""
    return game_config.tank_config.tank_stats[tank_type]


def is_obstacle_destructible(obstacle_type: ObstacleType) -> bool:
    """Sprawdza czy przeszkoda jest niszczalna."""
    return obstacle_type.value["destructible"]


def does_obstacle_block_vision(obstacle_type: ObstacleType) -> bool:
    """Sprawdza czy przeszkoda blokuje widok."""
    return obstacle_type.value["blocks_vision"]


def does_obstacle_block_shooting(obstacle_type: ObstacleType) -> bool:
    """Sprawdza czy przeszkoda blokuje strzały."""
    return obstacle_type.value["blocks_shooting"]
