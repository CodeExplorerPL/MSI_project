from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Literal, Any, Union
from enum import Enum
from dataclasses import dataclass, field


# ==============================================================================
# 1. STRUKTURY DANYCH (INPUT DLA AGENTA)
# ==============================================================================

@dataclass
class Position:
    """Reprezentuje pozycję X, Y na mapie."""
    x: float
    y: float


class PowerUpType(Enum):
    """Definicja typów PowerUp'ów i ich wartości."""
    MEDKIT = {"Name": "Medkit", "Value": 50}
    SHIELD = {"Name": "Shield", "Value": 20}
    OVERCHARGE = {"Name": "Overcharge", "Value": 2}
    AMMO_HEAVY = {"Name": "HeavyAmmo", "Value": 2, "AmmoType": "HEAVY"}
    AMMO_LIGHT = {"Name": "LightAmmo", "Value": 5, "AmmoType": "LIGHT"}
    AMMO_LONG_DISTANCE = {"Name": "LongDistanceAmmo", "Value": 2,
                          "AmmoType": "LONG_DISTANCE"}


class AmmoType(Enum):
    """Definicja typów amunicji i ich bazowych właściwości."""
    HEAVY = {"Value": -40, "Range": 5, "ReloadTime": 2}
    LIGHT = {"Value": -20, "Range": 10, "ReloadTime": 1}
    LONG_DISTANCE = {"Value": -25, "Range": 20, "ReloadTime": 2}


@dataclass
class AmmoSlot:
    """Informacje o danym typie amunicji w ekwipunku czołgu."""
    ammo_type: AmmoType
    count: int


@dataclass
class PowerUpData:
    """Informacje o przedmiocie do zebrania (np. Apteczka, Amunicja)."""
    position: Position
    powerup_type: PowerUpType
    size: List[int] = field(default_factory=lambda: [10, 10])

    @property
    def value(self) -> int: return self.powerup_type.value['Value']

    @property
    def name(self) -> str: return self.powerup_type.value['Name']


# --- PRZESZKODY (Obstacles) ---
class ObstacleType(Enum):
    """Definicja typów przeszkód i ich kluczowych właściwości."""
    WALL = {"destructible": False, "see_through": False}
    TREE = {"destructible": True, "see_through": False}
    ANTI_TANK_SPIKE = {"destructible": False, "see_through": True}


@dataclass
class Obstacle(ABC):
    """Abstrakcyjna klasa bazowa dla przeszkód."""
    id: str
    position: Position
    size: List[int] = field(default_factory=lambda: [10, 10])
    is_alive: bool = True

    obstacle_type: ObstacleType = field(init=False)

    @property
    def is_destructible(self) -> bool: return self.obstacle_type.value[
        'destructible']

    @property
    def is_see_through(self) -> bool: return self.obstacle_type.value[
        'see_through']


@dataclass
class Wall(Obstacle):
    """Mur: Niezniszczalny, blokuje widok."""
    obstacle_type: ObstacleType = field(default=ObstacleType.WALL, init=False)


@dataclass
class Tree(Obstacle):
    """Drzewo: Zniszczalne jednym trafieniem, blokuje widok."""
    obstacle_type: ObstacleType = field(default=ObstacleType.TREE, init=False)


@dataclass
class AntiTankSpike(Obstacle):
    """Kolce Przeciwpancerne: Niezniszczalne, PRZEZIERNE."""
    obstacle_type: ObstacleType = field(default=ObstacleType.ANTI_TANK_SPIKE,
                                        init=False)


ObstacleUnion = Union[Wall, Tree, AntiTankSpike]


# --- TERENY (Terrains) ---
@dataclass
class Terrain(ABC):
    """Abstrakcyjna klasa bazowa dla typów terenu."""
    id: str
    position: Position
    size: List[int] = field(default_factory=lambda: [10, 10])

    terrain_type: str = field(init=False)
    movement_speed_modifier: float = field(init=False)
    deal_damage: int = field(init=False)


@dataclass
class Grass(Terrain):
    """Trawa: Brak efektu."""
    terrain_type: Literal["Grass"] = field(default="Grass", init=False)
    movement_speed_modifier: float = 1
    deal_damage: int = 0


@dataclass
class Road(Terrain):
    """Droga: Zwiększa prędkość ruchu."""
    terrain_type: Literal["Road"] = field(default="Road", init=False)
    movement_speed_modifier: float = 1.5
    deal_damage: int = 0


@dataclass
class Swamp(Terrain):
    """Bagno: Spowalnia ruch."""
    terrain_type: Literal["Swamp"] = field(default="Swamp", init=False)
    movement_speed_modifier: float = 0.4
    deal_damage: int = 0


@dataclass
class PotholeRoad(Terrain):
    """Droga z Dziurami: Spowalnia i zadaje minimalne obrażenia."""
    terrain_type: Literal["PotholeRoad"] = field(default="PotholeRoad",
                                                 init=False)
    movement_speed_modifier: float = 0.95
    deal_damage: int = 5


@dataclass
class Water(Terrain):
    """Woda: Spowalnia i zadaje obrażenia."""
    terrain_type: Literal["Water"] = field(default="Water", init=False)
    movement_speed_modifier: float = 0.7
    deal_damage: int = 10


TerrainUnion = Union[Grass, Road, Swamp, PotholeRoad, Water]


@dataclass
class SeenTank:
    """Informacje o widocznym wrogu (dla Agentów to ID i pozycja)."""
    id: str
    position: Position
    heading: float
    barrel_angle: float
    distance: float
    team: int


@dataclass
class TankSensorData:
    """Dane wykryte przez systemy sensoryczne czołgu."""
    seen_tanks: List[SeenTank]
    seen_powerups: List[PowerUpData]
    seen_obstacles: List[ObstacleUnion]
    seen_terrains: List[TerrainUnion]


# --- CZOŁGI (Tanks) ---
@dataclass
class Tank(ABC):
    """Abstrakcyjna klasa bazowa dla wszystkich typów czołgów."""
    id: str
    team: int

    # Statystyki bazowe
    tank_type: str = field(init=False)
    vision_angle: float
    vision_range: float
    move_speed: float
    barrel_spin_rate: float
    heading_spin_rate: float

    # Dynamiczne statystyki
    hp: int
    shield: int
    position: Position
    ammo: Dict[AmmoType, AmmoSlot]
    ammo_loaded: Optional[AmmoType] = None
    reload_cooldown: float = 0.0
    barrel_angle: float = 0.0
    heading: float = 0.0
    is_overcharged: bool = False
    size: List[int] = field(default_factory=lambda: [5, 5])

    @abstractmethod
    def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]: pass


@dataclass
class LightTank(Tank):
    tank_type: Literal["LightTank"] = field(default="LightTank", init=False)
    hp: int = 80
    shield: int = 30
    move_speed: float = 10
    vision_range: float = 10
    vision_angle: float = 40
    barrel_spin_rate: float = 180
    heading_spin_rate: float = 140

    def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]:
        return {AmmoType.HEAVY: AmmoSlot(AmmoType.HEAVY, 1),
                AmmoType.LIGHT: AmmoSlot(AmmoType.LIGHT, 10),
                AmmoType.LONG_DISTANCE: AmmoSlot(AmmoType.LONG_DISTANCE, 2)}


@dataclass
class HeavyTank(Tank):
    tank_type: Literal["HeavyTank"] = field(default="HeavyTank", init=False)
    hp: int = 120
    shield: int = 80
    move_speed: float = 2
    vision_range: float = 8
    vision_angle: float = 60
    barrel_spin_rate: float = 140
    heading_spin_rate: float = 60

    def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]:
        return {AmmoType.HEAVY: AmmoSlot(AmmoType.HEAVY, 5),
                AmmoType.LIGHT: AmmoSlot(AmmoType.LIGHT, 15),
                AmmoType.LONG_DISTANCE: AmmoSlot(AmmoType.LONG_DISTANCE, 5)}


@dataclass
class Sniper(Tank):
    tank_type: Literal["Sniper"] = field(default="Sniper", init=False)
    hp: int = 40
    shield: int = 30    
    move_speed: float = 5
    vision_range: float = 25
    vision_angle: float = 20
    barrel_spin_rate: float = 200
    heading_spin_rate: float = 90

    def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]:
        return {AmmoType.HEAVY: AmmoSlot(AmmoType.HEAVY, 1),
                AmmoType.LIGHT: AmmoSlot(AmmoType.LIGHT, 5),
                AmmoType.LONG_DISTANCE: AmmoSlot(AmmoType.LONG_DISTANCE, 10)}


TankUnion = Union[LightTank, HeavyTank, Sniper]


@dataclass
class MapInfo:
    """Informacje o mapie (rozmiar, przeszkody, itp.)."""
    map_seed: str
    obstacle_list: List[ObstacleUnion]
    powerup_list: List[PowerUpData]
    terrain_list: List[TerrainUnion]  # Użycie Unii
    all_tanks: List[TankUnion]
    size: List[int] = field(default_factory=lambda: [500, 500])


# ==============================================================================
# 2. KONTRAKT API (IAgentController)
# ==============================================================================

class IAgentController(ABC):
    """Abstrakcyjny kontroler."""

    @abstractmethod
    def get_action(self, my_tank_status: TankUnion,
                   sensor_data: TankSensorData,
                   delta_time: float) -> 'ActionCommand': pass

    @abstractmethod
    def destroy(self): pass

    @abstractmethod
    def end(self): pass


# ==============================================================================
# 3. KLASA DECYZYJNA (OUTPUT Z AGENTA DO SILNIKA)
# ==============================================================================

@dataclass
class ActionCommand:
    """Pojedynczy obiekt zawierający wszystkie polecenia dla Silnika w danej klatce."""
    barrel_rotation_angle: float
    heading_rotation_angle: float
    should_move: bool = False
    ammo_to_load: Optional[AmmoType] = None
    should_fire: bool = False