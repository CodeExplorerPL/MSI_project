from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Literal, Any, Union
from enum import Enum
from dataclasses import dataclass, field

# ==============================================================================
# 1. STRUKTURY DANYCH (INPUT DLA AGENTA)
# ==============================================================================

# ... (Klasy Position, Direction, PowerUpType, AmmoType, AmmoSlot pozostają bez zmian)

@dataclass
class Position:
    """Reprezentuje pozycję X, Y na mapie."""
    x: float
    y: float

class Direction(Enum):
    """Kierunek kadłuba czołgu lub ruchu (obrót tylko o 90 stopni)."""
    N = 0  # North - 0 stopni
    E = 90 # East - 90 stopni
    S = 180 # South - 180 stopni
    W = 270 # West - 270 stopni

class PowerUpType(Enum):
    """Definicja typów PowerUp'ów i ich wartości."""
    MEDKIT = {"Name": "Medkit", "Value": 50}
    SHIELD = {"Name": "Shield", "Value": 20}
    OVERCHARGE = {"Name": "Overcharge", "Value": 2} # Nastepny atak zadaje podwojne obrazenia
    AMMO = {"Name": "Ammo", "Value": 10}

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


# ... (Klasy PowerUpData, ObstacleData, TerrainData, SeenTank, TankSensorData pozostają bez zmian)
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

@dataclass
class ObstacleData:
    """Informacje o przeszkodzie na mapie."""
    id: str
    position: Position
    size: List[int] = field(default_factory=lambda: [10, 10])
    is_destructible: bool = False
    is_alive: bool = True

@dataclass
class TerrainData:
    """Informacje o typie terenu na danym obszarze."""
    id: str
    position: Position
    size: List[int] = field(default_factory=lambda: [10, 10])
    movement_speed_modifier: float = 1.0
    deal_damage: int = 0

    # Typy terenu (tylko do celów demonstracyjnych, lepszy byłby Enum)
    @property
    def type(self) -> str: return "road" if self.movement_speed_modifier == 1.5 else "default"

@dataclass
class SeenTank:
    """Informacje o widocznym wrogu (dla Agentów to ID i pozycja)."""
    id: str
    position: Position
    heading: Direction
    barrel_angle: float
    distance: float
    team: int

@dataclass
class TankSensorData:
    """Dane wykryte przez systemy sensoryczne czołgu."""
    seen_tanks: List[SeenTank]
    seen_powerups: List[PowerUpData]
    seen_obstacles: List[ObstacleData]
    seen_terrains: List[TerrainData]


### Nowa Struktura Czołgów

@dataclass
class Tank(ABC):
    """
    Abstrakcyjna klasa bazowa dla wszystkich typów czołgów.
    Zawiera statystyki dynamiczne (stan bieżący) i statystyki bazowe (wartości domyślne).
    """
    id: str
    team: int

    # Statystyki bazowe (bazujące na typie czołgu - muszą być zdefiniowane w klasach dziedziczących)
    tank_type: str = field(init=False) # Typ czołgu (np. "LightTank")
    vision_angle: float
    vision_range: float
    move_speed: float               # Maks. dystans ruchu w turze
    size: List[int]
    barrel_spin_rate: float   # Maks. kąt obrotu lufy w turze
    heading_spin_rate: float   # Maks. kąt obrotu kadłuba w turze (w stopniach)

    # Dynamiczne statystyki (zmieniające się podczas gry)
    hp: int
    shield: int
    position: Position
    ammo: Dict[AmmoType, AmmoSlot]
    ammo_loaded: Optional[AmmoType] = None
    reload_cooldown: float = 0.0    # Pozostały czas do przeładowania (w turach/tickach)
    barrel_angle: float = 0         # Kąt lufy (0-360, 0 to N)
    heading: Direction = Direction.N# Kierunek kadłuba
    is_overcharged: bool = False    # Czy zebrał powerup OVERCHARGE

    @abstractmethod
    def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]:
        """Metoda zwracająca startowy zestaw amunicji dla danego typu czołgu."""
        pass


@dataclass
class LightTank(Tank):
    """Specyfikacja czołgu lekkiego."""
    tank_type: Literal["LightTank"] = field(default="LightTank", init=False)
    hp: int = 80
    shield: int = 30
    move_speed: float = 30
    vision_range: float = 10
    vision_angle: float = 40

    def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]:
        return {
            AmmoType.HEAVY: AmmoSlot(AmmoType.HEAVY, 1),
            AmmoType.LIGHT: AmmoSlot(AmmoType.LIGHT, 10),
            AmmoType.LONG_DISTANCE: AmmoSlot(AmmoType.LONG_DISTANCE, 2),
        }

@dataclass
class HeavyTank(Tank):
    """Specyfikacja czołgu ciężkiego."""
    tank_type: Literal["HeavyTank"] = field(default="HeavyTank", init=False)
    hp: int = 120
    shield: int = 80
    move_speed: float = 10
    vision_range: float = 8
    vision_angle: float = 60

    def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]:
        return {
            AmmoType.HEAVY: AmmoSlot(AmmoType.HEAVY, 5),
            AmmoType.LIGHT: AmmoSlot(AmmoType.LIGHT, 15),
            AmmoType.LONG_DISTANCE: AmmoSlot(AmmoType.LONG_DISTANCE, 5),
        }

@dataclass
class Sniper(Tank):
    """Specyfikacja czołgu snajperskiego."""
    tank_type: Literal["Sniper"] = field(default="Sniper", init=False)
    hp: int = 40
    shield: int = 30
    move_speed: float = 20
    vision_range: float = 25
    vision_angle: float = 20

    def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]:
        return {
            AmmoType.HEAVY: AmmoSlot(AmmoType.HEAVY, 1),
            AmmoType.LIGHT: AmmoSlot(AmmoType.LIGHT, 5),
            AmmoType.LONG_DISTANCE: AmmoSlot(AmmoType.LONG_DISTANCE, 10),
        }

# Użyjemy Union do określenia, że lista all_tanks zawiera jeden z konkretnych typów czołgów
TankUnion = Union[LightTank, HeavyTank, Sniper]

@dataclass
class MapInfo:
    """Informacje o mapie (rozmiar, przeszkody, itp.)."""
    map_seed: str
    obstacle_list: List[ObstacleData]
    powerup_list: List[PowerUpData]
    terrain_list: List[TerrainData]
    all_tanks: List[TankUnion]     # Lista WSZYSTKICH czołgów (startowy stan mapy)
    size: List[int] = field(default_factory=lambda: [500, 500])

# ==============================================================================
# 2. KONTRAKT API (IAgentController i ActionCommand pozostają bez zmian)
# ==============================================================================

class IAgentController(ABC):
    """Abstrakcyjny kontroler."""

    @abstractmethod
    def get_action(self,
                   my_tank_status: TankUnion, # Zmieniono typ na Union
                   sensor_data: TankSensorData,
                   delta_time: float
                   ) -> 'ActionCommand':
        pass
    # ... destroy i end pozostają bez zmian

    @abstractmethod
    def destroy(self):
        """Metoda wywoływana przy niszczeniu Agenta."""
        pass

    @abstractmethod
    def end(self):
        """Metoda wywoływana na końcu symulacji."""
        pass

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