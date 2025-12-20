from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Literal, Any, Union
from enum import Enum
from dataclasses import dataclass, field


# ==============================================================================
# 1. STRUKTURY DANYCH (INPUT DLA AGENTA)
# ==============================================================================

# --- POZYCJA (Position) ---
@dataclass
class Position:
    """Reprezentuje pozycję X, Y na mapie."""
    x: float # współrzędna X
    y: float # współrzędna Y

# --- POWER-UPY (Power-Ups) ---
class PowerUpType(Enum):
    """Definicja typów PowerUp'ów i ich wartości."""
    MEDKIT = {"Name": "Medkit", "Value": 50} # przywraca 50 punktów życia
    SHIELD = {"Name": "Shield", "Value": 20} # przywraca 20 punktów osłony
    OVERCHARGE = {"Name": "Overcharge", "Value": 2} # następny strzał zadaje podwójne obrażenia
    AMMO_HEAVY = {"Name": "HeavyAmmo", "Value": 2, "AmmoType": "HEAVY"} # amunicja ciężka - 2 sztuki
    AMMO_LIGHT = {"Name": "LightAmmo", "Value": 5, "AmmoType": "LIGHT"} # amunicja lekka - 5 sztuk
    AMMO_LONG_DISTANCE = {"Name": "LongDistanceAmmo", "Value": 2, "AmmoType": "LONG_DISTANCE"} # amunicja dalekiego zasięgu - 2 sztuki

# --- AMUNICJA (Ammunition) ---
class AmmoType(Enum):
    """Definicja typów amunicji i ich bazowych właściwości."""
    HEAVY = {"Value": -40, "Range": 5, "ReloadTime": 10} # duże obrażenia, krótki zasięg, długi czas przeładowania
    LIGHT = {"Value": -20, "Range": 10, "ReloadTime": 5} # mniejsze obrażenia, średni zasięg, krótki czas przeładowania
    LONG_DISTANCE = {"Value": -25, "Range": 20, "ReloadTime": 10} # średnie obrażenia, długi zasięg, długi czas przeładowania


@dataclass
class AmmoSlot:
    """Informacje o danym typie amunicji w ekwipunku czołgu."""
    _ammo_type: AmmoType # typ amunicji
    count: int # ilość dostępnej amunicji danego typu

# --- POWER-UPY (Power-Ups) ---
@dataclass
class PowerUpData:
    """Informacje o przedmiocie do zebrania (np. Apteczka, Amunicja)."""
    _id: str # unikalne ID power-up'u
    _position: Position # pozycja power-up'u na mapie
    _powerup_type: PowerUpType # typ power-up'u
    _size: List[int] = field(default_factory=lambda: [2, 2]) # rozmiar power-up'u [szerokość, wysokość]

    @property
    def value(self) -> int: return self.powerup_type.value['Value'] # Wartość power-up'u

    @property
    def name(self) -> str: return self.powerup_type.value['Name'] # Nazwa power-up'u


# --- PRZESZKODY (Obstacles) ---
class ObstacleType(Enum):
    """Definicja typów przeszkód i ich kluczowych właściwości."""
    WALL = {"destructible": False, "see_through": False} # Niezniszczalny, blokuje widok, pocisk nie przechodzi
    TREE = {"destructible": True, "see_through": False} # Zniszczalny, blokuje widok, pocisk nie przechodzi
    ANTI_TANK_SPIKE = {"destructible": False, "see_through": True} # Niezniszczalny, umożliwia widok, pocisk nie przechodzi


@dataclass
class Obstacle(ABC):
    """Abstrakcyjna klasa bazowa dla przeszkód."""
    _id: str # unikalne ID przeszkody
    _position: Position # pozycja przeszkody na mapie
    _size: List[int] = field(default_factory=lambda: [10, 10]) # rozmiar przeszkody [szerokość, wysokość]
    _obstacle_type: ObstacleType = field(init=False) # typ przeszkody
    is_alive: bool = True # czy przeszkoda jest nadal obecna na mapie

    @property
    def is_destructible(self) -> bool: return self.obstacle_type.value['destructible'] # czy przeszkoda jest zniszczalna

    @property
    def is_see_through(self) -> bool: return self.obstacle_type.value['see_through'] # czy przeszkoda umożliwia widok


@dataclass
class Wall(Obstacle):
    """Mur: Niezniszczalny, blokuje widok."""
    obstacle_type: ObstacleType = field(default=ObstacleType.WALL, init=False) # typ przeszkody


@dataclass
class Tree(Obstacle):
    """Drzewo: Zniszczalne jednym trafieniem, blokuje widok."""
    obstacle_type: ObstacleType = field(default=ObstacleType.TREE, init=False) # typ przeszkody


@dataclass
class AntiTankSpike(Obstacle):
    """Kolce Przeciwpancerne: Niezniszczalne, PRZEZIERNE."""
    obstacle_type: ObstacleType = field(default=ObstacleType.ANTI_TANK_SPIKE, init=False) # typ przeszkody


ObstacleUnion = Union[Wall, Tree, AntiTankSpike] # Wszystkie typy przeszkód

# --- TERENY (Terrains) ---
@dataclass
class Terrain(ABC):
    """Abstrakcyjna klasa bazowa dla typów terenu."""
    _id: str # unikalne ID terenu
    _position: Position # pozycja terenu na mapie
    _size: List[int] = field(default_factory=lambda: [10, 10]) # rozmiar terenu [szerokość, wysokość]
    
    _terrain_type: str = field(init=False) # typ terenu
    _movement_speed_modifier: float = field(init=False) # modyfikator prędkości ruchu (0.0 = zatrzymany, 1.0 = normalna prędkość)
    _deal_damage: int = field(init=False) # obrażenia zadawane co tick


@dataclass
class Grass(Terrain):
    """Trawa: Brak efektu."""
    _terrain_type: Literal["Grass"] = field(default="Grass", init=False) # typ terenu
    _movement_speed_modifier: float = 1 # modyfikator prędkości ruchu 
    _deal_damage: int = 0 # obrażenia zadawane co tick


@dataclass
class Road(Terrain):
    """Droga: Zwiększa prędkość ruchu."""
    _terrain_type: Literal["Road"] = field(default="Road", init=False) # typ terenu
    _movement_speed_modifier: float = 1.5 # modyfikator prędkości ruchu
    _deal_damage: int = 0 # obrażenia zadawane co tick


@dataclass
class Swamp(Terrain):
    """Bagno: Spowalnia ruch."""
    _terrain_type: Literal["Swamp"] = field(default="Swamp", init=False) # typ terenu
    _movement_speed_modifier: float = 0.4 # modyfikator prędkości ruchu
    _deal_damage: int = 0 # obrażenia zadawane co tick


@dataclass
class PotholeRoad(Terrain):
    """Droga z Dziurami: Spowalnia i zadaje minimalne obrażenia."""
    _terrain_type: Literal["PotholeRoad"] = field(default="PotholeRoad", init=False) # typ terenu
    _movement_speed_modifier: float = 0.95 # modyfikator prędkości ruchu
    _deal_damage: int = 1 # obrażenia zadawane co tick


@dataclass
class Water(Terrain):
    """Woda: Spowalnia i zadaje obrażenia."""
    _terrain_type: Literal["Water"] = field(default="Water", init=False) # typ terenu
    _movement_speed_modifier: float = 0.7 # modyfikator prędkości ruchu
    _deal_damage: int = 2 # obrażenia zadawane co tick


TerrainUnion = Union[Grass, Road, Swamp, PotholeRoad, Water] # Wszystkie typy terenów

# --- DANE SENSORYCZNE CZOŁGU (Tank Sensor Data) ---
@dataclass
class SeenTank:
    """Informacje o widocznym wrogu (dla Agentów to ID i pozycja)."""
    _id: str # unikalne ID wrogiego czołgu
    _team: int # ID drużyny wroga
    _tank_type: str # typ czołgu jako string
    position: Position # pozycja wroga na mapie
    is_damaged: bool # czy wróg jest uszkodzony (hp < 30% max_hp)
    heading: float # kąt kadłuba wroga
    barrel_angle: float # kąt lufy wroga
    distance: float # odległość od naszego czołgu
    

@dataclass
class TankSensorData:
    """Dane wykryte przez systemy sensoryczne czołgu."""
    seen_tanks: List[SeenTank] # Lista widocznych wrogich czołgów
    seen_powerups: List[PowerUpData] # Lista widocznych power-up'ów
    seen_obstacles: List[ObstacleUnion] # Lista widocznych przeszkód
    seen_terrains: List[TerrainUnion] # Lista widocznych terenów


# --- CZOŁGI (Tanks) ---
@dataclass
class Tank(ABC):
    """Abstrakcyjna klasa bazowa dla wszystkich typów czołgów."""

    # Statystyki bazowe
    _id: str # unikalne ID czołgu
    _team: int # ID drużyny czołgu
    _tank_type: str = field(init=False) # typ czołgu jako string
    _vision_angle: float # kąt widzenia w stopniach (-_vision_angle do +_vision_angle)
    _vision_range: float # zasięg widzenia (odległość)
    _top_speed: float # maksymalna prędkość ruchu
    _barrel_spin_rate: float # maksymalna prędkość obrotu lufy (stopnie na tick)
    _heading_spin_rate: float # maksymalna prędkość obrotu kadłuba (stopnie na tick)
    _max_hp: int # maksymalna ilość punktów życia
    _max_shield: int # maksymalna ilość punktów osłony
    _size: List[int] = field(default_factory=lambda: [5, 5]) # rozmiar czołgu [szerokość, wysokość]
    _max_ammo: Dict[AmmoType, int] # maksymalna pojemność amunicji

    # Dynamiczne statystyki
    hp: int # aktualne punkty życia
    shield: int # aktualne punkty osłony
    position: Position # aktualna pozycja na mapie
    move_speed: float # aktualna prędkość ruchu (0 = stojący, dodatnia = do przodu, ujemna = do tyłu, max = _top_speed)
    ammo: Dict[AmmoType, AmmoSlot] # aktualny ekwipunek amunicji
    ammo_loaded: Optional[AmmoType] = None # aktualnie załadowany typ amunicji
    current_reload_progress: int = 0 # <--- 0 = gotowy, >0 = czas do końca przeładowania (w tickach)
    barrel_angle: float = 0.0 # kąt lufy
    heading: float = 0.0 # kąt kadłuba
    is_overcharged: bool = False # czy czołg jest w trybie overcharge (następny strzał zada podwójne obrażenia)

    @abstractmethod
    def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]: pass
    """Zwraca bazowy ekwipunek amunicji dla danego typu czołgu."""


@dataclass
class LightTank(Tank):
    _tank_type: Literal["LightTank"] = field(default="LightTank", init=False)
    hp: int = 80 # punkty życia
    shield: int = 30 # punkty osłony
    _max_hp: int = hp # maksymalna ilość punktów życia
    _max_shield: int = shield # maksymalna ilość punktów osłony
    _top_speed: float = 5 # maksymalna prędkość ruchu
    _vision_range: float = 10 # zasięg widzenia
    _vision_angle: float = 40 # kąt widzenia 
    _barrel_spin_rate: float = 90 # prędkość obrotu lufy/tick
    _heading_spin_rate: float = 70 # prędkość obrotu kadłuba/tick
    _max_ammo: Dict[AmmoType, int] = field( # maksymalna pojemność amunicji
        default_factory=lambda: {
            AmmoType.HEAVY: 1,
            AmmoType.LIGHT: 15,
            AmmoType.LONG_DISTANCE: 2
        }
    )

    def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]:
        """Zwraca bazowy ekwipunek amunicji dla lekkiego czołgu."""
        return {AmmoType.HEAVY: AmmoSlot(AmmoType.HEAVY, 1),
                AmmoType.LIGHT: AmmoSlot(AmmoType.LIGHT, 15),
                AmmoType.LONG_DISTANCE: AmmoSlot(AmmoType.LONG_DISTANCE, 2)}


@dataclass
class HeavyTank(Tank):
    _tank_type: Literal["HeavyTank"] = field(default="HeavyTank", init=False)
    hp: int = 120 # punkty życia
    shield: int = 80 # punkty osłony
    _max_hp: int = hp # maksymalna ilość punktów życia
    _max_shield: int = shield # maksymalna ilość punktów osłony
    _top_speed: float = 1 # maksymalna prędkość ruchu
    _vision_range: float = 8 # zasięg widzenia
    _vision_angle: float = 60 # kąt widzenia
    _barrel_spin_rate: float = 70 # prędkość obrotu lufy/tick
    _heading_spin_rate: float = 30 # prędkość obrotu kadłuba/tick
    _max_ammo: Dict[AmmoType, int] = field(
        default_factory=lambda: { # maksymalna pojemność amunicji
            AmmoType.HEAVY: 5,
            AmmoType.LIGHT: 10,
            AmmoType.LONG_DISTANCE: 2
        }
    )

    def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]:
        """Zwraca bazowy ekwipunek amunicji dla czołgu ciężkiego."""
        return {AmmoType.HEAVY: AmmoSlot(AmmoType.HEAVY, 5),
                AmmoType.LIGHT: AmmoSlot(AmmoType.LIGHT, 10),
                AmmoType.LONG_DISTANCE: AmmoSlot(AmmoType.LONG_DISTANCE, 2)}


@dataclass
class Sniper(Tank):
    _tank_type: Literal["Sniper"] = field(default="Sniper", init=False)
    hp: int = 40 # ilość punktów życia
    shield: int = 30 # ilość punktów osłony
    _max_hp: int = hp # maksymalna ilość punktów życia
    _max_shield: int = shield # maksymalna ilość punktów osłony
    _top_speed: float = 3 # maksymalna prędkość ruchu
    _vision_range: float = 25 # zasięg widzenia
    _vision_angle: float = 20 # kąt widzenia
    _barrel_spin_rate: float = 100 # prędkość obrotu lufy/tick
    _heading_spin_rate: float = 45 # prędkość obrotu kadłuba/tick
    _max_ammo: Dict[AmmoType, int] = field( # maksymalna pojemność amunicji
        default_factory=lambda: {
            AmmoType.HEAVY: 1,
            AmmoType.LIGHT: 5,
            AmmoType.LONG_DISTANCE: 10
        }
    )

    def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]:
        """Zwraca bazowy ekwipunek amunicji dla Snipera."""
        return {AmmoType.HEAVY: AmmoSlot(AmmoType.HEAVY, 1),
                AmmoType.LIGHT: AmmoSlot(AmmoType.LIGHT, 5),
                AmmoType.LONG_DISTANCE: AmmoSlot(AmmoType.LONG_DISTANCE, 10)}


TankUnion = Union[LightTank, HeavyTank, Sniper] # Wszystkie typy czołgów


@dataclass
class MapInfo:
    """Informacje o mapie (rozmiar, przeszkody, tereny, czołgi, power-upy)."""
    _map_seed: str # Unikalny identyfikator mapy
    _size: List[int] = field(default_factory=lambda: [500, 500]) # Rozmiar mapy [szerokość, wysokość]
    obstacle_list: List[ObstacleUnion] # Lista przeszkód na mapie
    powerup_list: List[PowerUpData] # Lista power-up'ów na mapie
    terrain_list: List[TerrainUnion] # Lista terenów na mapie
    all_tanks: List[TankUnion] # Lista wszystkich czołgów na mapie


# ==============================================================================
# 2. KONTRAKT API (IAgentController)
# ==============================================================================

class IAgentController(ABC):
    """Abstrakcyjny kontroler."""

    @abstractmethod
    def get_action(
            self,
            current_tick: int, # Numer bieżącej klatki symulacji
            my_tank_status: TankUnion, # Aktualny stan czołgu agenta
            sensor_data: TankSensorData, # Dane sensoryczne czołgu agenta
            enemies_remaining: int # Liczba pozostałych wrogów 
    ) -> 'ActionCommand': pass
    """Metoda wywoływana w każdej klatce symulacji.
       Pozwala agentowi podjąć decyzję na podstawie aktualnego stanu czołgu
       i danych sensorycznych."""

    @abstractmethod
    def destroy(self): pass
    """Metoda wywoływana, gdy czołg zostaje zniszczony."""

    @abstractmethod
    def end(self): pass
    """Metoda wywoływana raz po zakończeniu symulacji."""


# ==============================================================================
# 3. KLASA DECYZYJNA (OUTPUT Z AGENTA DO SILNIKA)
# ==============================================================================

@dataclass
class ActionCommand:
    """Pojedynczy obiekt zawierający wszystkie polecenia dla Silnika w danej klatce."""
    
    barrel_rotation_angle: float
    """ Zmiana kąta obrotu lufy (- w lewo, + w prawo) - silnik ograniczy obrót zgodnie z barrel_spin_rate."""
    
    heading_rotation_angle: float
    """ Zmiana kąta obrotu kadłuba (- w lewo, + w prawo) - silnik ograniczy obrót zgodnie z heading_spin_rate."""
    
    move_speed: float 
    """Docelowa prędkość ruchu czołgu (silnik ograniczy prędkość zgodnie z top_speed).  
       Prędkość dodatnia oznacza jazdę do przodu, ujemna - do tyłu. Przy wartości 0 czołg stoi w miejscu."""
    
    ammo_to_load: Optional[AmmoType] = None
    """Typ amunicji do załadowania."""
    
    should_fire: bool = False
    """Czy czołg ma strzelać (jeśli lufa jest załadowana)."""
