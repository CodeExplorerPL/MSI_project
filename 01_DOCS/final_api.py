from abc import ABC, abstractmethod
from typing import List, Optional, Dict

# ==============================================================================
# 1. STRUKTURY DANYCH (INPUT DLA AGENTA)
#    Definicje danych zwracanych przez Silnik do Agenta w każdym kroku.
# ==============================================================================

class Position:
    """Reprezentuje pozycję X, Y na mapie."""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

class TankStatus:
    """Stan własnego czołgu (statystyki i zasoby)."""
    def __init__(self,
                 position: Position,
                 heading: float,  # Kąt, w którym jest zwrócony czołg (w stopniach)
                 structure_points: float, # Punkty Struktury (HP) 
                 front_armor: float,      # Pancerz przedni
                 fuel_level: float,       # Zużywalne paliwo 
                 ammo_count: Dict[str, int], # Liczba amunicji w podziale na typy (np. {'AP': 10, 'HE': 5}) 
                 heat_level: float,       # Aktualne przegrzanie działa (0.0 do 1.0) 
                 is_jammed: bool,         # Czy działo jest zablokowane z powodu przegrzania? 
                 is_in_stealth: bool      # Czy czołg jest w ukryciu (np. w krzakach)? 
                 ):
        self.position = position
        self.heading = heading
        self.structure_points = structure_points
        self.front_armor = front_armor
        self.fuel_level = fuel_level
        self.ammo_count = ammo_count
        self.heat_level = heat_level
        self.is_jammed = is_jammed
        self.is_in_stealth = is_in_stealth

class SensorData:
    """Dane wykryte przez systemy sensoryczne czołgu."""
    def __init__(self,
                 enemy_tanks: List['EnemyData'], # Lista widocznych wrogów
                 visible_powerups: List['PowerUpData'], # Widoczne przedmioty do zebrania 
                 terrain_modifiers: List['TerrainModifier'] # Modyfikatory terenu w zasięgu (np. błoto, asfalt) 
                 ):
        self.enemy_tanks = enemy_tanks
        self.visible_powerups = visible_powerups
        self.terrain_modifiers = terrain_modifiers

class EnemyData:
    """Informacje o widocznym wrogu (dla Agentów to ID i pozycja)."""
    def __init__(self,
                 enemy_id: int,
                 position: Position,
                 heading: float,
                 distance: float,
                 # Opcjonalnie: ostatni znany typ amunicji, którym strzelał
                 ):
        self.enemy_id = enemy_id
        self.position = position
        self.heading = heading
        self.distance = distance

class PowerUpData:
    """Informacje o przedmiocie do zebrania (np. Apteczka, Amunicja)."""
    def __init__(self,
                 item_type: str, # Np. 'MEDKIT', 'AMMO', 'FUEL' 
                 position: Position,
                 ):
        self.item_type = item_type
        self.position = position

class TerrainModifier:
    """Informacja o typie terenu w danej pozycji."""
    def __init__(self,
                 position: Position,
                 type: str # Np. 'ASPHALT', 'MUD', 'WATER', 'BUSH' 
                 ):
        self.position = position
        self.type = type


# ==============================================================================
# 2. KONTRAKT API (FUNKCJE WYWOŁYWANE PRZEZ AGENTA)
#    Abstrakcyjna klasa, która musi być zaimplementowana w logice Agenta.
# ==============================================================================

class IAgentController(ABC):
    """
    Abstrakcyjny kontroler, definiujący interfejs do sterowania czołgiem.
    Agent implementuje te metody. Silnik je wywołuje w każdej klatce.
    """

    @abstractmethod
    def get_action(self,
                   my_tank_status: TankStatus,
                   sensor_data: SensorData,
                   delta_time: float
                   ) -> 'ActionCommand':
        """
        Główna metoda Al, wywoływana przez Silnik w każdym kroku symulacji.

        :param my_tank_status: Aktualne statystyki czołgu.
        :param sensor_data: Dane sensoryczne o widocznych wrogach i obiektach.
        :param delta_time: Czas, jaki upłynął od ostatniej klatki (do obliczeń fizycznych).
        :return: Obiekt zawierający wszystkie decyzje podjęte w danym kroku.
        """
        pass

# ==============================================================================
# 3. KLASA DECYZYJNA (OUTPUT Z AGENTA DO SILNIKA)
#    Pojedynczy obiekt, który Agent zwraca, by przekazać wszystkie swoje intencje
#    do Silnika.
# ==============================================================================

class ActionCommand:
    """
    Pojedynczy obiekt zawierający wszystkie polecenia dla Silnika w danej klatce.
    """
    def __init__(self,
                 forward_backward_force: float = 0.0, # Moc silnika (-1.0 do 1.0)
                 steering_angle: float = 0.0,       # Pożądany kąt skrętu (np. -10.0 do 10.0 stopni)
                 turret_rotation_speed: float = 0.0, # Prędkość obrotu wieży (w stopniach/s)
                 should_fire: bool = False,         # Czy Agent chce strzelić? 
                 ammo_type_to_use: Optional[str] = None, # Jaki typ amunicji użyć? (np. 'AP', 'HE') 
                 should_drop_mine: bool = False     # Czy Agent chce podrzucić minę/pułapkę? 
                 ):
        self.forward_backward_force = forward_backward_force
        self.steering_angle = steering_angle
        self.turret_rotation_speed = turret_rotation_speed
        self.should_fire = should_fire
        self.ammo_type_to_use = ammo_type_to_use
        self.should_drop_mine = should_drop_mine