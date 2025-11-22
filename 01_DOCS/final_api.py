from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from enum import Enum

# ==============================================================================
# 1. STRUKTURY DANYCH (INPUT DLA AGENTA)
#    Definicje danych zwracanych przez Silnik do Agenta w każdym kroku.
# ==============================================================================

AmmoDamage = int(25)  # Domyślne obrażenia zadawane przez standardową amunicję
ReloadTime = int(1)   # Domyślny czas przeładowania amunicji w tickach

class ObstacleData:
    """Informacje o przeszkodzie na mapie."""
    def __init__(self,
                 position: 'Position',
                 size: 1, # Rozmiar przeszkody (np. promień dla okrągłej przeszkody)
                 is_destructible: bool, # Typ przeszkody (np. 'WALL' - False, 'TREE' - True)
                 is_alive: bool = True
                 ):
        
        self.position = position
        self.size = size
        self.is_destructible = is_destructible
        self.is_alive = is_alive

class Position:
    """Reprezentuje pozycję X, Y na mapie."""
    def __init__(self, x: float, y: float):
        self.x = x 
        self.y = y


class Direction(Enum): # Obrót możliwy tylko o 90 stopni
    N = "North"
    E = "East"
    S = "South"
    W = "West"
    

class PowerUpType(Enum):
    M = {"Name": "Medkit", "Value": 50}     # Przywraca 50 punktów struktury
    A = {"Name": "Ammo", "Value": 10}       # Dodaje 10 sztuk amunicji





class TankStatus:
    """Stan własnego czołgu (statystyki i zasoby)."""
    def __init__(self,
                 position: Position,                    # Pozycja czołgu na mapie
                 barrel: float,                         # Kąt lufy (w stopniach od 0 do 360, 0 to N)
                 heading: Direction,                    # Kierunek czołgu (N, E, S, W)
                 team: int,                             # Numer drużyny czołgu
                 vision_range: float = 60.0,            # Zasięg widzenia czołgu ( 60 stopni od kąta lufy)
                 ammo_count: int = 20,                  # Liczba amunicji dostępnej do strzału
                 hp: float = 100.0,                     # Punkty (HP) czołgu
                 is_loaded: bool = True,                # Czy czołg jest gotowy do strzału
                 barrel_spin_rate: float = 180.0,       # Prędkość obrotu wieży (stopnie na sekundę)
                
                 ):
        
        self.position = position
        self.barrel = barrel
        self.heading = heading
        self.hp = hp
        self.ammo_count = ammo_count
        self.team = team
        self.is_loaded = is_loaded
        self.vision_range = vision_range
        self.barrel_spin_rate = barrel_spin_rate
        

class MapInfo:
    """Informacje o mapie (rozmiar, przeszkody, itp.)."""
    def __init__(self,
                 width: 50,
                 height: 50,
                 obstacles: List['ObstacleData'],       # Lista przeszkód na mapie
                 tanks: List['TankStatus'],             # Lista czołgów na mapie
                 powerups: List['PowerUpData'] = []     # Lista przedmiotów do zebrania
                 ):
        
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.tanks = tanks
        self.powerups = powerups

class SensorData:
    """Dane wykryte przez systemy sensoryczne czołgu."""
    def __init__(self,
                 enemy_tanks: List['EnemyData'],        # Lista widocznych wrogów
                 visible_powerups: List['PowerUpData'], # Widoczne przedmioty do zebrania 
                 obstacles: List['ObstacleData'] = []   # Widoczne przeszkody
                 ):
        
        self.enemy_tanks = enemy_tanks
        self.visible_powerups = visible_powerups
        self.obstacles = obstacles

class EnemyData:
    """Informacje o widocznym wrogu (dla Agentów to ID i pozycja)."""
    def __init__(self,
                 enemy_id: int,
                 position: Position,
                 heading: float,
                 distance: float,
                 team: int
               
                 ):
        
        self.enemy_id = enemy_id
        self.position = position
        self.heading = heading
        self.distance = distance
        self.team = team

class PowerUpData:
    """Informacje o przedmiocie do zebrania (np. Apteczka, Amunicja)."""
    def __init__(self,
                 PowerUpType: PowerUpType, # Np. 'MEDKIT', 'AMMO', 
                 position: Position,
                 ):
        
        self.position = position
        self.value = PowerUpType.value['Value']
        self.name = PowerUpType.value['Name']

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
        :param delta_time: Czas, jaki upłynął od ostatniej klatki w sekundach (do obliczeń fizycznych).
        :return: Obiekt zawierający wszystkie decyzje podjęte w danym kroku.
        """
        pass

    @abstractmethod
    def destroy(self):
        """
        Metoda wywoływana przy niszczeniu Agenta (np. do czyszczenia zasobów).
        Domyślnie nic nie robi.
        """
        pass
    
    @abstractmethod
    def end(self):
        """
        Metoda wywoływana na końcu symulacji (np. do podsumowania wyników).
        Domyślnie nic nie robi.
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
                 barrel_rotation_angle: float,              # Zmiana kąta wieżyczki (w stopniach)
                 heading_direction: Direction,              # Zmiana Kierunku jazdy czołgu (N, E, S, W)
                 should_move: bool = False,                 # Czy Agent chce się ruszyć?
                 should_fire: bool = False,                 # Czy Agent chce strzelić? 

                 ):
        self.heading_direction = heading_direction
        self.barrel_rotation_angle = barrel_rotation_angle
        self.should_fire = should_fire
        self.should_move = should_move