from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Literal, Any, Union
from dataclasses import dataclass
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Importy z backendu
# from backend.structures.position import Position
from backend.structures.ammo import AmmoType, AmmoSlot

from backend.tank.base_tank import Tank
from backend.tank.light_tank import LightTank
from backend.tank.heavy_tank import HeavyTank
from backend.tank.sniper_tank import SniperTank as Sniper
from backend.tank.sensor_data import TankSensorData, SeenTank

TankUnion = Union[LightTank, HeavyTank, Sniper]

# ==============================================================================
# STRUKTURY DANYCH
# ==============================================================================

@dataclass
class Scoreboard:
    """Struktura przechowująca wynik agenta po zakończeniu gry."""
    damage_dealt: float
    tanks_killed: int

# ==============================================================================
# KONTRAKT API (IAgentController)
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
    def end(self, final_score: Scoreboard): pass
    """Metoda wywoływana raz po zakończeniu symulacji."""


# ==============================================================================
# KLASA DECYZYJNA (OUTPUT Z AGENTA DO SILNIKA)
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


# ==============================================================================
# REJESTRACJA AGENTA (Dependency Injection)
# ==============================================================================

_active_agent: Optional[IAgentController] = None

def set_active_agent(agent: IAgentController):
    """Ustawia instancję agenta, która będzie obsługiwać żądania API."""
    global _active_agent
    _active_agent = agent

def get_active_agent() -> IAgentController:
    """Zwraca aktywną instancję agenta."""
    if _active_agent is None:
        raise RuntimeError("Agent nie został zainicjalizowany! Użyj set_active_agent() przed uruchomieniem serwera.")
    return _active_agent