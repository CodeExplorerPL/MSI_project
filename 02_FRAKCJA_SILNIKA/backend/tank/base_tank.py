from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from ..structures.position import Position


""" Klasa abstrakcyjna czolgu """


class AmmoType:
    pass


class AmmoSlot:
    pass


@dataclass
class Tank(ABC):
    """Abstrakcyjna klasa bazowa dla wszystkich typów czołgów."""
    _id: str
    _team: int

    # Statystyki bazowe
    _tank_type: str = field(init=False)
    _vision_angle: float
    _vision_range: float
    move_speed: float # can be from (-top_speed, top_speed)
    _top_speed: float
    _barrel_spin_rate: float
    _heading_spin_rate: float
    _max_hp: int
    _max_shield: int

    # Dynamiczne statystyki
    hp: int
    shield: int
    position: Position
    ammo: Dict[AmmoType, AmmoSlot]
    _max_ammo: Dict[AmmoType, int]
    ammo_loaded: Optional[AmmoType] = None
    barrel_angle: float = 0.0
    heading: float = 0.0
    is_overcharged: bool = False
    size: List[int] = field(default_factory=lambda: [5, 5])

    @abstractmethod
    def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]: pass


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