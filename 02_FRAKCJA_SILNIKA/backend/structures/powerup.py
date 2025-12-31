"""Klasa powerupu"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Union
from .position import Position

class PowerUpType(Enum):
    """Definicja typów PowerUp'ów i ich wartości."""
    MEDKIT = {"Name": "Medkit", "Value": 50}
    SHIELD = {"Name": "Shield", "Value": 20}
    OVERCHARGE = {"Name": "Overcharge", "Value": 2}
    AMMO_HEAVY = {"Name": "HeavyAmmo", "Value": 2, "AmmoType": "HEAVY"}
    AMMO_LIGHT = {"Name": "LightAmmo", "Value": 5, "AmmoType": "LIGHT"}
    AMMO_LONG_DISTANCE = {"Name": "LongDistanceAmmo", "Value": 2, "AmmoType": "LONG_DISTANCE"}


@dataclass
class PowerUpData:
    """Informacje o przedmiocie do zebrania (np. Apteczka, Amunicja)."""
    _position: Position
    _powerup_type: PowerUpType
    _size: List[int] = field(default_factory=lambda: [2, 2])
    
    @property
    def position(self) -> Position:
        return self._position
    
    @property
    def powerup_type(self) -> PowerUpType:
        return self._powerup_type
    
    @property
    def size(self) -> List[int]:
        return self._size
    
    @property
    def value(self) -> int:
        return self._powerup_type.value['Value']
    
    @property
    def name(self) -> str:
        return self._powerup_type.value['Name']

    @property
    def ammo_type(self) -> Union[str, None]:
        return self.powerup_type.value.get("AmmoType")
