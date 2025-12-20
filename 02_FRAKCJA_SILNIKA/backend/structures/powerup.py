""" Klasa powerupu """
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from .position import Position



class PowerUpType(Enum):
    """Definicja typów PowerUp'ów i ich wartości."""
    MEDKIT = {"Name": "Medkit", "Value": 50}
    SHIELD = {"Name": "Shield", "Value": 20}
    OVERCHARGE = {"Name": "Overcharge", "Value": 2}
    AMMO_HEAVY = {"Name": "HeavyAmmo", "Value": 2, "AmmoType": "HEAVY"}
    AMMO_LIGHT = {"Name": "LightAmmo", "Value": 5, "AmmoType": "LIGHT"}
    AMMO_LONG_DISTANCE = {"Name": "LongDistanceAmmo", "Value": 2,
                          "AmmoType": "LONG_DISTANCE"}

@dataclass
class PowerUpData:
    """Informacje o przedmiocie do zebrania (np. Apteczka, Amunicja)."""
    position: Position
    powerup_type: PowerUpType
    size: List[int] = field(default_factory=lambda: [2, 2])

    @property
    def value(self) -> int: return self.powerup_type.value['Value']

    @property
    def name(self) -> str: return self.powerup_type.value['Name']