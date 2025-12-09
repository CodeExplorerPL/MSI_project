from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AmmoType(Enum):
    """Definicja typów amunicji i ich bazowych właściwości."""
    HEAVY = {"Value": -40, "Range": 5, "ReloadTime": 2.0}
    LIGHT = {"Value": -20, "Range": 10, "ReloadTime": 1.0}
    LONG_DISTANCE = {"Value": -25, "Range": 20, "ReloadTime": 2.0}

    @property
    def value_amount(self) -> int:
        return self.value["Value"]

    @property
    def range(self) -> float:
        return self.value["Range"]

    @property
    def reload_time(self) -> float:
        return self.value["ReloadTime"]


@dataclass
class AmmoSlot:
    """Informacje o danym typie amunicji w ekwipunku czołgu."""
    ammo_type: AmmoType
    count: int
