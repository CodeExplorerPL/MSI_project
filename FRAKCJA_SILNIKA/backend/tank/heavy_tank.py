""" Klasa ciezkiego czolgu """
from __future__ import annotations

from typing import Dict

from .base_tank import Tank
from ..structures import Position, AmmoType, AmmoSlot


class HeavyTank(Tank):
    def __init__(self, _id: str, team: int, start_pos: Position):
        super().__init__(
            _id=_id,
            _team=team,
            _vision_angle=60.0,
            _vision_range=40.0,
            _top_speed=1.0,  # Cięższy, więc wolniejszy
            _barrel_spin_rate=70.0,
            _heading_spin_rate=30.0,
            _max_hp=120,
            _max_shield=80,
        )
        self.position = start_pos
        self._tank_type = "HEAVY"

    def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]:
        return {
            AmmoType.HEAVY: AmmoSlot(AmmoType.HEAVY, 5),
            AmmoType.LIGHT: AmmoSlot(AmmoType.LIGHT, 10),
            AmmoType.LONG_DISTANCE: AmmoSlot(AmmoType.LONG_DISTANCE, 2),
        }
