""" Klasa ciezkiego czolgu """
from __future__ import annotations

from typing import Dict

from base_tank import Tank
from ..structures import Position, AmmoType, AmmoSlot


class HeavyTank(Tank):
    def __init__(self, _id: str, team: int, start_pos: Position):
        super().__init__(
            _id=_id,
            _team=team,
            _vision_angle=90.0,
            _vision_range=10.0,
            _top_speed=5.0,  # Cięższy, więc wolniejszy
            _barrel_spin_rate=60.0,
            _heading_spin_rate=60.0,
            _max_hp=150,
            _max_shield=100,
        )
        self.position = start_pos
        self._tank_type = "HEAVY"

    def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]:
        return {
            AmmoType.HEAVY: AmmoSlot(AmmoType.HEAVY, 15),
            AmmoType.LIGHT: AmmoSlot(AmmoType.LIGHT, 10),
        }
