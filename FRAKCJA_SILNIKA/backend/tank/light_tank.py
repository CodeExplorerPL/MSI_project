""" Klasa lekkiego czolgu """
from __future__ import annotations

from typing import Dict

from .base_tank import Tank
from ..structures import Position, AmmoType, AmmoSlot


class LightTank(Tank):
    def __init__(self, _id: str, team: int, start_pos: Position):
        super().__init__(
            _id=_id,
            _team=team,
            _vision_angle=40.0,
            _vision_range=70.0,
            _top_speed=5.0,
            _barrel_spin_rate=90.0,
            _heading_spin_rate=70.0,
            _max_hp=80,
            _max_shield=30,
        )
        self.position = start_pos
        self._tank_type = "LIGHT"

    def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]:
        return {
            AmmoType.HEAVY: AmmoSlot(AmmoType.HEAVY, 1),
            AmmoType.LIGHT: AmmoSlot(AmmoType.LIGHT, 15),
            AmmoType.LONG_DISTANCE: AmmoSlot(AmmoType.LONG_DISTANCE, 2),
        }
