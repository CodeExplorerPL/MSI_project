from __future__ import annotations
from typing import Dict
from .base_tank import Tank
from ..structures import Position, AmmoType, AmmoSlot

class SniperTank(Tank):
    def __init__(self, _id: str, team: int, start_pos: Position):
        super().__init__(
            _id=_id,
            _team=team,
            _vision_angle=20.0,
            _vision_range=120.0,
            _top_speed=3.0,
            _barrel_spin_rate=100.0,
            _heading_spin_rate=45.0,
            _max_hp=40,
            _max_shield=30,
        )
        self.position = start_pos
        self._tank_type = "Sniper"

    def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]:
        return {
            AmmoType.HEAVY: AmmoSlot(AmmoType.HEAVY, 1),
            AmmoType.LIGHT: AmmoSlot(AmmoType.LIGHT, 5),
            AmmoType.LONG_DISTANCE: AmmoSlot(AmmoType.LONG_DISTANCE, 10),
        }
