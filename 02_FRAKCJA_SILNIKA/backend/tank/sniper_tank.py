""" Klasa czolgu dalekigo zasiegu"""


# @dataclass
# class Sniper(Tank):
#     tank_type: Literal["Sniper"] = field(default="Sniper", init=False)
#     hp: int = 40
#     shield: int = 30
#     _max_hp: int = hp
#     _max_shield: int = shield
#     _top_speed: float = 5
#     _vision_range: float = 25
#     _vision_angle: float = 20
#     _barrel_spin_rate: float = 200
#     _heading_spin_rate: float = 90
#     _max_ammo: Dict[AmmoType, int] = field(
#         default_factory=lambda: {
#             AmmoType.HEAVY: 1,
#             AmmoType.LIGHT: 5,
#             AmmoType.LONG_DISTANCE: 10
#         }
#     )
#
#     def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]:
#         return {AmmoType.HEAVY: AmmoSlot(AmmoType.HEAVY, 1),
#                 AmmoType.LIGHT: AmmoSlot(AmmoType.LIGHT, 5),
#                 AmmoType.LONG_DISTANCE: AmmoSlot(AmmoType.LONG_DISTANCE, 10)}