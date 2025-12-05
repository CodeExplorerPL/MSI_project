""" Klasa ciezkiego czolgu """


# @dataclass
# class HeavyTank(Tank):
#     tank_type: Literal["HeavyTank"] = field(default="HeavyTank", init=False)
#     hp: int = 120
#     shield: int = 80
#     _max_hp: int = hp
#     _max_shield: int = shield
#     _top_speed: float = 2
#     _vision_range: float = 8
#     _vision_angle: float = 60
#     _barrel_spin_rate: float = 140
#     _heading_spin_rate: float = 60
#     _max_ammo: Dict[AmmoType, int] = field(
#         default_factory=lambda: {
#             AmmoType.HEAVY: 5,
#             AmmoType.LIGHT: 10,
#             AmmoType.LONG_DISTANCE: 2
#         }
#     )
#
#     def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]:
#         return {AmmoType.HEAVY: AmmoSlot(AmmoType.HEAVY, 5),
#                 AmmoType.LIGHT: AmmoSlot(AmmoType.LIGHT, 10),
#                 AmmoType.LONG_DISTANCE: AmmoSlot(AmmoType.LONG_DISTANCE, 2)}