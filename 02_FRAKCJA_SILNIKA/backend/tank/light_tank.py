""" Klasa lekkiego czolgu """


# @dataclass
# class LightTank(Tank):
#     tank_type: Literal["LightTank"] = field(default="LightTank", init=False)
#     hp: int = 80
#     shield: int = 30
#     _max_hp: int = hp
#     _max_shield: int = shield
#     _top_speed: float = 10
#     _vision_range: float = 10
#     _vision_angle: float = 40
#     _barrel_spin_rate: float = 180
#     _heading_spin_rate: float = 140
#     _max_ammo: Dict[AmmoType, int] = field(
#         default_factory=lambda: {
#             AmmoType.HEAVY: 1,
#             AmmoType.LIGHT: 15,
#             AmmoType.LONG_DISTANCE: 2
#         }
#     )
#
#     def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]:
#         return {AmmoType.HEAVY: AmmoSlot(AmmoType.HEAVY, 1),
#                 AmmoType.LIGHT: AmmoSlot(AmmoType.LIGHT, 15),
#                 AmmoType.LONG_DISTANCE: AmmoSlot(AmmoType.LONG_DISTANCE, 2)}