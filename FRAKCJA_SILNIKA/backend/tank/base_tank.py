""" Klasa abstrakcyjna czolgu """
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, List

from ..structures import Position, AmmoType, AmmoSlot


@dataclass
class Tank(ABC):
    """Abstrakcyjna klasa bazowa dla wszystkich typów czołgów."""
    _id: str
    _team: int

    # Statystyki bazowe
    _tank_type: str = field(init=False, default="BASE")
    _vision_angle: float = 90.0
    _vision_range: float = 10.0
    _top_speed: float = 5.0
    _barrel_spin_rate: float = 90.0  # stopnie / sekunda
    _heading_spin_rate: float = 90.0  # stopnie / sekunda
    _max_hp: int = 100
    _max_shield: int = 50

    # Dynamiczne statystyki
    move_speed: float = 0.0  # (-top_speed, top_speed)
    hp: int = field(init=False)
    shield: int = field(init=False)
    position: Position = field(default_factory=lambda: Position(0.0, 0.0))
    ammo: Dict[AmmoType, AmmoSlot] = field(default_factory=dict)
    _max_ammo: Dict[AmmoType, int] = field(default_factory=dict)
    ammo_loaded: Optional[AmmoType] = None
    barrel_angle: float = 0.0  # kąt lufy
    heading: float = 0.0  # kąt kadłuba
    is_overcharged: bool = False
    size: List[int] = field(default_factory=lambda: [5, 5])

    # Reload
    _reload_timer: float = 0.0  # ile jeszcze do końca przeładowania

    def __post_init__(self):
        self.hp = self._max_hp
        self.shield = self._max_shield
        # startowy magazyn (dzieci mogą to nadpisać)
        base_ammo = self.get_base_ammo()
        if base_ammo:
            self.ammo = base_ammo
        # domyślnie ładuje pierwszy dostępny typ
        if self.ammo and self.ammo_loaded is None:
            self.ammo_loaded = next(iter(self.ammo.keys()))

    # =============================
    #   METODA ABSTRAKCYJNA
    # =============================
    @abstractmethod
    def get_base_ammo(self) -> Dict[AmmoType, AmmoSlot]:
        """Zwraca bazowy zestaw amunicji dla tego typu czołgu."""
        pass

    # =============================
    #   LOGIKA OBRAŻEŃ
    # =============================
    def take_damage(self, damage: int) -> None:
        if damage <= 0 or not self.is_alive():
            return

        # najpierw w tarczę
        shield_damage = min(self.shield, damage)
        self.shield -= shield_damage
        remaining = damage - shield_damage

        if remaining > 0:
            self.hp -= remaining
            if self.hp < 0:
                self.hp = 0

    def is_alive(self) -> bool:
        return self.hp > 0

    # =============================
    #   RUCH I OBRÓT
    # =============================
    def set_move_speed(self, value: float) -> None:
        # clamp do [-top_speed, top_speed]
        self.move_speed = max(-self._top_speed, min(value, self._top_speed))

    def move(self, dir_x: float, dir_y: float, delta_time: float, speed_factor: float = 1.0) -> None:
        length = (dir_x ** 2 + dir_y ** 2) ** 0.5
        if length == 0:
            return

        # final speed = terrain_factor * desired move_speed
        speed = self.move_speed * max(0.0, speed_factor)

        nx = dir_x / length
        ny = dir_y / length

        self.position.x += nx * speed * delta_time
        self.position.y += ny * speed * delta_time

    def _normalize_angle(self, angle: float) -> float:
        angle = angle % 360.0
        if angle < 0:
            angle += 360.0
        return angle

    def rotate_heading(self, delta_degrees: float, delta_time: float) -> None:
        """Obrót kadłuba z ograniczeniem prędkości obrotu."""
        max_delta = self._heading_spin_rate * delta_time
        delta = max(-max_delta, min(delta_degrees, max_delta))
        self.heading = self._normalize_angle(self.heading + delta)

    def rotate_barrel(self, delta_degrees: float, delta_time: float) -> None:
        """Obrót lufy z ograniczeniem prędkości obrotu."""
        max_delta = self._barrel_spin_rate * delta_time
        delta = max(-max_delta, min(delta_degrees, max_delta))
        self.barrel_angle = self._normalize_angle(self.barrel_angle + delta)

    # =============================
    #   STRZELANIE I RELOAD
    # =============================
    def update_reload(self, delta_time: float) -> None:
        if self._reload_timer > 0:
            self._reload_timer -= delta_time
            if self._reload_timer < 0:
                self._reload_timer = 0

    def can_shoot(self) -> bool:
        if not self.is_alive():
            return False
        if self._reload_timer > 0:
            return False
        if self.ammo_loaded is None:
            return False
        slot = self.ammo.get(self.ammo_loaded)
        return slot is not None and slot.count > 0

    def shoot(self) -> Optional[int]:
        """
        Oddaje strzał.
        Zwraca wartość obrażeń (dodatnią liczbę) albo None jeśli strzał niemożliwy.
        """
        if not self.can_shoot():
            return None

        slot = self.ammo[self.ammo_loaded]
        slot.count -= 1

        dmg = -self.ammo_loaded.value_amount  # bo w enumie jest -40, -20 itd.
        reload_time = self.ammo_loaded.reload_time
        self._reload_timer = reload_time
        return dmg

    # =============================
    #   GETTERY / SETTERY (properties)
    # =============================

    @property
    def id(self) -> str:
        return self._id

    @property
    def team(self) -> int:
        return self._team


    @property
    def tank_type(self) -> str:
        return self._tank_type


    @property
    def vision_angle(self) -> float:
        return self._vision_angle


    @property
    def vision_range(self) -> float:
        return self._vision_range


    @property
    def top_speed(self) -> float:
        return self._top_speed


    @property
    def max_hp(self) -> int:
        return self._max_hp

    @property
    def max_shield(self) -> int:
        return self._max_shield

    @property
    def reload_timer(self) -> float:
        return self._reload_timer
