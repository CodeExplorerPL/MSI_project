from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple

import numpy as np
import random


class MotionMode(Enum):
    ROTATE = auto()
    BACKUP = auto()
    FORWARD = auto()
    GOTO = auto()


@dataclass(slots=True)
class MapConfig:
    width: float = 200.0
    height: float = 200.0
    margin: float = 5.0


@dataclass(slots=True)
class RecoveryConfig:
    backup_ticks: int = 3
    rotate_ticks: int = 3
    backup_speed: float = -5.0
    forward_speed: float = 5.0

    stuck_eps: float = 0.01
    stuck_patience_ticks: int = 50

    border_patience_ticks: int = 10


def _wrap_angle_deg(a: float) -> float:
    # wynik w [-180, 180)
    return (a + 180.0) % 360.0 - 180.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(slots=True)
class IterState:
    # --- FSM ruchu ---
    motion_mode: MotionMode = MotionMode.FORWARD
    motion_ticks_left: int = 0
    rotation_dir: int = 1  # 1=left, -1=right

    # "intencja" po recovery (do czego wracamy po BACKUP+ROTATE)
    desired_mode: MotionMode = MotionMode.FORWARD
    desired_speed: float = 5.0

    goto_x: Optional[float] = None
    goto_y: Optional[float] = None
    goto_stop_radius: float = 1.0

    # --- pamięć do guardów ---
    last_x: Optional[float] = None
    last_y: Optional[float] = None
    stuck_timer: int = 50

    border_timer: int = 10

    # debug/telemetry
    last_heading_cmd: float = 0.0

    # ===== API dla taktyk =====

    def current_motion_state(self) -> str:
        if self.motion_mode in (MotionMode.BACKUP, MotionMode.ROTATE):
            return (
                f"{self.motion_mode.name}(left={self.motion_ticks_left},"
                f" dir={self.rotation_dir})"
            )
        if self.motion_mode == MotionMode.GOTO:
            return f"GOTO(target=({self.goto_x},{self.goto_y}))"
        return "FORWARD"

    def set_forward(self, speed: float) -> None:
        self.desired_mode = MotionMode.FORWARD
        self.desired_speed = float(speed)

    def set_goto(self, x: float, y: float, speed: float, stop_radius: float) -> None:
        self.desired_mode = MotionMode.GOTO
        self.goto_x = float(x)
        self.goto_y = float(y)
        self.desired_speed = float(speed)
        self.goto_stop_radius = float(stop_radius)

    # ===== Guardy (oddzielne, reużywalne) =====

    def guard_border(
        self,
        curr_x: float,
        curr_y: float,
        map_cfg: MapConfig,
        rec_cfg: RecoveryConfig,
    ) -> bool:
        near_left = curr_x < map_cfg.margin
        near_right = curr_x > (map_cfg.width - map_cfg.margin)
        near_bottom = curr_y < map_cfg.margin
        near_top = curr_y > (map_cfg.height - map_cfg.margin)

        near = near_left or near_right or near_bottom or near_top
        if not near:
            self.border_timer = rec_cfg.border_patience_ticks
            return False

        if self.border_timer > 0:
            self.border_timer -= 1
            return False

        self.border_timer = rec_cfg.border_patience_ticks
        return True

    def guard_stuck(
        self,
        curr_x: float,
        curr_y: float,
        rec_cfg: RecoveryConfig,
    ) -> bool:
        triggered = False

        if self.last_x is not None and self.last_y is not None:
            dx = abs(curr_x - float(self.last_x))
            dy = abs(curr_y - float(self.last_y))
            dist = float(np.hypot(dx, dy))

            if dist < rec_cfg.stuck_eps:
                if self.stuck_timer > 0:
                    self.stuck_timer -= 1
                else:
                    self.stuck_timer = rec_cfg.stuck_patience_ticks
                    triggered = True
            else:
                self.stuck_timer = rec_cfg.stuck_patience_ticks
        else:
            self.stuck_timer = rec_cfg.stuck_patience_ticks

        self.last_x = curr_x
        self.last_y = curr_y
        return triggered

    @staticmethod
    def guard_obstacle(summary: Dict[str, Any]) -> bool:
        return bool(summary["self"]["obstacle_ahead"])

    @staticmethod
    def guard_terrain(summary: Dict[str, Any]) -> bool:
        return int(summary["self"].get("terrain_damage", 0)) > 0

    # ===== FSM: przejścia i wykonanie =====

    def _enter_backup(self, rec_cfg: RecoveryConfig) -> None:
        self.motion_mode = MotionMode.BACKUP
        self.motion_ticks_left = rec_cfg.backup_ticks

    def _enter_rotate(self, rec_cfg: RecoveryConfig) -> None:
        self.motion_mode = MotionMode.ROTATE
        self.motion_ticks_left = rec_cfg.rotate_ticks
        self.rotation_dir = random.choice([-1, 1])

    def _resume_desired(self) -> None:
        # wróć do FORWARD albo GOTO
        self.motion_mode = self.desired_mode

    def motion_step(
        self,
        summary: Dict[str, Any],
        *,
        rot_ang: float,
        map_cfg: MapConfig,
        rec_cfg: RecoveryConfig,
        enable_border: bool = True,
        enable_obstacle: bool = True,
        enable_stuck: bool = True,
        enable_terrein: bool = True,
    ) -> Tuple[float, float]:
        """
        Zwraca: (speed, heading_rot)

        Zasada:
        - BACKUP/ROTATE są nieprzerywalne (ignorują guardy).
        - FORWARD/GOTO automatycznie uwzględniają guardy (border/obstacle/stuck).
          Możesz je per-taktyka włączać/wyłączać flagami enable_*.
        """
        curr = summary["self"]["pos"]
        curr_x = float(curr["x"])
        curr_y = float(curr["y"])
        heading = float(summary["self"]["heading"])  # zakładam stopnie

        # max 2 "wejścia" w recovery w jednej klatce bez rekurencji
        for _ in range(3):
            if self.motion_mode == MotionMode.BACKUP:
                speed = rec_cfg.backup_speed
                heading_rot = 0.0

                self.motion_ticks_left -= 1
                if self.motion_ticks_left <= 0:
                    # po BACKUP zawsze ROTATE (dokańczalne)
                    self._enter_rotate(rec_cfg)

                self.last_heading_cmd = heading_rot
                return speed, heading_rot

            if self.motion_mode == MotionMode.ROTATE:
                speed = 0.0
                heading_rot = float(self.rotation_dir) * float(rot_ang)

                self.motion_ticks_left -= 1
                if self.motion_ticks_left <= 0:
                    self._resume_desired()

                self.last_heading_cmd = heading_rot
                return speed, heading_rot

            # ===== FORWARD / GOTO: tu działają guardy =====
            if self.motion_mode not in (MotionMode.FORWARD, MotionMode.GOTO):
                self.motion_mode = self.desired_mode

            border_hit = False
            if enable_border:
                border_hit = self.guard_border(curr_x, curr_y, map_cfg, rec_cfg)

            obstacle_hit = False
            if enable_obstacle:
                obstacle_hit = self.guard_obstacle(summary)

            terrain_hit = False
            if enable_terrein:
                terrain_hit = self.guard_terrain(summary)

            stuck_hit = False
            if enable_stuck:
                stuck_hit = self.guard_stuck(curr_x, curr_y, rec_cfg)

            if border_hit or obstacle_hit or terrain_hit:
                # przy granicy i przeszkodzie: lepiej BACKUP -> ROTATE
                self._enter_backup(rec_cfg)
                continue

            if stuck_hit:
                # jeśli stoimy w miejscu, czasem wystarczy sama rotacja
                self._enter_rotate(rec_cfg)
                continue

            # brak recovery -> wykonaj ruch "normalny"
            if self.motion_mode == MotionMode.FORWARD:
                speed = self.desired_speed
                heading_rot = 0.0
                self.last_heading_cmd = heading_rot
                return speed, heading_rot

            # GOTO: skręcaj w stronę targetu
            if self.goto_x is None or self.goto_y is None:
                speed = self.desired_speed
                heading_rot = 0.0
                self.last_heading_cmd = heading_rot
                return speed, heading_rot

            dx = float(self.goto_x) - curr_x
            dy = float(self.goto_y) - curr_y
            dist = float(np.hypot(dx, dy))

            if dist <= self.goto_stop_radius:
                speed = 0.0
                heading_rot = 0.0
                self.last_heading_cmd = heading_rot
                return speed, heading_rot

            target_angle = float(np.degrees(np.arctan2(dy, dx)))
            err = _wrap_angle_deg(target_angle - heading)

            # Sterowanie proporcjonalne + ograniczenie do max ROT_ANG/tick
            # k dobierz: 0.5..2.0 (większe = agresywniejszy skręt)
            k = 1.0
            heading_rot = _clamp(k * err, -float(rot_ang), float(rot_ang))

            # Deadband żeby nie "mielił" gdy prawie prosto
            if abs(err) < 1.0:
                heading_rot = 0.0

            speed = self.desired_speed
            self.last_heading_cmd = heading_rot
            return speed, heading_rot

        # fallback (nie powinno się zdarzyć)
        return 0.0, 0.0