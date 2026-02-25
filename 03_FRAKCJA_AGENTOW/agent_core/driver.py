from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

from .geometry import euclidean_distance, heading_to_angle_deg, normalize_angle_diff
from .world_model import WorldModel


class MotionDriver:
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model
        self.path: List[Tuple[int, int]] = []
        self.last_position: Optional[Tuple[float, float]] = None
        self.last_move_cmd: float = 0.0
        self.stuck_ticks: int = 0
        self.escape_ticks: int = 0
        self.escape_heading: Optional[float] = None
        self.unblock_ticks: int = 0

    @staticmethod
    def neighbors4(cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = cell
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    def best_immediate_safe_neighbor(
        self, my_cell: Tuple[int, int], allow_risky: bool = False
    ) -> Optional[Tuple[int, int]]:
        best = None
        best_score = -1e9
        for cell in self.neighbors4(my_cell):
            if not allow_risky and self.world_model.is_blocked_for_pathing(cell):
                continue
            state = self.world_model.get_state(cell)
            score = (
                4.0 * state.safe
                - 8.0 * state.danger
                - 4.0 * state.blocked
                - 0.2 * self.world_model.visit_counts.get(cell, 0)
            )
            if score > best_score:
                best_score = score
                best = cell
        return best

    def drive_to_cell(
        self,
        my_x: float,
        my_y: float,
        my_heading: float,
        target_cell: Tuple[int, int],
        top_speed: float,
    ) -> Tuple[float, float]:
        tx, ty = self.world_model.to_world_center(target_cell)
        return self.drive_to_point(my_x, my_y, my_heading, tx, ty, top_speed)

    def drive_to_point(
        self,
        my_x: float,
        my_y: float,
        my_heading: float,
        tx: float,
        ty: float,
        top_speed: float,
    ) -> Tuple[float, float]:
        target_angle = heading_to_angle_deg(my_x, my_y, tx, ty)
        diff = normalize_angle_diff(target_angle, my_heading)
        abs_diff = abs(diff)
        if abs_diff <= 18.0:
            turn_limit = 10.0
        elif abs_diff <= 45.0:
            turn_limit = 16.0
        else:
            turn_limit = 22.0
        turn = max(-turn_limit, min(turn_limit, diff))

        if abs_diff > 60.0:
            speed = top_speed * 0.50
        elif abs_diff > 30.0:
            speed = top_speed * 0.76
        else:
            speed = top_speed
        return turn, speed

    def drive_path(
        self, my_x: float, my_y: float, my_heading: float, top_speed: float
    ) -> Tuple[float, float]:
        if not self.path:
            return 0.0, max(0.7, top_speed * 0.58)

        next_cell = self.path[0]

        if self.world_model.is_dangerous_cell(next_cell):
            my_cell = self.world_model.to_cell(my_x, my_y)
            safe_neighbor = self.best_immediate_safe_neighbor(my_cell)
            if safe_neighbor is not None and safe_neighbor != my_cell:
                return self.drive_to_cell(
                    my_x, my_y, my_heading, safe_neighbor, top_speed
                )

        wx, wy = self.world_model.to_world_center(self.path[0])
        target_angle = heading_to_angle_deg(my_x, my_y, wx, wy)
        diff = normalize_angle_diff(target_angle, my_heading)

        abs_diff = abs(diff)

        if abs_diff < 4.0:
            turn = 0.0
        else:
            turn_limit = 13.0 if abs_diff <= 20.0 else 18.0
            turn = max(-turn_limit, min(turn_limit, diff))

        if abs_diff > 55.0:
            speed = top_speed * 0.60
        elif abs_diff > 24.0:
            speed = top_speed * 0.82
        else:
            speed = top_speed
        return turn, speed

    def update_stuck(
        self,
        my_x: float,
        my_y: float,
        enemies_visible: bool,
        heading: float,
        blocking_tank_in_front: bool = False,
    ) -> bool:
        if self.last_position is None:
            self.last_position = (my_x, my_y)
            self.stuck_ticks = 0
            return False

        moved = euclidean_distance(
            my_x, my_y, self.last_position[0], self.last_position[1]
        )
        trying = self.last_move_cmd > (0.2 if blocking_tank_in_front else 0.4)

        if trying and moved < 0.15 and not enemies_visible:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = max(0, self.stuck_ticks - 1)

        if self.stuck_ticks >= 10:
            for angle_offset in (-28.0, 0.0, 28.0):
                h = math.radians((heading + angle_offset) % 360)
                px = my_x + math.cos(h) * 9.0
                py = my_y + math.sin(h) * 9.0
                blocked_cell = self.world_model.to_cell(px, py)
                self.world_model.get_state(blocked_cell).blocked += 1.25
                self.world_model.mark_dead_end(blocked_cell, ttl=560.0)
            self.path = []
            self.stuck_ticks = 0
            self.last_position = (my_x, my_y)
            return True

        self.last_position = (my_x, my_y)
        return False

    def start_escape(self, my_heading: float, force_new: bool = False) -> None:
        if self.escape_heading is None or force_new:
            turn = random.choice(
                [120, 135, 150, 165, 180, -120, -135, -150, -165, -180]
            )
            self.escape_heading = (my_heading + turn) % 360
        self.escape_ticks = max(self.escape_ticks, 45)

    def start_unblock(self) -> None:
        self.unblock_ticks = 12

    def unblock_drive(
        self, top_speed: float, add_turn: bool = False
    ) -> Tuple[float, float]:
        if self.unblock_ticks <= 0:
            return 0.0, 0.0
        self.unblock_ticks -= 1
        speed = -top_speed * 0.6
        turn = 0.0
        if add_turn and self.unblock_ticks > 6:
            turn = 22.0 if (self.unblock_ticks % 2 == 0) else -22.0
        return turn, speed

    def escape_drive(
        self, my_x: float, my_y: float, my_heading: float, top_speed: float
    ) -> Tuple[float, float]:
        my_cell = self.world_model.to_cell(my_x, my_y)
        safe_neighbor = self.best_immediate_safe_neighbor(my_cell)
        if safe_neighbor is None:
            safe_neighbor = self.best_immediate_safe_neighbor(my_cell, allow_risky=True)
        if safe_neighbor is not None:
            turn, speed = self.drive_to_cell(
                my_x, my_y, my_heading, safe_neighbor, top_speed
            )
            self.escape_ticks = max(0, self.escape_ticks - 1)
            if self.escape_ticks == 0:
                self.escape_heading = None
            return turn, speed

        if self.escape_heading is None:
            self.start_escape(my_heading)

        assert self.escape_heading is not None
        diff = normalize_angle_diff(self.escape_heading, my_heading)
        turn = max(-18.0, min(18.0, diff))
        speed = top_speed if abs(diff) < 30 else top_speed * 0.5
        self.escape_ticks = max(0, self.escape_ticks - 1)
        if self.escape_ticks == 0:
            self.escape_heading = None

        return turn, speed
