from __future__ import annotations

from typing import Dict, List, Tuple, Any
import math

from .controller_types import MotionOutput, NavigatorOutput


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _gauss(x: float, c: float, s: float) -> float:
    s = max(1e-6, float(s))
    z = (float(x) - float(c)) / s
    return math.exp(-0.5 * z * z)


def _quantize(v: float, levels: Tuple[float, ...]) -> float:
    if not levels:
        return float(v)
    return float(min(levels, key=lambda x: abs(float(v) - float(x))))


class TSK_MotionController:
    """
    Low-level fuzzy motion controller.
    Controls:
    - move_speed (throttle / braking), normalized in [-1, 1]
    - heading_rotation_angle (hull steering), normalized in [-1, 1]
    """

    def __init__(self, weights: Dict[str, Any] | None = None) -> None:
        cfg = dict((weights or {}).get("motion", {}))

        self._MOVE_LEVELS: Tuple[float, ...] = tuple(
            float(v)
            for v in cfg.get("move_levels", (-0.30, 0.0, 0.18, 0.32, 0.48, 0.72, 0.92))
        )
        self._TURN_LEVELS: Tuple[float, ...] = tuple(
            float(v)
            for v in cfg.get("turn_levels", (-1.0, -0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 0.75, 1.0))
        )
        self.forward_bias = float(cfg.get("forward_bias", 0.10))

        self._prev_pos: Tuple[float, float] | None = None
        self._stuck_ticks: int = 0
        self._recovery_phase: float = 0.0
        self._prev_move_cmd: float = 0.0
        self._prev_turn_cmd: float = 0.0
        self._turn_flip_memory: float = 0.0

    @property
    def stuck_ticks(self) -> int:
        return int(self._stuck_ticks)

    @staticmethod
    def _weighted_avg(items: Tuple[Tuple[float, float], ...], default: float = 0.0) -> float:
        w_sum = 0.0
        y_sum = 0.0
        for w, y in items:
            w = max(0.0, float(w))
            w_sum += w
            y_sum += w * float(y)
        if w_sum <= 1e-8:
            return float(default)
        return y_sum / w_sum

    def _update_stuck_counter(self, x: float, y: float) -> None:
        cur = (float(x), float(y))
        if self._prev_pos is not None:
            if math.hypot(cur[0] - self._prev_pos[0], cur[1] - self._prev_pos[1]) < 0.45:
                self._stuck_ticks += 1
            else:
                self._stuck_ticks = 0
        self._prev_pos = cur

    def decide(
        self,
        state: Dict[str, float | str],
        sensor_snapshot: Dict[str, List[Tuple[float, float]]],
        nav: NavigatorOutput,
    ) -> MotionOutput:
        px = float(state.get("x", 0.0))
        py = float(state.get("y", 0.0))
        self._update_stuck_counter(px, py)
        self._recovery_phase += 0.15

        map_w = float(state.get("map_w", 200.0))
        map_h = float(state.get("map_h", 200.0))
        max_dist = max(1.0, math.hypot(map_w, map_h))

        goal_dist_norm = _clamp(float(nav.distance_to_waypoint) / max_dist, 0.0, 1.0)
        goal_bearing_norm = _clamp(float(nav.heading_error_deg) / 180.0, -1.0, 1.0)

        seen_enemy = bool(nav.seen_enemy or float(state.get("seen_opp", 0.0)) > 0.5 or sensor_snapshot.get("seen_tanks"))
        enemy_seen = 1.0 if seen_enemy else 0.0
        enemy_dist_norm = _clamp(float(state.get("enemy_dist", max_dist)) / max_dist, 0.0, 1.0)
        enemy_bearing_norm = _clamp(float(state.get("enemy_bearing", 0.0)) / 180.0, -1.0, 1.0)

        terrain_speed = float(state.get("terrain_speed", 1.0))
        terrain_dmg = float(state.get("terrain_dmg", 0.0))
        terrain_badness = _clamp(
            0.65 * _clamp((1.0 - terrain_speed) / 0.6, 0.0, 1.0)
            + 0.35 * _clamp(terrain_dmg / 2.0, 0.0, 1.0),
            0.0,
            1.0,
        )
        avoid = 1.0 if nav.avoid_obstacle else 0.0

        target_b = enemy_bearing_norm if enemy_seen > 0.5 else goal_bearing_norm
        dist_ref = enemy_dist_norm if enemy_seen > 0.5 else goal_dist_norm

        d_near = _gauss(dist_ref, 0.12, 0.12)
        d_mid = _gauss(dist_ref, 0.45, 0.20)
        d_far = _gauss(dist_ref, 0.85, 0.24)

        b_left = _gauss(target_b, +0.85, 0.35)
        b_center = _gauss(target_b, 0.0, 0.22)
        b_right = _gauss(target_b, -0.85, 0.35)

        bad_low = _gauss(terrain_badness, 0.0, 0.25)
        bad_high = _gauss(terrain_badness, 1.0, 0.35)

        move = self._weighted_avg(
            (
                ((1.0 - enemy_seen) * d_far * bad_low * (1.0 - avoid), +0.92),
                ((1.0 - enemy_seen) * d_mid * bad_low, +0.65),
                ((1.0 - enemy_seen) * d_near, +0.20),
                ((1.0 - enemy_seen) * bad_high, +0.25),
                (enemy_seen * d_far * bad_low, +0.75),
                (enemy_seen * d_mid * bad_low, +0.42),
                (enemy_seen * d_near, -0.15),
                (enemy_seen * bad_high, +0.18),
                (avoid * d_near, +0.18),
                (avoid * d_mid, +0.32),
            ),
            default=0.40,
        )

        turn = self._weighted_avg(
            (
                (b_left * d_far, +0.95),
                (b_left * d_mid, +0.78),
                (b_left * d_near, +0.56),
                (b_center, 0.00),
                (b_right * d_far, -0.95),
                (b_right * d_mid, -0.78),
                (b_right * d_near, -0.56),
                (avoid * _gauss(target_b, +0.65, 0.35), +0.88),
                (avoid * _gauss(target_b, -0.65, 0.35), -0.88),
            ),
            default=0.0,
        )
        turn = 0.55 * turn + 0.45 * _clamp(target_b, -1.0, 1.0)

        if enemy_seen > 0.5:
            enemy_dist = float(state.get("enemy_dist", max_dist))
            if enemy_dist > 34.0:
                move = max(move, 0.72)
            elif enemy_dist < 16.0:
                if abs(enemy_bearing_norm) < 0.18:
                    move = min(move, -0.22)
                else:
                    move = max(move, 0.05)
            else:
                move = min(max(move, 0.15), 0.48)
            if abs(enemy_bearing_norm) > 0.60:
                move = min(move, 0.22)

        if self._stuck_ticks > 14 and enemy_seen < 0.5:
            turn += 0.35 if ((self._stuck_ticks // 9) % 2 == 0) else -0.35
            move = max(move, 0.90)
        elif nav.avoid_obstacle:
            turn += 0.18 * math.sin(self._recovery_phase)
            move = min(move, 0.45)

        abs_target_b = abs(target_b)
        is_powerup_mode = ("POWERUP" in str(nav.mode).upper()) and enemy_seen < 0.5
        wp_far = float(nav.distance_to_waypoint) > 4.0
        if abs_target_b > 0.55:
            if is_powerup_mode and wp_far:
                move = max(move, 0.12)
            else:
                move = 0.0
        elif abs_target_b > 0.35:
            if is_powerup_mode and wp_far:
                move = min(max(move, 0.12), 0.30)
            else:
                move = min(max(move, 0.06), 0.20)
        elif abs_target_b > 0.20:
            move = max(move, 0.0)

        if move < 0.0 and abs_target_b > 0.20:
            move = 0.0

        enemy_dist = float(state.get("enemy_dist", max_dist))
        allow_reverse = bool(
            enemy_seen > 0.5
            and enemy_dist < 10.0
            and abs(enemy_bearing_norm) < 0.10
            and abs_target_b < 0.12
            and not nav.avoid_obstacle
        )
        if move < 0.0 and not allow_reverse:
            move = self.forward_bias if abs_target_b < 0.25 else 0.0

        prev_turn_big = abs(self._prev_turn_cmd) > 0.16
        cur_turn_big = abs(turn) > 0.16
        turn_flip = prev_turn_big and cur_turn_big and (self._prev_turn_cmd * turn < 0.0)
        if turn_flip and abs_target_b > 0.18 and not nav.avoid_obstacle:
            self._turn_flip_memory = min(8.0, self._turn_flip_memory + 0.70)
        else:
            self._turn_flip_memory = max(0.0, self._turn_flip_memory - 0.22)
        if abs(turn) < 0.08:
            self._turn_flip_memory = max(0.0, self._turn_flip_memory - 0.30)

        turn_penalty_cap = 0.25 if is_powerup_mode else 0.40
        flip_penalty = _clamp(self._turn_flip_memory / 8.0, 0.0, turn_penalty_cap)
        turn *= (1.0 - flip_penalty)

        if is_powerup_mode and wp_far and self._stuck_ticks >= 8 and enemy_seen < 0.5:
            move = max(move, 0.18)
            turn += 0.15 * math.sin(1.3 * self._recovery_phase)

        move = _clamp(move, -1.0, 1.0)
        turn = _clamp(turn, -1.0, 1.0)
        move = _quantize(move, self._MOVE_LEVELS)
        turn = _quantize(turn, self._TURN_LEVELS)
        if is_powerup_mode and wp_far and abs(move) < 1e-6:
            move = 0.18

        self._prev_move_cmd = float(move)
        self._prev_turn_cmd = float(turn)

        return MotionOutput(
            move_speed=float(move),
            heading_rotation_angle=float(turn),
            stuck_ticks=int(self._stuck_ticks),
        )
