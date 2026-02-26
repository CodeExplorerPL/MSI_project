from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import math

from .controller_types import NavigatorOutput, WeaponOutput


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class TSK_WeaponController:
    """
    Fuzzy-like weapon module.
    Controls:
    - fire decision
    - ammo switching
    - turret correction (barrel rotation), normalized in [-1, 1]
    """

    def __init__(self, weights: Dict[str, Any] | None = None) -> None:
        cfg = dict((weights or {}).get("weapon", {}))

        self.sweep_speed = float(cfg.get("sweep_speed", 0.12))
        self.enemy_memory_ttl = max(1, int(cfg.get("enemy_memory_ttl", 14)))
        self.force_fire_aim_deg = float(cfg.get("force_fire_aim_deg", 3.0))
        self.fire_score_threshold = float(cfg.get("fire_score_threshold", 0.24))

        self._sweep_phase = 0.0
        self._last_enemy_pos: Optional[Tuple[float, float]] = None
        self._last_enemy_age: int = 10_000
        self._last_ammo_cmd: str = "LIGHT"
        self._ammo_lock_ticks: int = 0

    @staticmethod
    def _signed_angle_diff_deg(target: float, current: float) -> float:
        return ((target - current + 180.0) % 360.0) - 180.0

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    def _nearest_enemy(
        self,
        state: Dict[str, float | str],
        sensor_snapshot: Dict[str, List[Tuple[float, float]]],
    ) -> Optional[Tuple[float, float]]:
        seen_tanks = list(sensor_snapshot.get("seen_tanks", []))
        if not seen_tanks:
            return None
        px = float(state.get("x", 0.0))
        py = float(state.get("y", 0.0))
        return min(
            seen_tanks,
            key=lambda p: math.hypot(float(p[0]) - px, float(p[1]) - py),
        )

    def _pick_ammo(
        self,
        state: Dict[str, float | str],
        enemy_visible: bool,
        enemy_dist: float,
    ) -> str:
        current_loaded = str(state.get("ammo_loaded", "LIGHT")).upper()
        reload_now = float(state.get("reload", 0.0))
        ammo_h = float(state.get("ammo_h", 0.0))
        ammo_l = float(state.get("ammo_l", 0.0))
        ammo_s = float(state.get("ammo_s", 0.0))
        has_h = ammo_h > 0.0
        has_l = ammo_l > 0.0
        has_s = ammo_s > 0.0

        def available(name: str) -> bool:
            n = str(name).upper()
            if n == "HEAVY":
                return has_h
            if n == "LIGHT":
                return has_l
            if n == "LONG_DISTANCE":
                return has_s
            return False

        def fallback_for_distance(d: float) -> str:
            if d <= 24.0:
                order = ("HEAVY", "LIGHT", "LONG_DISTANCE")
            elif d >= 62.0:
                order = ("LONG_DISTANCE", "LIGHT", "HEAVY")
            else:
                order = ("LIGHT", "HEAVY", "LONG_DISTANCE")
            for n in order:
                if available(n):
                    return n
            return current_loaded

        def ammo_suits_distance(name: str, d: float) -> bool:
            n = str(name).upper()
            if n == "HEAVY":
                return d <= 30.0
            if n == "LIGHT":
                return 18.0 <= d <= 75.0
            if n == "LONG_DISTANCE":
                return d >= 52.0
            return False

        if self._ammo_lock_ticks > 0:
            self._ammo_lock_ticks -= 1

        if reload_now > 1e-3:
            if available(current_loaded):
                return current_loaded
            return fallback_for_distance(enemy_dist)

        if enemy_visible:
            preferred = fallback_for_distance(enemy_dist)
            if (
                self._ammo_lock_ticks > 0
                and available(current_loaded)
                and ammo_suits_distance(current_loaded, enemy_dist)
            ):
                return current_loaded
            if available(current_loaded) and ammo_suits_distance(current_loaded, enemy_dist):
                return current_loaded
            if preferred != current_loaded:
                self._ammo_lock_ticks = 5
            return preferred

        if available(current_loaded):
            return current_loaded
        return fallback_for_distance(enemy_dist)

    def decide(
        self,
        state: Dict[str, float | str],
        sensor_snapshot: Dict[str, List[Tuple[float, float]]],
        nav: NavigatorOutput,
    ) -> WeaponOutput:
        self._sweep_phase += self.sweep_speed

        heading = float(state.get("heading", 0.0))
        barrel = float(state.get("barrel", 0.0))
        px = float(state.get("x", 0.0))
        py = float(state.get("y", 0.0))
        reload_now = float(state.get("reload", 0.0))
        reload_ready = reload_now <= 1e-3

        nearest_enemy = self._nearest_enemy(state=state, sensor_snapshot=sensor_snapshot)
        state_seen_enemy = float(state.get("seen_opp", 0.0)) > 0.5
        enemy_visible = bool(state_seen_enemy or nearest_enemy is not None)
        shot_error_signed = 0.0

        if enemy_visible:
            enemy_dist = float(state.get("enemy_dist", 1e9))
            enemy_rel = float(state.get("enemy_bearing", 0.0))
            shot_error_signed = self._signed_angle_diff_deg(enemy_rel, barrel)

            if nearest_enemy is not None:
                self._last_enemy_pos = (float(nearest_enemy[0]), float(nearest_enemy[1]))
                self._last_enemy_age = 0
            elif enemy_dist < 1e8:
                enemy_world = heading + enemy_rel
                ex = px + enemy_dist * math.cos(math.radians(enemy_world))
                ey = py + enemy_dist * math.sin(math.radians(enemy_world))
                self._last_enemy_pos = (float(ex), float(ey))
                self._last_enemy_age = 0
        else:
            self._last_enemy_age += 1
            enemy_dist = float(state.get("enemy_dist", 1e9))
            if self._last_enemy_pos is not None and self._last_enemy_age <= self.enemy_memory_ttl:
                enemy_world = math.degrees(
                    math.atan2(float(self._last_enemy_pos[1]) - py, float(self._last_enemy_pos[0]) - px)
                )
                enemy_dist = self._dist((px, py), self._last_enemy_pos)
                shot_error_signed = self._signed_angle_diff_deg(enemy_world, heading + barrel)
            else:
                nav_shot_error = self._signed_angle_diff_deg(float(nav.desired_heading_deg), heading + barrel)
                shot_error_signed = 0.75 * nav_shot_error + 12.0 * math.sin(self._sweep_phase)

        aim_error = abs(shot_error_signed)
        if float(enemy_dist) <= 22.0:
            denom = 15.0
        elif float(enemy_dist) <= 60.0:
            denom = 21.0
        else:
            denom = 28.0

        barrel_cmd = _clamp(shot_error_signed / denom, -1.0, 1.0)
        if aim_error < 0.9:
            barrel_cmd = 0.0
        elif aim_error < 4.0:
            barrel_cmd *= 0.45

        ammo_to_load = self._pick_ammo(
            state=state,
            enemy_visible=enemy_visible,
            enemy_dist=float(enemy_dist),
        )
        self._last_ammo_cmd = str(ammo_to_load)

        fire_score = 0.0
        should_fire = False
        if enemy_visible and reload_ready:
            align = _clamp(1.0 - float(aim_error) / 42.0, 0.0, 1.0)
            if ammo_to_load == "HEAVY":
                ideal = 16.0
                sigma = 12.0
                if float(enemy_dist) < 18.0:
                    fire_angle = 13.0
                elif float(enemy_dist) < 38.0:
                    fire_angle = 10.0
                else:
                    fire_angle = 8.0
            elif ammo_to_load == "LONG_DISTANCE":
                ideal = 85.0
                sigma = 35.0
                if float(enemy_dist) < 45.0:
                    fire_angle = 8.0
                elif float(enemy_dist) < 90.0:
                    fire_angle = 6.5
                else:
                    fire_angle = 5.5
            else:
                ideal = 42.0
                sigma = 26.0
                if float(enemy_dist) < 25.0:
                    fire_angle = 11.0
                elif float(enemy_dist) < 62.0:
                    fire_angle = 9.0
                else:
                    fire_angle = 7.0

            z = (float(enemy_dist) - ideal) / max(1e-6, sigma)
            range_fit = math.exp(-0.5 * z * z)
            fire_score = 0.72 * align + 0.28 * range_fit
            if float(aim_error) <= self.force_fire_aim_deg:
                should_fire = True
            else:
                should_fire = bool(float(aim_error) <= fire_angle and fire_score >= self.fire_score_threshold)

        return WeaponOutput(
            barrel_rotation_angle=float(_clamp(barrel_cmd, -1.0, 1.0)),
            ammo_to_load=str(ammo_to_load),
            should_fire=bool(should_fire),
            fire_score=float(_clamp(fire_score, 0.0, 1.0)),
            aim_error_deg=float(_clamp(aim_error, 0.0, 180.0)),
            target_visible=bool(enemy_visible),
        )
