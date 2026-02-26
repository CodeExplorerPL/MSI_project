from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import math
import random

from .dqn_neuro_anfis import DQNTurretNeuroANFIS
from .state_encoder import TurretStateEncoder


AMMO_RANGES = {
    "HEAVY": 25.0,
    "LIGHT": 50.0,
    "LONG_DISTANCE": 100.0,
}


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _clip11(v: float) -> float:
    return max(-1.0, min(1.0, float(v)))


def _to_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _signed_angle_diff_deg(target: float, current: float) -> float:
    return ((float(target) - float(current) + 180.0) % 360.0) - 180.0


def _resolve_checkpoint(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    base = Path(__file__).resolve().parent
    return (base / p).resolve()


@dataclass
class TurretRuntimeConfig:
    tank_type: str = "ANY"
    device: str = "auto"
    seed: int = 42
    epsilon: float = 0.0
    scan_action: float = 0.35
    scan_flip_ticks: int = 120
    close_distance: float = 22.0
    close_track_error_deg: float = 2.0
    close_track_action: float = 0.35
    fire_error_deg: float = 4.8
    fire_conf_threshold: float = 0.48
    enable_fire: bool = True


class TurretRuntimeAgent:
    """
    Runtime wrapper for turret-only DQN policy.
    """

    def __init__(
        self,
        checkpoint: str,
        config: Optional[TurretRuntimeConfig] = None,
    ):
        cfg = config if config is not None else TurretRuntimeConfig()
        self.config = cfg
        self.rng = random.Random(int(cfg.seed))

        ckpt = _resolve_checkpoint(checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"Turret checkpoint not found: {ckpt}")

        self._agent, payload = DQNTurretNeuroANFIS.load(str(ckpt), device=str(cfg.device))
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        enc_meta = meta.get("encoder", {}) if isinstance(meta, dict) else {}
        self._encoder = TurretStateEncoder(
            max_enemies=max(1, int(enc_meta.get("max_enemies", 4))),
            max_powerups=max(1, int(enc_meta.get("max_powerups", 4))),
        )
        self.checkpoint_path = str(ckpt)
        self.reset_runtime()

    def reset_runtime(self) -> None:
        self._scan_dir = 1.0
        self._scan_ticks = 0
        self._tick = 0

    @staticmethod
    def _snapshot_has_enemy(sensor_snapshot: Optional[Dict[str, List[Any]]]) -> bool:
        if not isinstance(sensor_snapshot, dict):
            return False
        seen = sensor_snapshot.get("seen_tanks", [])
        return bool(seen)

    def _build_observation(
        self,
        state: Dict[str, float | str],
        sensor_snapshot: Optional[Dict[str, List[Any]]] = None,
    ) -> Dict[str, Any]:
        s = state if isinstance(state, dict) else {}
        map_w = max(1.0, _to_float(s.get("map_w", 200.0), 200.0))
        map_h = max(1.0, _to_float(s.get("map_h", 200.0), 200.0))
        heading = _to_float(s.get("heading", 0.0), 0.0)
        barrel = _to_float(s.get("barrel", 0.0), 0.0)

        seen_opp_state = _to_float(s.get("seen_opp", 0.0), 0.0) > 0.5
        seen_opp = seen_opp_state or self._snapshot_has_enemy(sensor_snapshot)

        enemy_bearing = _to_float(s.get("enemy_bearing", 0.0), 0.0)
        enemy_dist = _to_float(s.get("enemy_dist", math.hypot(map_w, map_h)), math.hypot(map_w, map_h))
        enemy_hp = _to_float(s.get("enemy_hp", 70.0), 70.0)
        enemy_hp_ratio = _clip01(enemy_hp / 100.0)
        enemy_shot_error = _signed_angle_diff_deg(enemy_bearing, barrel)

        seen_pu = _to_float(s.get("seen_pu", 0.0), 0.0) > 0.5
        powerup_bearing = _to_float(s.get("powerup_bearing", 0.0), 0.0)
        powerup_dist = _to_float(s.get("powerup_dist", math.hypot(map_w, map_h)), math.hypot(map_w, map_h))
        powerup_shot_error = _signed_angle_diff_deg(powerup_bearing, barrel)

        obs = {
            "tick": int(self._tick),
            "map": {"width": float(map_w), "height": float(map_h)},
            "self": {
                "x": _to_float(s.get("x", 0.0), 0.0),
                "y": _to_float(s.get("y", 0.0), 0.0),
                "heading": float(heading),
                "barrel": float(barrel),
                "reload": _to_float(s.get("reload", 0.0), 0.0),
                "ammo_loaded": str(s.get("ammo_loaded", "LIGHT")),
                "ammo": {
                    "HEAVY": max(0.0, _to_float(s.get("ammo_h", 0.0), 0.0)),
                    "LIGHT": max(0.0, _to_float(s.get("ammo_l", 0.0), 0.0)),
                    "LONG_DISTANCE": max(0.0, _to_float(s.get("ammo_s", 0.0), 0.0)),
                },
            },
            "enemies": [
                {
                    "id": "enemy_0",
                    "distance": max(0.0, float(enemy_dist)),
                    "bearing_deg": float(enemy_bearing),
                    "shot_error_deg": float(enemy_shot_error),
                    "hp_ratio": float(enemy_hp_ratio),
                    "visible": 1.0 if seen_opp else 0.0,
                }
            ],
            "powerups": [
                {
                    "distance": max(0.0, float(powerup_dist)),
                    "bearing_deg": float(powerup_bearing),
                    "shot_error_deg": float(powerup_shot_error),
                    "visible": 1.0 if seen_pu else 0.0,
                    "is_ammo": 0.0,
                    "ammo_type": "",
                }
            ],
            "switch": {
                "pending": 0.0,
                "ticks_since_switch": 0.0,
                "event_id": 0,
            },
        }
        return obs

    @staticmethod
    def _pick_ammo(
        dist: float,
        ammo_h: float,
        ammo_l: float,
        ammo_s: float,
        current_loaded: str,
        close_distance: float,
    ) -> str:
        d = max(0.0, float(dist))
        loaded = str(current_loaded).upper().strip()

        if d <= float(close_distance):
            for name, count in (("LIGHT", ammo_l), ("HEAVY", ammo_h), ("LONG_DISTANCE", ammo_s)):
                if count > 0.0:
                    return name
        elif d <= 35.0:
            for name, count in (("HEAVY", ammo_h), ("LIGHT", ammo_l), ("LONG_DISTANCE", ammo_s)):
                if count > 0.0:
                    return name
        elif d <= 65.0:
            for name, count in (("LIGHT", ammo_l), ("LONG_DISTANCE", ammo_s), ("HEAVY", ammo_h)):
                if count > 0.0:
                    return name
        else:
            for name, count in (("LONG_DISTANCE", ammo_s), ("LIGHT", ammo_l), ("HEAVY", ammo_h)):
                if count > 0.0:
                    return name

        if loaded in ("HEAVY", "LIGHT", "LONG_DISTANCE"):
            return loaded
        return "LIGHT"

    def act_from_state(
        self,
        state: Dict[str, float | str],
        epsilon: Optional[float] = None,
        sensor_snapshot: Optional[Dict[str, List[Any]]] = None,
    ) -> Dict[str, Any]:
        self._tick += 1
        obs = self._build_observation(state=state, sensor_snapshot=sensor_snapshot)
        features = self._encoder.encode(obs)
        eps = _clip01(self.config.epsilon if epsilon is None else float(epsilon))
        _idx, barrel_action, _diag = self._agent.select_action(features, eps)
        barrel_cmd = _clip11(float(barrel_action))

        me = obs.get("self", {})
        enemy = (obs.get("enemies") or [{}])[0]
        ammo = me.get("ammo", {}) if isinstance(me, dict) else {}
        ammo_h = max(0.0, _to_float(ammo.get("HEAVY", 0.0), 0.0))
        ammo_l = max(0.0, _to_float(ammo.get("LIGHT", 0.0), 0.0))
        ammo_s = max(0.0, _to_float(ammo.get("LONG_DISTANCE", 0.0), 0.0))

        visible_enemy = _to_float(enemy.get("visible", 0.0), 0.0) > 0.5
        enemy_dist = max(0.0, _to_float(enemy.get("distance", 1e9), 1e9))
        enemy_err = abs(_to_float(enemy.get("shot_error_deg", 180.0), 180.0))
        reload_ready = _to_float(me.get("reload", 0.0), 0.0) <= 1e-3
        ammo_loaded = str(me.get("ammo_loaded", "LIGHT"))

        if not visible_enemy:
            self._scan_ticks += 1
            if self._scan_ticks % max(10, int(self.config.scan_flip_ticks)) == 0:
                self._scan_dir *= -1.0
            if abs(barrel_cmd) < float(self.config.scan_action):
                barrel_cmd = self._scan_dir * float(self.config.scan_action)
        else:
            self._scan_ticks = 0
            if (
                enemy_dist <= float(self.config.close_distance)
                and enemy_err >= float(self.config.close_track_error_deg)
                and abs(barrel_cmd) < float(self.config.close_track_action)
            ):
                sign = 1.0 if _to_float(enemy.get("shot_error_deg", 0.0), 0.0) > 0.0 else -1.0
                barrel_cmd = sign * float(self.config.close_track_action)

        ammo_to_load = self._pick_ammo(
            dist=enemy_dist,
            ammo_h=ammo_h,
            ammo_l=ammo_l,
            ammo_s=ammo_s,
            current_loaded=ammo_loaded,
            close_distance=float(self.config.close_distance),
        )
        selected_count = {"HEAVY": ammo_h, "LIGHT": ammo_l, "LONG_DISTANCE": ammo_s}.get(ammo_to_load, 0.0)
        ammo_range = AMMO_RANGES.get(ammo_to_load, 50.0)
        range_ok = enemy_dist <= ammo_range

        align = _clip01(1.0 - enemy_err / max(1.0, float(self.config.fire_error_deg)))
        range_quality = _clip01(1.0 - enemy_dist / max(1.0, ammo_range))
        confidence = _clip01((0.75 * align) + (0.25 * range_quality))

        should_fire = bool(
            bool(self.config.enable_fire)
            and visible_enemy
            and reload_ready
            and selected_count > 0.0
            and range_ok
            and enemy_err <= float(self.config.fire_error_deg)
            and (
                confidence >= float(self.config.fire_conf_threshold)
                or (
                    enemy_dist <= float(self.config.close_distance)
                    and enemy_err <= 5.0
                )
            )
        )

        return {
            "barrel_rotation_angle": float(barrel_cmd),
            "ammo_to_load": str(ammo_to_load),
            "should_fire": bool(should_fire),
            "target_type": "ENEMY" if visible_enemy else "NONE",
            "aim_error": float(enemy_err),
            "confidence": float(confidence),
            "target_visible": bool(visible_enemy),
            "epsilon": float(eps),
            "checkpoint": self.checkpoint_path,
        }
