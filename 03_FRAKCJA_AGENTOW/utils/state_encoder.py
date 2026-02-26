from __future__ import annotations

from typing import Any, Dict, List
import math


def _clip(v: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(v)))


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _sin_cos_deg(angle_deg: float) -> tuple[float, float]:
    r = math.radians(float(angle_deg))
    return math.sin(r), math.cos(r)


class TurretStateEncoder:
    """
    Deterministic encoder matching the 81D turret checkpoint format.

    Layout:
    - 21 global/self/switch features
    - 4 enemy slots x 8 features
    - 4 powerup slots x 7 features
    """

    def __init__(self, max_enemies: int = 4, max_powerups: int = 4):
        self.max_enemies = max(1, int(max_enemies))
        self.max_powerups = max(1, int(max_powerups))
        self.state_dim = 81

    def _encode_global(self, obs: Dict[str, Any], diag: float) -> List[float]:
        m = obs.get("map", {}) if isinstance(obs.get("map", {}), dict) else {}
        s = obs.get("self", {}) if isinstance(obs.get("self", {}), dict) else {}
        sw = obs.get("switch", {}) if isinstance(obs.get("switch", {}), dict) else {}

        map_w = max(1.0, _to_float(m.get("width", 200.0), 200.0))
        map_h = max(1.0, _to_float(m.get("height", 200.0), 200.0))
        x = _to_float(s.get("x", 0.0), 0.0)
        y = _to_float(s.get("y", 0.0), 0.0)
        heading = _to_float(s.get("heading", 0.0), 0.0)
        barrel = _to_float(s.get("barrel", 0.0), 0.0)
        reload_now = _to_float(s.get("reload", 0.0), 0.0)
        ammo_loaded = str(s.get("ammo_loaded", "LIGHT")).upper()
        ammo = s.get("ammo", {}) if isinstance(s.get("ammo", {}), dict) else {}
        ammo_h = _to_float(ammo.get("HEAVY", 0.0), 0.0)
        ammo_l = _to_float(ammo.get("LIGHT", 0.0), 0.0)
        ammo_s = _to_float(ammo.get("LONG_DISTANCE", 0.0), 0.0)

        tick = _to_float(obs.get("tick", 0.0), 0.0)
        tick_phase = tick / 200.0
        sw_pending = _to_float(sw.get("pending", 0.0), 0.0)
        sw_t = _to_float(sw.get("ticks_since_switch", 0.0), 0.0)
        sw_event = _to_float(sw.get("event_id", 0.0), 0.0)

        hd_s, hd_c = _sin_cos_deg(heading)
        br_s, br_c = _sin_cos_deg(barrel)
        ev_s, ev_c = _sin_cos_deg(sw_event * 30.0)

        loaded_h = 1.0 if ammo_loaded == "HEAVY" else 0.0
        loaded_l = 1.0 if ammo_loaded == "LIGHT" else 0.0
        loaded_s = 1.0 if ammo_loaded == "LONG_DISTANCE" else 0.0

        v = [
            _clip((map_w / 200.0) - 1.0, -1.0, 1.0),
            _clip((map_h / 200.0) - 1.0, -1.0, 1.0),
            _clip(math.sin(tick_phase), -1.0, 1.0),
            _clip(math.cos(tick_phase), -1.0, 1.0),
            _clip((x / map_w) * 2.0 - 1.0, -1.0, 1.0),
            _clip((y / map_h) * 2.0 - 1.0, -1.0, 1.0),
            _clip(hd_s, -1.0, 1.0),
            _clip(hd_c, -1.0, 1.0),
            _clip(br_s, -1.0, 1.0),
            _clip(br_c, -1.0, 1.0),
            _clip(reload_now / 10.0, 0.0, 1.0),
            loaded_h,
            loaded_l,
            loaded_s,
            _clip(ammo_h / 10.0, 0.0, 1.0),
            _clip(ammo_l / 20.0, 0.0, 1.0),
            _clip(ammo_s / 10.0, 0.0, 1.0),
            _clip(sw_pending, 0.0, 1.0),
            _clip(sw_t / 120.0, 0.0, 1.0),
            _clip(ev_s, -1.0, 1.0),
            _clip(ev_c, -1.0, 1.0),
        ]
        return v

    def _encode_enemy_slots(self, obs: Dict[str, Any], diag: float) -> List[float]:
        slots: List[float] = []
        enemies = obs.get("enemies", [])
        if not isinstance(enemies, list):
            enemies = []
        enemies_sorted = sorted(
            enemies,
            key=lambda e: _to_float((e or {}).get("distance", 1e9), 1e9),
        )

        for idx in range(self.max_enemies):
            e = enemies_sorted[idx] if idx < len(enemies_sorted) else {}
            if not isinstance(e, dict):
                e = {}
            vis = _to_float(e.get("visible", 0.0), 0.0)
            dist = _to_float(e.get("distance", 1e9), 1e9)
            bearing = _to_float(e.get("bearing_deg", 0.0), 0.0)
            shot_err = _to_float(e.get("shot_error_deg", 0.0), 0.0)
            hp = _to_float(e.get("hp_ratio", 0.0), 0.0)

            b_s, b_c = _sin_cos_deg(bearing)
            s_s, s_c = _sin_cos_deg(shot_err)
            slot = [
                _clip(vis, 0.0, 1.0),
                _clip(dist / max(1.0, diag), 0.0, 1.0),
                _clip(b_s, -1.0, 1.0),
                _clip(b_c, -1.0, 1.0),
                _clip(s_s, -1.0, 1.0),
                _clip(s_c, -1.0, 1.0),
                _clip(hp, 0.0, 1.0),
                1.0 if idx == 0 and vis > 0.5 else 0.0,
            ]
            slots.extend(slot)
        return slots

    def _encode_powerup_slots(self, obs: Dict[str, Any], diag: float) -> List[float]:
        slots: List[float] = []
        powerups = obs.get("powerups", [])
        if not isinstance(powerups, list):
            powerups = []
        powerups_sorted = sorted(
            powerups,
            key=lambda p: _to_float((p or {}).get("distance", 1e9), 1e9),
        )

        for idx in range(self.max_powerups):
            p = powerups_sorted[idx] if idx < len(powerups_sorted) else {}
            if not isinstance(p, dict):
                p = {}
            vis = _to_float(p.get("visible", 0.0), 0.0)
            dist = _to_float(p.get("distance", 1e9), 1e9)
            bearing = _to_float(p.get("bearing_deg", 0.0), 0.0)
            shot_err = _to_float(p.get("shot_error_deg", 0.0), 0.0)
            is_ammo = _to_float(p.get("is_ammo", 0.0), 0.0)

            b_s, b_c = _sin_cos_deg(bearing)
            s_s, s_c = _sin_cos_deg(shot_err)
            slot = [
                _clip(vis, 0.0, 1.0),
                _clip(dist / max(1.0, diag), 0.0, 1.0),
                _clip(b_s, -1.0, 1.0),
                _clip(b_c, -1.0, 1.0),
                _clip(s_s, -1.0, 1.0),
                _clip(s_c, -1.0, 1.0),
                _clip(is_ammo, 0.0, 1.0),
            ]
            slots.extend(slot)
        return slots

    def encode(self, obs: Dict[str, Any]) -> List[float]:
        m = obs.get("map", {}) if isinstance(obs.get("map", {}), dict) else {}
        map_w = max(1.0, _to_float(m.get("width", 200.0), 200.0))
        map_h = max(1.0, _to_float(m.get("height", 200.0), 200.0))
        diag = math.hypot(map_w, map_h)

        features: List[float] = []
        features.extend(self._encode_global(obs, diag))
        features.extend(self._encode_enemy_slots(obs, diag))
        features.extend(self._encode_powerup_slots(obs, diag))

        if len(features) < self.state_dim:
            features.extend([0.0] * (self.state_dim - len(features)))
        elif len(features) > self.state_dim:
            features = features[: self.state_dim]

        return [float(_clip(v, -1.0, 1.0)) for v in features]

