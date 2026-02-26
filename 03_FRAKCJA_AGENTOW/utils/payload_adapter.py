from __future__ import annotations

from typing import Any, Dict, List, Tuple
import math

from .controller_types import AgentAction


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _signed_angle_diff_deg(target: float, current: float) -> float:
    return ((float(target) - float(current) + 180.0) % 360.0) - 180.0


def normalize_tank_type(raw: Any) -> str:
    # Competition packaging mode: treat every tank type as LIGHT.
    s = str(raw or "LIGHT").strip().upper()
    if "." in s:
        s = s.split(".")[-1]
    _ = s
    return "LIGHT"


def _dict_get_xy(raw: Any) -> Tuple[float, float]:
    if isinstance(raw, dict):
        return _safe_float(raw.get("x", 0.0)), _safe_float(raw.get("y", 0.0))
    return 0.0, 0.0


def _nearest_point(src: Tuple[float, float], points: List[Tuple[float, float]]) -> Tuple[float, float] | None:
    if not points:
        return None
    return min(points, key=lambda p: math.hypot(float(p[0]) - src[0], float(p[1]) - src[1]))


def build_controller_inputs(payload: Dict[str, Any], weights: Dict[str, Any]) -> Tuple[Dict[str, float | str], Dict[str, List[Tuple[float, float]]], str]:
    my = dict(payload.get("my_tank_status", {}) or {})
    sensor = dict(payload.get("sensor_data", {}) or {})

    tank_type = normalize_tank_type(my.get("_tank_type"))

    pos = my.get("position", {})
    x, y = _dict_get_xy(pos)
    heading = _safe_float(my.get("heading", 0.0), 0.0)
    barrel = _safe_float(my.get("barrel_angle", 0.0), 0.0)

    ammo = dict(my.get("ammo", {}) or {})
    ammo_h = _safe_float((ammo.get("HEAVY") or {}).get("count", 0.0), 0.0)
    ammo_l = _safe_float((ammo.get("LIGHT") or {}).get("count", 0.0), 0.0)
    ammo_s = _safe_float((ammo.get("LONG_DISTANCE") or {}).get("count", 0.0), 0.0)

    seen_tanks_raw = list(sensor.get("seen_tanks", []) or [])
    seen_powerups_raw = list(sensor.get("seen_powerups", []) or [])
    seen_terrains_raw = list(sensor.get("seen_terrains", []) or [])

    seen_tanks_xy: List[Tuple[float, float]] = []
    for t in seen_tanks_raw:
        if not isinstance(t, dict):
            continue
        tx, ty = _dict_get_xy(t.get("position", {}))
        seen_tanks_xy.append((tx, ty))

    seen_powerups_xy: List[Tuple[float, float]] = []
    for p in seen_powerups_raw:
        if not isinstance(p, dict):
            continue
        px, py = _dict_get_xy(p.get("position", {}))
        seen_powerups_xy.append((px, py))

    cur_pos = (x, y)
    nearest_enemy = _nearest_point(cur_pos, seen_tanks_xy)
    nearest_powerup = _nearest_point(cur_pos, seen_powerups_xy)

    seen_opp = 1.0 if nearest_enemy is not None else 0.0
    enemy_dist = 1e9
    enemy_bearing = 0.0
    aim_error = 180.0
    if nearest_enemy is not None:
        ex, ey = nearest_enemy
        enemy_dist = math.hypot(ex - x, ey - y)
        enemy_world = math.degrees(math.atan2(ey - y, ex - x))
        enemy_bearing = _signed_angle_diff_deg(enemy_world, heading)
        aim_error = abs(_signed_angle_diff_deg(enemy_bearing, barrel))

    seen_pu = 1.0 if nearest_powerup is not None else 0.0
    powerup_dist = 1e9
    powerup_bearing = 0.0
    if nearest_powerup is not None:
        px, py = nearest_powerup
        powerup_dist = math.hypot(px - x, py - y)
        powerup_world = math.degrees(math.atan2(py - y, px - x))
        powerup_bearing = _signed_angle_diff_deg(powerup_world, heading)

    terrain_speed = 1.0
    terrain_dmg = 0.0
    min_terrain_d = 1e9
    for t in seen_terrains_raw:
        if not isinstance(t, dict):
            continue
        tx, ty = _dict_get_xy(t.get("position", {}))
        d = math.hypot(tx - x, ty - y)
        if d < min_terrain_d:
            min_terrain_d = d
            terrain_speed = _safe_float(t.get("speed_modifier", 1.0), 1.0)
            terrain_dmg = _safe_float(t.get("dmg", 0.0), 0.0)

    map_cfg = dict((weights or {}).get("map", {}))
    map_w = _safe_float(map_cfg.get("width", 200.0), 200.0)
    map_h = _safe_float(map_cfg.get("height", 200.0), 200.0)

    state: Dict[str, float | str] = {
        "x": float(x),
        "y": float(y),
        "heading": float(heading),
        "barrel": float(barrel),
        "top_speed": _safe_float(my.get("_top_speed", 3.0), 3.0),
        "map_w": float(map_w),
        "map_h": float(map_h),
        "reload": _safe_float(my.get("_reload_timer", 0.0), 0.0),
        "ammo_loaded": str(my.get("ammo_loaded") or "LIGHT"),
        "ammo_h": float(ammo_h),
        "ammo_l": float(ammo_l),
        "ammo_s": float(ammo_s),
        "seen_opp": float(seen_opp),
        "enemy_dist": float(enemy_dist),
        "enemy_bearing": float(enemy_bearing),
        "seen_pu": float(seen_pu),
        "powerup_dist": float(powerup_dist),
        "powerup_bearing": float(powerup_bearing),
        "terrain_speed": float(terrain_speed),
        "terrain_dmg": float(terrain_dmg),
        "aim_error": float(aim_error),
    }

    sensor_snapshot = {
        "seen_tanks": seen_tanks_xy,
        "seen_powerups": seen_powerups_xy,
    }

    return state, sensor_snapshot, tank_type


def _by_tank_type(config_map: Dict[str, Any], tank_type: str, default: float) -> float:
    if not isinstance(config_map, dict):
        return float(default)
    key = normalize_tank_type(tank_type)
    raw = config_map.get(key, config_map.get(key.lower(), default))
    return _safe_float(raw, default)


def to_engine_action(
    action: AgentAction,
    state: Dict[str, float | str],
    tank_type: str,
    weights: Dict[str, Any],
    prev: Dict[str, float],
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    control = dict((weights or {}).get("control", {}))

    move_scale = _safe_float(control.get("move_speed_scale", 1.0), 1.0)
    top_speed = _safe_float(state.get("top_speed", 3.0), 3.0)
    target_move_speed = _clamp(float(action.move_speed), -1.0, 1.0) * top_speed * move_scale

    speed_slew = _by_tank_type(control.get("speed_slew_per_tick_by_type", {}), tank_type, max(0.4, top_speed * 0.2))
    move_speed = _clamp(
        target_move_speed,
        _safe_float(prev.get("move_speed", 0.0), 0.0) - speed_slew,
        _safe_float(prev.get("move_speed", 0.0), 0.0) + speed_slew,
    )

    max_heading_delta = _by_tank_type(control.get("max_heading_delta_deg_by_type", {}), tank_type, 5.0)
    heading_target = _clamp(float(action.heading_rotation_angle), -1.0, 1.0) * max_heading_delta
    heading_slew = _safe_float(control.get("heading_slew_deg_per_tick", 2.0), 2.0)
    heading_delta = _clamp(
        heading_target,
        _safe_float(prev.get("heading_delta", 0.0), 0.0) - heading_slew,
        _safe_float(prev.get("heading_delta", 0.0), 0.0) + heading_slew,
    )

    max_barrel_delta = _by_tank_type(control.get("max_barrel_delta_deg_by_type", {}), tank_type, 5.0)
    barrel_target = _clamp(float(action.barrel_rotation_angle), -1.0, 1.0) * max_barrel_delta
    barrel_slew = _safe_float(control.get("barrel_slew_deg_per_tick", 3.0), 3.0)
    barrel_delta = _clamp(
        barrel_target,
        _safe_float(prev.get("barrel_delta", 0.0), 0.0) - barrel_slew,
        _safe_float(prev.get("barrel_delta", 0.0), 0.0) + barrel_slew,
    )

    should_fire = bool(action.should_fire)
    fire_hold_error = _safe_float(control.get("fire_stabilize_aim_error_deg", 7.0), 7.0)
    if should_fire:
        seen_opp = _safe_float(state.get("seen_opp", 0.0), 0.0)
        aim_error = abs(_safe_float(state.get("aim_error", 180.0), 180.0))
        if seen_opp <= 0.5 or aim_error > fire_hold_error:
            should_fire = False
        else:
            heading_delta = 0.0
            barrel_delta = 0.0

    ammo_to_load = str(action.ammo_to_load).upper().strip()
    if ammo_to_load not in {"HEAVY", "LIGHT", "LONG_DISTANCE"}:
        ammo_to_load = None

    engine_action = {
        "barrel_rotation_angle": float(barrel_delta),
        "heading_rotation_angle": float(heading_delta),
        "move_speed": float(move_speed),
        "ammo_to_load": ammo_to_load,
        "should_fire": bool(should_fire),
    }

    new_prev = {
        "move_speed": float(move_speed),
        "heading_delta": float(heading_delta),
        "barrel_delta": float(barrel_delta),
    }
    return engine_action, new_prev
