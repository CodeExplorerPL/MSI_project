from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import math


@dataclass
class _Point:
    x: float
    y: float


@dataclass
class _Obstacle:
    _position: _Point
    _size: List[float]
    is_alive: bool = True


@dataclass
class _Terrain:
    _position: _Point
    _size: List[float]
    _movement_speed_modifier: float
    _deal_damage: float


@dataclass
class _MemoryObstacle:
    x: float
    y: float
    size: float
    ttl: int


@dataclass
class _MemoryTerrain:
    x: float
    y: float
    size: float
    speed_modifier: float
    dmg: float
    ttl: int


class RuntimeMapMemory:
    """
    Runtime map memory built only from sensor_data.
    Keeps short-term obstacle/terrain memory so A* can plan around seen map parts.
    """

    def __init__(
        self,
        map_w: float = 200.0,
        map_h: float = 200.0,
        obstacle_ttl: int = 1200,
        terrain_ttl: int = 900,
        obstacle_default_size: float = 9.0,
        obstacle_size_by_type: Dict[str, float] | None = None,
        terrain_patch_size: float = 10.0,
    ) -> None:
        self.size = [float(map_w), float(map_h)]
        self._size = list(self.size)

        self.obstacle_ttl = max(1, int(obstacle_ttl))
        self.terrain_ttl = max(1, int(terrain_ttl))
        self.obstacle_default_size = max(2.0, float(obstacle_default_size))
        self.terrain_patch_size = max(2.0, float(terrain_patch_size))

        self.obstacle_size_by_type = {
            str(k).upper(): float(v) for k, v in (obstacle_size_by_type or {}).items()
        }

        self._obstacles: Dict[str, _MemoryObstacle] = {}
        self._terrains: Dict[str, _MemoryTerrain] = {}
        self.version: int = 0

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    @staticmethod
    def _pos_xy(raw: Any) -> Tuple[float, float]:
        if isinstance(raw, dict):
            return float(raw.get("x", 0.0)), float(raw.get("y", 0.0))
        if hasattr(raw, "x") and hasattr(raw, "y"):
            return float(raw.x), float(raw.y)
        if isinstance(raw, (tuple, list)) and len(raw) >= 2:
            return float(raw[0]), float(raw[1])
        return 0.0, 0.0

    def _obstacle_size(self, obstacle_type: str) -> float:
        key = str(obstacle_type).upper().strip()
        if key in self.obstacle_size_by_type:
            return float(self.obstacle_size_by_type[key])
        return float(self.obstacle_default_size)

    @staticmethod
    def _obstacle_key(item: Dict[str, Any]) -> str:
        oid = str(item.get("id", "")).strip()
        if oid:
            return f"id:{oid}"
        pos = item.get("position", {})
        x = round(float(pos.get("x", 0.0)), 1)
        y = round(float(pos.get("y", 0.0)), 1)
        t = str(item.get("type", "UNK")).upper()
        return f"{t}:{x}:{y}"

    @staticmethod
    def _terrain_key(item: Dict[str, Any]) -> str:
        pos = item.get("position", {})
        x = round(float(pos.get("x", 0.0)), 1)
        y = round(float(pos.get("y", 0.0)), 1)
        t = str(item.get("type", "TERRAIN"))
        return f"{t}:{x}:{y}"

    def _decay(self) -> bool:
        changed = False
        remove_obs = []
        for k, obs in self._obstacles.items():
            obs.ttl -= 1
            if obs.ttl <= 0:
                remove_obs.append(k)
        for k in remove_obs:
            del self._obstacles[k]
            changed = True

        remove_ter = []
        for k, ter in self._terrains.items():
            ter.ttl -= 1
            if ter.ttl <= 0:
                remove_ter.append(k)
        for k in remove_ter:
            del self._terrains[k]
            changed = True

        return changed

    def update(self, sensor_data: Dict[str, Any], my_position: Tuple[float, float] | None = None) -> bool:
        changed = self._decay()

        seen_obstacles = list((sensor_data or {}).get("seen_obstacles", []) or [])
        seen_terrains = list((sensor_data or {}).get("seen_terrains", []) or [])

        for obs in seen_obstacles:
            if not isinstance(obs, dict):
                continue
            key = self._obstacle_key(obs)
            pos = obs.get("position", {})
            x = self._safe_float(pos.get("x", 0.0), 0.0)
            y = self._safe_float(pos.get("y", 0.0), 0.0)
            obs_type = str(obs.get("type", "")).upper()
            size = self._obstacle_size(obs_type)
            prev = self._obstacles.get(key)
            self._obstacles[key] = _MemoryObstacle(x=float(x), y=float(y), size=float(size), ttl=self.obstacle_ttl)
            if prev is None or abs(prev.x - x) > 1e-3 or abs(prev.y - y) > 1e-3:
                changed = True

        for ter in seen_terrains:
            if not isinstance(ter, dict):
                continue
            key = self._terrain_key(ter)
            pos = ter.get("position", {})
            x = self._safe_float(pos.get("x", 0.0), 0.0)
            y = self._safe_float(pos.get("y", 0.0), 0.0)
            speed_modifier = self._safe_float(ter.get("speed_modifier", 1.0), 1.0)
            dmg = self._safe_float(ter.get("dmg", 0.0), 0.0)
            prev = self._terrains.get(key)
            self._terrains[key] = _MemoryTerrain(
                x=float(x),
                y=float(y),
                size=float(self.terrain_patch_size),
                speed_modifier=float(speed_modifier),
                dmg=float(dmg),
                ttl=self.terrain_ttl,
            )
            if prev is None:
                changed = True

        if changed:
            self.version += 1
        return changed

    @property
    def obstacle_list(self) -> List[_Obstacle]:
        return [
            _Obstacle(_position=_Point(o.x, o.y), _size=[o.size, o.size], is_alive=True)
            for o in self._obstacles.values()
        ]

    @property
    def terrain_list(self) -> List[_Terrain]:
        return [
            _Terrain(
                _position=_Point(t.x, t.y),
                _size=[t.size, t.size],
                _movement_speed_modifier=float(t.speed_modifier),
                _deal_damage=float(t.dmg),
            )
            for t in self._terrains.values()
        ]
