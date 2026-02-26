from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import math


GridCell = Tuple[int, int]


@dataclass
class AStarResult:
    path_cells: List[GridCell]
    reached: bool


class GridMapAStar:
    """
    A* planner on a coarse map grid.
    map_info interface:
    - size or _size => [map_w, map_h]
    - obstacle_list objects with _position(x,y), _size(w,h)
    - terrain_list objects with _position(x,y), _size(w,h), _movement_speed_modifier, _deal_damage
    """

    def __init__(self, map_info: Any, cell_size: float = 10.0):
        self.map_info = map_info
        self.cell_size = max(2.0, float(cell_size))

        map_size = getattr(map_info, "_size", None) or getattr(map_info, "size", [200, 200])
        self.map_w = float(map_size[0]) if len(map_size) > 0 else 200.0
        self.map_h = float(map_size[1]) if len(map_size) > 1 else 200.0

        self.grid_w = max(1, int(math.ceil(self.map_w / self.cell_size)))
        self.grid_h = max(1, int(math.ceil(self.map_h / self.cell_size)))

        self.passable: List[List[bool]] = [[True for _ in range(self.grid_h)] for _ in range(self.grid_w)]
        self.cost: List[List[float]] = [[1.0 for _ in range(self.grid_h)] for _ in range(self.grid_w)]

        self._rasterize_terrains()
        self._rasterize_obstacles()

    def _world_rect_to_cells(self, center_x: float, center_y: float, size_w: float, size_h: float) -> Iterable[GridCell]:
        min_x = center_x - size_w * 0.5
        max_x = center_x + size_w * 0.5
        min_y = center_y - size_h * 0.5
        max_y = center_y + size_h * 0.5

        gx0 = max(0, min(self.grid_w - 1, int(math.floor(min_x / self.cell_size))))
        gx1 = max(0, min(self.grid_w - 1, int(math.floor(max_x / self.cell_size))))
        gy0 = max(0, min(self.grid_h - 1, int(math.floor(min_y / self.cell_size))))
        gy1 = max(0, min(self.grid_h - 1, int(math.floor(max_y / self.cell_size))))

        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                yield gx, gy

    @staticmethod
    def _as_xy(pos: Any) -> Optional[Tuple[float, float]]:
        if pos is None:
            return None
        if hasattr(pos, "x") and hasattr(pos, "y"):
            return float(pos.x), float(pos.y)
        if isinstance(pos, dict):
            return float(pos.get("x", 0.0)), float(pos.get("y", 0.0))
        if isinstance(pos, (tuple, list)) and len(pos) >= 2:
            return float(pos[0]), float(pos[1])
        return None

    def _rasterize_terrains(self) -> None:
        terrains = list(getattr(self.map_info, "terrain_list", []) or [])
        for terrain in terrains:
            pos_obj = getattr(terrain, "_position", getattr(terrain, "position", None))
            pos = self._as_xy(pos_obj)
            if pos is None:
                continue
            size = getattr(terrain, "_size", getattr(terrain, "size", [10, 10]))
            tw = float(size[0]) if len(size) > 0 else 10.0
            th = float(size[1]) if len(size) > 1 else 10.0
            speed_mod = float(getattr(terrain, "_movement_speed_modifier", getattr(terrain, "movement_speed_modifier", 1.0)))
            dmg = float(getattr(terrain, "_deal_damage", getattr(terrain, "deal_damage", 0.0)))
            terrain_cost = (1.0 / max(0.2, speed_mod)) + max(0.0, dmg) * 1.2
            terrain_cost = max(0.3, min(8.0, terrain_cost))
            for gx, gy in self._world_rect_to_cells(float(pos[0]), float(pos[1]), tw, th):
                self.cost[gx][gy] = terrain_cost

    def _rasterize_obstacles(self) -> None:
        obstacles = list(getattr(self.map_info, "obstacle_list", []) or [])
        for obs in obstacles:
            alive = bool(getattr(obs, "is_alive", True))
            if not alive:
                continue
            pos_obj = getattr(obs, "_position", getattr(obs, "position", None))
            pos = self._as_xy(pos_obj)
            if pos is None:
                continue
            size = getattr(obs, "_size", getattr(obs, "size", [10, 10]))
            ow = float(size[0]) if len(size) > 0 else 10.0
            oh = float(size[1]) if len(size) > 1 else 10.0
            for gx, gy in self._world_rect_to_cells(float(pos[0]), float(pos[1]), ow, oh):
                self.passable[gx][gy] = False
                self.cost[gx][gy] = 9999.0

    def world_to_cell(self, x: float, y: float) -> GridCell:
        gx = max(0, min(self.grid_w - 1, int(math.floor(float(x) / self.cell_size))))
        gy = max(0, min(self.grid_h - 1, int(math.floor(float(y) / self.cell_size))))
        return gx, gy

    def cell_to_world_center(self, cell: GridCell) -> Tuple[float, float]:
        gx, gy = cell
        x = (float(gx) + 0.5) * self.cell_size
        y = (float(gy) + 0.5) * self.cell_size
        x = max(0.0, min(self.map_w, x))
        y = max(0.0, min(self.map_h, y))
        return x, y

    def clamp_to_nearest_passable(self, cell: GridCell, search_radius: int = 4) -> GridCell:
        gx, gy = cell
        if self.passable[gx][gy]:
            return cell
        best = cell
        best_d = float("inf")
        for r in range(1, max(1, int(search_radius)) + 1):
            for nx in range(max(0, gx - r), min(self.grid_w, gx + r + 1)):
                for ny in range(max(0, gy - r), min(self.grid_h, gy + r + 1)):
                    if not self.passable[nx][ny]:
                        continue
                    d = abs(nx - gx) + abs(ny - gy)
                    if d < best_d:
                        best = (nx, ny)
                        best_d = d
            if best_d < float("inf"):
                break
        return best

    def neighbors(self, cell: GridCell) -> Iterable[GridCell]:
        gx, gy = cell
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
            nx, ny = gx + dx, gy + dy
            if nx < 0 or ny < 0 or nx >= self.grid_w or ny >= self.grid_h:
                continue
            if not self.passable[nx][ny]:
                continue
            yield nx, ny

    @staticmethod
    def _heur(a: GridCell, b: GridCell) -> float:
        return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))

    def _move_cost(self, src: GridCell, dst: GridCell) -> float:
        sx, sy = src
        dx, dy = dst
        diag = 1.4142 if (sx != dx and sy != dy) else 1.0
        return self.cost[dx][dy] * diag

    def astar(self, start: GridCell, goal: GridCell) -> AStarResult:
        start = self.clamp_to_nearest_passable(start)
        goal = self.clamp_to_nearest_passable(goal)

        open_heap: List[Tuple[float, GridCell]] = []
        heappush(open_heap, (0.0, start))
        came_from: Dict[GridCell, GridCell] = {}
        g_score: Dict[GridCell, float] = {start: 0.0}
        closed: Set[GridCell] = set()

        reached = False
        best = start
        best_h = self._heur(start, goal)

        while open_heap:
            _, cur = heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)

            h_cur = self._heur(cur, goal)
            if h_cur < best_h:
                best_h = h_cur
                best = cur

            if cur == goal:
                reached = True
                best = cur
                break

            for nb in self.neighbors(cur):
                if nb in closed:
                    continue
                tentative = g_score[cur] + self._move_cost(cur, nb)
                if tentative < g_score.get(nb, float("inf")):
                    came_from[nb] = cur
                    g_score[nb] = tentative
                    f = tentative + self._heur(nb, goal)
                    heappush(open_heap, (f, nb))

        if best not in came_from and best != start:
            return AStarResult(path_cells=[start], reached=False)

        path = [best]
        while path[-1] in came_from:
            path.append(came_from[path[-1]])
        path.reverse()
        if not path:
            path = [start]
        return AStarResult(path_cells=path, reached=reached)
