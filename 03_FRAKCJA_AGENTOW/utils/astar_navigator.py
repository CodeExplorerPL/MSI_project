from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

from .controller_types import GridCell, NavigatorOutput
from .path_planning import GridMapAStar


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class AStar_Navigator:
    """
    High-level navigation module:
    - selects objective (enemy / power-up / exploration)
    - plans A* path
    - emits compact motion hints for low-level TSK motion controller
    """

    def __init__(
        self,
        map_info: object,
        cell_size: float = 10.0,
        enemy_memory_ttl: int = 30,
        powerup_memory_ttl: int = 1000,
        powerup_forget_distance: float = 10.0,
        powerup_missing_clear_ticks: int = 18,
    ):
        self.cell_size = float(cell_size)
        self.map_info = map_info
        self.planner = GridMapAStar(map_info=map_info, cell_size=cell_size)
        self.visited: List[List[int]] = [
            [0 for _ in range(self.planner.grid_h)] for _ in range(self.planner.grid_w)
        ]

        self.path_cells: List[GridCell] = []
        self.target_cell: Optional[GridCell] = None

        self.last_enemy_pos: Optional[Tuple[float, float]] = None
        self.last_enemy_age = 10_000
        self.enemy_memory_ttl = max(1, int(enemy_memory_ttl))

        self.last_powerup_pos: Optional[Tuple[float, float]] = None
        self.last_powerup_age = 10_000
        self.powerup_memory_ttl = max(1, int(powerup_memory_ttl))
        self.powerup_forget_distance = max(1.0, float(powerup_forget_distance))
        self.powerup_missing_clear_ticks = max(1, int(powerup_missing_clear_ticks))

    def refresh_map(self, map_info: object) -> None:
        """
        Rebuild planner when obstacle/terrain memory changes.
        Keeps visited matrix when grid size stays unchanged.
        """
        self.map_info = map_info
        new_planner = GridMapAStar(map_info=map_info, cell_size=self.cell_size)
        if new_planner.grid_w == self.planner.grid_w and new_planner.grid_h == self.planner.grid_h:
            pass
        else:
            self.visited = [[0 for _ in range(new_planner.grid_h)] for _ in range(new_planner.grid_w)]
        self.planner = new_planner

    @staticmethod
    def _signed_angle_diff_deg(target: float, current: float) -> float:
        return ((target - current + 180.0) % 360.0) - 180.0

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    def _mark_visited(self, x: float, y: float) -> GridCell:
        cell = self.planner.world_to_cell(x, y)
        gx, gy = cell
        self.visited[gx][gy] += 1
        return cell

    def _nearest_point(
        self, src: Tuple[float, float], points: List[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        if not points:
            return None
        return min(points, key=lambda p: self._dist(src, p))

    def _update_target_memory(
        self,
        pos: Tuple[float, float],
        sensor_snapshot: Dict[str, List[Tuple[float, float]]],
    ) -> None:
        seen_tanks = list(sensor_snapshot.get("seen_tanks", []))
        seen_powerups = list(sensor_snapshot.get("seen_powerups", []))

        enemy = self._nearest_point(pos, seen_tanks)
        if enemy is not None:
            self.last_enemy_pos = (float(enemy[0]), float(enemy[1]))
            self.last_enemy_age = 0
        else:
            self.last_enemy_age += 1

        powerup = self._nearest_point(pos, seen_powerups)
        if powerup is not None:
            self.last_powerup_pos = (float(powerup[0]), float(powerup[1]))
            self.last_powerup_age = 0
        else:
            if (
                self.last_powerup_pos is not None
                and self._dist(pos, self.last_powerup_pos) <= self.powerup_forget_distance
                and self.last_powerup_age >= self.powerup_missing_clear_ticks
            ):
                self.last_powerup_pos = None
                self.last_powerup_age = 10_000
            else:
                self.last_powerup_age += 1

    def _choose_frontier_cell(self, start_cell: GridCell) -> GridCell:
        sx, sy = start_cell
        best = start_cell
        best_score = float("inf")

        for gx in range(self.planner.grid_w):
            for gy in range(self.planner.grid_h):
                if not self.planner.passable[gx][gy]:
                    continue
                if self.visited[gx][gy] > 0:
                    continue
                d = abs(gx - sx) + abs(gy - sy)
                if d < best_score:
                    best = (gx, gy)
                    best_score = d

        if best_score < float("inf"):
            return best

        best_mix = float("inf")
        for gx in range(self.planner.grid_w):
            for gy in range(self.planner.grid_h):
                if not self.planner.passable[gx][gy]:
                    continue
                d = abs(gx - sx) + abs(gy - sy)
                mix = float(self.visited[gx][gy]) * 5.0 + float(d)
                if mix < best_mix:
                    best = (gx, gy)
                    best_mix = mix
        return best

    def _select_mode_and_target(
        self,
        state: Dict[str, float | str],
        sensor_snapshot: Dict[str, List[Tuple[float, float]]],
    ) -> Tuple[str, Tuple[float, float], bool, bool]:
        px = float(state.get("x", 0.0))
        py = float(state.get("y", 0.0))
        pos = (px, py)
        seen_tanks = list(sensor_snapshot.get("seen_tanks", []))
        seen_powerups = list(sensor_snapshot.get("seen_powerups", []))

        target_enemy = self._nearest_point(pos, seen_tanks)
        if target_enemy is not None:
            return "ENGAGE_ENEMY", target_enemy, True, bool(seen_powerups)

        target_pu = self._nearest_point(pos, seen_powerups)
        if target_pu is not None:
            return "COLLECT_POWERUP", target_pu, bool(seen_tanks), True
        if self.last_powerup_pos is not None and self.last_powerup_age <= self.powerup_memory_ttl:
            return "CHASE_LAST_POWERUP", self.last_powerup_pos, bool(seen_tanks), False

        start_cell = self.planner.world_to_cell(px, py)
        frontier = self._choose_frontier_cell(start_cell)
        return "EXPLORE", self.planner.cell_to_world_center(frontier), bool(seen_tanks), bool(seen_powerups)

    def _plan_to_target(self, start_cell: GridCell, target_world: Tuple[float, float], force: bool) -> None:
        target_cell = self.planner.world_to_cell(target_world[0], target_world[1])
        target_cell = self.planner.clamp_to_nearest_passable(target_cell)
        if not force and self.target_cell == target_cell and len(self.path_cells) > 2:
            return

        self.target_cell = target_cell
        result = self.planner.astar(start_cell, target_cell)
        self.path_cells = result.path_cells if result.path_cells else [start_cell]

    def _next_waypoint_world(self, start_cell: GridCell) -> Tuple[float, float]:
        if not self.path_cells:
            return self.planner.cell_to_world_center(start_cell)

        if self.path_cells[0] != start_cell:
            if start_cell in self.path_cells:
                idx = self.path_cells.index(start_cell)
                self.path_cells = self.path_cells[idx:]
            else:
                self.path_cells = [start_cell]

        if len(self.path_cells) >= 2:
            return self.planner.cell_to_world_center(self.path_cells[1])
        return self.planner.cell_to_world_center(self.path_cells[0])

    def navigate(
        self,
        state: Dict[str, float | str],
        sensor_snapshot: Dict[str, List[Tuple[float, float]]],
    ) -> NavigatorOutput:
        px = float(state.get("x", 0.0))
        py = float(state.get("y", 0.0))
        heading = float(state.get("heading", 0.0))
        cur_pos = (px, py)

        start_cell = self._mark_visited(px, py)
        self._update_target_memory(cur_pos, sensor_snapshot)
        mode, target_world, seen_enemy, seen_powerup = self._select_mode_and_target(
            state=state, sensor_snapshot=sensor_snapshot
        )

        force_replan = mode == "ENGAGE_ENEMY"
        self._plan_to_target(start_cell=start_cell, target_world=target_world, force=force_replan)
        waypoint_world = self._next_waypoint_world(start_cell)

        dx_wp = float(waypoint_world[0]) - px
        dy_wp = float(waypoint_world[1]) - py
        desired_heading = math.degrees(math.atan2(dy_wp, dx_wp))
        heading_error = self._signed_angle_diff_deg(desired_heading, heading)
        dist_to_wp = math.hypot(dx_wp, dy_wp)
        dist_to_target = math.hypot(float(target_world[0]) - px, float(target_world[1]) - py)

        avoid_obstacle = bool(
            len(self.path_cells) <= 1
            and dist_to_target > max(1.0, float(self.planner.cell_size) * 1.2)
        )
        if abs(heading_error) > 100.0 and dist_to_wp < max(1.0, float(self.planner.cell_size) * 0.7):
            avoid_obstacle = True

        return NavigatorOutput(
            mode=mode,
            target_world=(float(target_world[0]), float(target_world[1])),
            waypoint_world=(float(waypoint_world[0]), float(waypoint_world[1])),
            desired_heading_deg=float(desired_heading),
            heading_error_deg=float(_clamp(heading_error, -180.0, 180.0)),
            distance_to_waypoint=float(max(0.0, dist_to_wp)),
            distance_to_target=float(max(0.0, dist_to_target)),
            path_cells=list(self.path_cells),
            avoid_obstacle=avoid_obstacle,
            seen_enemy=bool(seen_enemy),
            seen_powerup=bool(seen_powerup),
        )
