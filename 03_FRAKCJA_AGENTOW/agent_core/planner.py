from __future__ import annotations

import heapq
from typing import Dict, List, Tuple

from .world_model import WorldModel


class AStarPlanner:
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model

    @staticmethod
    def _neighbors4(cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = cell
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def build_path(self, start: Tuple[int, int], goal: Tuple[int, int], radius: int = 18) -> List[Tuple[int, int]]:
        if start == goal:
            return [start]

        min_x, max_x = start[0] - radius, start[0] + radius
        min_y, max_y = start[1] - radius, start[1] + radius

        def in_bounds(cell: Tuple[int, int]) -> bool:
            return min_x <= cell[0] <= max_x and min_y <= cell[1] <= max_y

        frontier: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(frontier, (0.0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor in self._neighbors4(current):
                if not in_bounds(neighbor):
                    continue
                if self.world_model.is_blocked_for_pathing(neighbor):
                    continue

                tentative_g = g_score[current] + self.world_model.movement_cost(neighbor)
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(frontier, (f_score, neighbor))

        return []

    def path_risk(self, path: List[Tuple[int, int]]) -> float:
        if not path:
            return 1e9
        risk = 0.0
        for cell in path:
            state = self.world_model.get_state(cell)
            risk += (
                2.5 * state.danger
                + 3.0 * state.blocked
                + 2.2 * self.world_model.ally_occupancy_score(cell)
                + 2.8 * self.world_model.enemy_occupancy_score(cell)
                - 0.2 * min(state.safe, 3.0)
            )
        return risk / max(1, len(path))
