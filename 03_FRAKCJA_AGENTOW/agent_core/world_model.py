from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass
class CellState:
    safe: float = 0.0
    danger: float = 0.0
    blocked: float = 0.0


class WorldModel:
    def __init__(self, grid_size: float = 10.0):
        self.grid_size = grid_size
        self.cell_states: Dict[Tuple[int, int], CellState] = {}
        self.visit_counts: Dict[Tuple[int, int], int] = {}
        self.dead_end_ttl: Dict[Tuple[int, int], float] = {}
        self.ally_occupancy_ttl: Dict[Tuple[int, int], float] = {}
        self.enemy_occupancy_ttl: Dict[Tuple[int, int], float] = {}

        self.powerup_cells: Set[Tuple[int, int]] = set()
        self.preferred_powerup_cells: Set[Tuple[int, int]] = set()
        self.checkpoint_cells: Set[Tuple[int, int]] = set()
        self.pothole_cells: Set[Tuple[int, int]] = set()

    def to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return int(x // self.grid_size), int(y // self.grid_size)

    def to_world_center(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        return (cell[0] + 0.5) * self.grid_size, (cell[1] + 0.5) * self.grid_size

    @staticmethod
    def neighbors4(cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = cell
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    def get_state(self, cell: Tuple[int, int]) -> CellState:
        if cell not in self.cell_states:
            self.cell_states[cell] = CellState()
        return self.cell_states[cell]

    def increment_visit(self, cell: Tuple[int, int]) -> None:
        self.visit_counts[cell] = self.visit_counts.get(cell, 0) + 1

    def decay_dead_ends(self) -> None:
        remaining: Dict[Tuple[int, int], float] = {}
        for cell, ttl in self.dead_end_ttl.items():
            next_ttl = ttl - 1.0
            if next_ttl > 0:
                remaining[cell] = next_ttl
        self.dead_end_ttl = remaining

        ally_remaining: Dict[Tuple[int, int], float] = {}
        for cell, ttl in self.ally_occupancy_ttl.items():
            next_ttl = ttl - 1.0
            if next_ttl > 0:
                ally_remaining[cell] = next_ttl
        self.ally_occupancy_ttl = ally_remaining

        enemy_remaining: Dict[Tuple[int, int], float] = {}
        for cell, ttl in self.enemy_occupancy_ttl.items():
            next_ttl = ttl - 1.0
            if next_ttl > 0:
                enemy_remaining[cell] = next_ttl
        self.enemy_occupancy_ttl = enemy_remaining

    def mark_dead_end(self, cell: Tuple[int, int], ttl: float = 520.0) -> None:
        self.dead_end_ttl[cell] = max(self.dead_end_ttl.get(cell, 0.0), ttl)

    def mark_ally_occupancy(self, cell: Tuple[int, int], ttl: float = 4.0) -> None:
        self.ally_occupancy_ttl[cell] = max(self.ally_occupancy_ttl.get(cell, 0.0), ttl)

    def ally_occupancy_score(self, cell: Tuple[int, int]) -> float:
        return self.ally_occupancy_ttl.get(cell, 0.0)

    def mark_enemy_occupancy(self, cell: Tuple[int, int], ttl: float = 4.0) -> None:
        self.enemy_occupancy_ttl[cell] = max(
            self.enemy_occupancy_ttl.get(cell, 0.0), ttl
        )

    def enemy_occupancy_score(self, cell: Tuple[int, int]) -> float:
        return self.enemy_occupancy_ttl.get(cell, 0.0)

    def is_blocked_for_pathing(self, cell: Tuple[int, int]) -> bool:
        if cell in self.dead_end_ttl:
            return True
        state = self.cell_states.get(cell)
        if state is None:
            return False
        if cell in self.pothole_cells:
            return state.blocked >= 2.5 or (state.danger >= 9.0 and state.safe < 0.6)
        if state.blocked >= 1.0:
            return True
        if state.danger >= 4.0 and state.safe < 1.5:
            return True
        return False

    def is_dangerous_cell(self, cell: Tuple[int, int]) -> bool:
        state = self.cell_states.get(cell)
        if state is None:
            return False
        return state.danger >= 1.0

    def local_block_pressure(self, cell: Tuple[int, int]) -> float:
        pressure = 0.0
        for neighbor in self.neighbors4(cell):
            if neighbor in self.dead_end_ttl:
                pressure += 1.2
                continue
            state = self.cell_states.get(neighbor)
            if state is None:
                continue
            pressure += 0.45 * state.blocked + 0.25 * state.danger
        return pressure

    def movement_cost(self, cell: Tuple[int, int]) -> float:
        state = self.cell_states.get(cell, CellState())
        visits = float(self.visit_counts.get(cell, 0))
        local_pressure = self.local_block_pressure(cell)
        ally_occupancy = self.ally_occupancy_score(cell)
        enemy_occupancy = self.enemy_occupancy_score(cell)
        base = 1.9

        if cell in self.checkpoint_cells:
            base *= 0.75

        if cell in self.powerup_cells:
            base *= 0.92
        if cell in self.preferred_powerup_cells:
            base *= 0.5
        if cell in self.pothole_cells:
            base += 1.2

        if cell not in self.cell_states:
            base += 2.8

        base -= 0.35 * min(state.safe, 3.0)
        base += 4.8 * state.danger
        base += 7.2 * state.blocked
        base += 0.8 * local_pressure
        base += 0.12 * min(visits, 12.0)
        base += 6.5 * ally_occupancy
        base += 8.0 * enemy_occupancy

        return max(0.35, base)
