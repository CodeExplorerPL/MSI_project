""" Klasa przeszkod """
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Union

from .position import Position



# --- PRZESZKODY (Obstacles) ---
class ObstacleType(Enum):
    """Definicja typów przeszkód i ich kluczowych właściwości."""
    WALL = {"destructible": False, "see_through": False}
    TREE = {"destructible": True, "see_through": False}
    ANTI_TANK_SPIKE = {"destructible": False, "see_through": True}


@dataclass
class Obstacle(ABC):
    """Abstrakcyjna klasa bazowa dla przeszkód."""
    id: str
    position: Position
    size: List[int] = field(default_factory=lambda: [10, 10])
    is_alive: bool = True

    obstacle_type: ObstacleType = field(init=False)

    @property
    def is_destructible(self) -> bool: return self.obstacle_type.value[
        'destructible']

    @property
    def is_see_through(self) -> bool: return self.obstacle_type.value[
        'see_through']


@dataclass
class Wall(Obstacle):
    """Mur: Niezniszczalny, blokuje widok."""
    obstacle_type: ObstacleType = field(default=ObstacleType.WALL, init=False)


@dataclass
class Tree(Obstacle):
    """Drzewo: Zniszczalne jednym trafieniem, blokuje widok."""
    obstacle_type: ObstacleType = field(default=ObstacleType.TREE, init=False)


@dataclass
class AntiTankSpike(Obstacle):
    """Kolce Przeciwpancerne: Niezniszczalne, PRZEZIERNE."""
    obstacle_type: ObstacleType = field(default=ObstacleType.ANTI_TANK_SPIKE,
                                        init=False)


ObstacleUnion = Union[Wall, Tree, AntiTankSpike]