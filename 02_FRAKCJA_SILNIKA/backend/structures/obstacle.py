"""Klasa przeszkod"""
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Union
from .position import Position


class ObstacleType(Enum):
    """Definicja typów przeszkód i ich kluczowych właściwości."""
    WALL = {"destructible": False, "see_through": False}
    TREE = {"destructible": True, "see_through": False}
    ANTI_TANK_SPIKE = {"destructible": False, "see_through": True}


@dataclass
class Obstacle(ABC):
    """Abstrakcyjna klasa bazowa dla przeszkód."""
    _id: str
    _position: Position
    _size: List[int] = field(default_factory=lambda: [10, 10])
    _is_alive: bool = True
    
    _obstacle_type: ObstacleType = field(init=False)
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def position(self) -> Position:
        return self._position
    
    @property
    def size(self) -> List[int]:
        return self._size
    
    @property
    def is_alive(self) -> bool:
        return self._is_alive
    
    @is_alive.setter
    def is_alive(self, value: bool) -> None:
        self._is_alive = value
    
    @property
    def obstacle_type(self) -> ObstacleType:
        return self._obstacle_type
    
    @property
    def is_destructible(self) -> bool:
        return self._obstacle_type.value['destructible']
    
    @property
    def is_see_through(self) -> bool:
        return self._obstacle_type.value['see_through']


@dataclass
class Wall(Obstacle):
    """Mur: Niezniszczalny, blokuje widok."""
    _obstacle_type: ObstacleType = field(default=ObstacleType.WALL, init=False)


@dataclass
class Tree(Obstacle):
    """Drzewo: Zniszczalne jednym trafieniem, blokuje widok."""
    _obstacle_type: ObstacleType = field(default=ObstacleType.TREE, init=False)


@dataclass
class AntiTankSpike(Obstacle):
    """Kolce Przeciwpancerne: Niezniszczalne, PRZEZIERNE."""
    _obstacle_type: ObstacleType = field(default=ObstacleType.ANTI_TANK_SPIKE, init=False)


ObstacleUnion = Union[Wall, Tree, AntiTankSpike]