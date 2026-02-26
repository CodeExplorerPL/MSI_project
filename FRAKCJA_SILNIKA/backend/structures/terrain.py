"""Klasa terenu"""
from abc import ABC
from dataclasses import dataclass, field
from typing import List, Literal, Union
from .position import Position


@dataclass
class Terrain(ABC):
    """Abstrakcyjna klasa bazowa dla typów terenu."""
    _id: str
    _position: Position
    _size: List[int] = field(default_factory=lambda: [10, 10])
    
    _terrain_type: str = field(init=False)
    _movement_speed_modifier: float = field(init=False)
    _deal_damage: int = field(init=False)
    
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
    def terrain_type(self) -> str:
        return self._terrain_type
    
    @property
    def movement_speed_modifier(self) -> float:
        return self._movement_speed_modifier
    
    @property
    def deal_damage(self) -> int:
        return self._deal_damage


@dataclass
class Grass(Terrain):
    """Trawa: Brak efektu."""
    _terrain_type: Literal["Grass"] = field(default="Grass", init=False)
    _movement_speed_modifier: float = field(default=1.0, init=False)
    _deal_damage: int = field(default=0, init=False)


@dataclass
class Road(Terrain):
    """Droga: Zwiększa prędkość ruchu."""
    _terrain_type: Literal["Road"] = field(default="Road", init=False)
    _movement_speed_modifier: float = field(default=1.5, init=False)
    _deal_damage: int = field(default=0, init=False)


@dataclass
class Swamp(Terrain):
    """Bagno: Spowalnia ruch."""
    _terrain_type: Literal["Swamp"] = field(default="Swamp", init=False)
    _movement_speed_modifier: float = field(default=0.4, init=False)
    _deal_damage: int = field(default=0, init=False)


@dataclass
class PotholeRoad(Terrain):
    """Droga z Dziurami: Spowalnia i zadaje minimalne obrażenia."""
    _terrain_type: Literal["PotholeRoad"] = field(default="PotholeRoad", init=False)
    _movement_speed_modifier: float = field(default=0.95, init=False)
    _deal_damage: int = field(default=1, init=False)


@dataclass
class Water(Terrain):
    """Woda: Spowalnia i zadaje obrażenia."""
    _terrain_type: Literal["Water"] = field(default="Water", init=False)
    _movement_speed_modifier: float = field(default=0.7, init=False)
    _deal_damage: int = field(default=1, init=False)


TerrainUnion = Union[Grass, Road, Swamp, PotholeRoad, Water]