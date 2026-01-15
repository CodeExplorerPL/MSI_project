"""Klasa terenu"""
from abc import ABC
from dataclasses import dataclass, field
from typing import List, Literal, Union
from .position import Position


@dataclass
class Terrain(ABC):
    """Abstrakcyjna klasa bazowa dla typów terenu."""
    id: str
    position: Position
    size: List[int] = field(default_factory=lambda: [10, 10])
    
    terrain_type: str = field(init=False)
    movement_speed_modifier: float = field(init=False)
    deal_damage: int = field(init=False)


@dataclass
class Grass(Terrain):
    """Trawa: Brak efektu."""
    terrain_type: Literal["Grass"] = field(default="Grass", init=False)
    movement_speed_modifier: float = field(default=1.0, init=False)
    deal_damage: int = field(default=0, init=False)


@dataclass
class Road(Terrain):
    """Droga: Zwiększa prędkość ruchu."""
    terrain_type: Literal["Road"] = field(default="Road", init=False)
    movement_speed_modifier: float = field(default=1.5, init=False)
    deal_damage: int = field(default=0, init=False)


@dataclass
class Swamp(Terrain):
    """Bagno: Spowalnia ruch."""
    terrain_type: Literal["Swamp"] = field(default="Swamp", init=False)
    movement_speed_modifier: float = field(default=0.4, init=False)
    deal_damage: int = field(default=0, init=False)


@dataclass
class PotholeRoad(Terrain):
    """Droga z Dziurami: Spowalnia i zadaje minimalne obrażenia."""
    terrain_type: Literal["PotholeRoad"] = field(default="PotholeRoad", init=False)
    movement_speed_modifier: float = field(default=0.95, init=False)
    deal_damage: int = field(default=1, init=False)


@dataclass
class Water(Terrain):
    """Woda: Spowalnia i zadaje obrażenia."""
    terrain_type: Literal["Water"] = field(default="Water", init=False)
    movement_speed_modifier: float = field(default=0.7, init=False)
    deal_damage: int = field(default=2, init=False)


TerrainUnion = Union[Grass, Road, Swamp, PotholeRoad, Water]