"""Klasa mapy"""
from dataclasses import dataclass, field
from typing import List, Tuple, TYPE_CHECKING
from .obstacle import ObstacleUnion
from .powerup import PowerUpData
from .terrain import TerrainUnion

if TYPE_CHECKING:
    from ..tank.base_tank import Tank


@dataclass
class MapInfo:
    """
    Przechowuje wszystkie statyczne i dynamiczne informacje o mapie gry.
    """
    map_seed: str
    size: Tuple[int, int]

    # Siatka 2D z nazwami kafelków dla renderera
    grid_data: List[List[str]] = field(default_factory=list)

    # Listy obiektów dla silnika fizyki i logiki gry
    obstacle_list: List[ObstacleUnion] = field(default_factory=list)
    powerup_list: List[PowerUpData] = field(default_factory=list)
    terrain_list: List[TerrainUnion] = field(default_factory=list)
    all_tanks: List['Tank'] = field(default_factory=list)
