""" Klasa mapy """
from dataclasses import dataclass, field
from typing import List

from .obstacle import ObstacleUnion
from .powerup import PowerUpData
from .terrain import TerrainUnion
from ..tank.base_tank import Tank


@dataclass
class MapInfo:
    """Informacje o mapie (rozmiar, przeszkody, itp.)."""
    map_seed: str
    obstacle_list: List[ObstacleUnion]
    powerup_list: List[PowerUpData]
    terrain_list: List[TerrainUnion]  # Użycie Unii
    all_tanks: List[Tank]
    size: List[int] = field(default_factory=lambda: [500, 500])