""" Inicjalizacja pakietu struktur """

from .map_info import MapInfo
from .obstacle import (
    Obstacle,
    ObstacleType,
    Wall,
    Tree,
    AntiTankSpike,
    ObstacleUnion
)
from .position import Position
from .powerup import PowerUpData, PowerUpType
from .terrain import (
    Terrain,
    Grass,
    Road,
    Swamp,
    PotholeRoad,
    Water,
    TerrainUnion
)

__all__ = [
    "MapInfo",
    "MapLoader",
    "Obstacle",
    "ObstacleType",
    "Wall",
    "Tree",
    "AntiTankSpike",
    "ObstacleUnion",
    "Position",
    "PowerUpData",
    "PowerUpType",
    "Terrain",
    "Grass",
    "Road",
    "Swamp",
    "PotholeRoad",
    "Water",
    "TerrainUnion",
]
