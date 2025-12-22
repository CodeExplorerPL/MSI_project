"""Moduł structures - zawiera wszystkie klasy strukturalne gry"""

from .position import Position
from .obstacle import (
    ObstacleType,
    Obstacle,
    Wall,
    Tree,
    AntiTankSpike,
    ObstacleUnion
)
from .terrain import (
    Terrain,
    Grass,
    Road,
    Swamp,
    PotholeRoad,
    Water,
    TerrainUnion
)
from .powerup import PowerUpType, PowerUpData
from .map_info import MapInfo
from .ammo import AmmoType, AmmoSlot

__all__ = [
    'Position',
    'ObstacleType',
    'Obstacle',
    'Wall',
    'Tree',
    'AntiTankSpike',
    'ObstacleUnion',
    'Terrain',
    'Grass',
    'Road',
    'Swamp',
    'PotholeRoad',
    'Water',
    'TerrainUnion',
    'PowerUpType',
    'PowerUpData',
    'MapInfo',
]