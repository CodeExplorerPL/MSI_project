"""Klasa mapy"""
from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING
from .obstacle import ObstacleUnion
from .powerup import PowerUpData
from .terrain import TerrainUnion

if TYPE_CHECKING:
    from ..tank.base_tank import BaseTank


@dataclass
class MapInfo:
    """Informacje o mapie (rozmiar, przeszkody, itp.)."""
    _map_seed: str
    _obstacle_list: List[ObstacleUnion]
    _powerup_list: List[PowerUpData]
    _terrain_list: List[TerrainUnion]
    _all_tanks: List['BaseTank']
    _size: List[int] = field(default_factory=lambda: [500, 500])
    
    @property
    def map_seed(self) -> str:
        return self._map_seed
    
    @property
    def obstacle_list(self) -> List[ObstacleUnion]:
        return self._obstacle_list
    
    @property
    def powerup_list(self) -> List[PowerUpData]:
        return self._powerup_list
    
    @property
    def terrain_list(self) -> List[TerrainUnion]:
        return self._terrain_list
    
    @property
    def all_tanks(self) -> List['BaseTank']:
        return self._all_tanks
    
    @property
    def size(self) -> List[int]:
        return self._size
