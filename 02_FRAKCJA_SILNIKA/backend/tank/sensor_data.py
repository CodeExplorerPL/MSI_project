""" Klasa z informacjami dla czolgu """
from dataclasses import dataclass
from typing import List
from ..structures import Position, PowerUpData, ObstacleUnion, TerrainUnion

@dataclass
class SeenTank:
    """Informacje o widocznym wrogu (dla Agentów to ID i pozycja)."""
    id: str
    position: Position
    is_damaged: bool
    heading: float
    barrel_angle: float
    distance: float
    tank_type: str
    team: int


@dataclass
class TankSensorData:
    """Dane wykryte przez systemy sensoryczne czołgu."""
    seen_tanks: List[SeenTank]
    seen_powerups: List[PowerUpData]
    seen_obstacles: List[ObstacleUnion]
    seen_terrains: List[TerrainUnion]
