"""Klasa pozycji"""
from dataclasses import dataclass


@dataclass
class Position:
    """Reprezentuje pozycję X, Y na mapie."""
    _x: float
    _y: float
    
    @property
    def x(self) -> float:
        return self._x
    
    @x.setter
    def x(self, value: float) -> None:
        self._x = value
    
    @property
    def y(self) -> float:
        return self._y
    
    @y.setter
    def y(self, value: float) -> None:
        self._y = value

