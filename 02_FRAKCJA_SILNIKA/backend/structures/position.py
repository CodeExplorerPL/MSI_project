"""Klasa pozycji"""
from dataclasses import dataclass

@dataclass
class Position:
    """Reprezentuje pozycję X, Y na mapie."""
    x: float
    y: float
