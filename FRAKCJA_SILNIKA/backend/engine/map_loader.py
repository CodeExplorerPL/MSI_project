"""
Moduł do wczytywania map z plików CSV.
"""
import csv
import os
import uuid
from typing import Dict, Type, List, Union

from ..structures.map_info import MapInfo
from ..structures.position import Position
from ..structures.obstacle import Obstacle, Wall, Tree, AntiTankSpike, ObstacleUnion
from ..structures.terrain import Terrain, Grass, Road, Swamp, PotholeRoad, Water, TerrainUnion

# Mapowanie nazw kafelków na klasy
TILE_CLASSES: Dict[str, Union[Type[Obstacle], Type[Terrain]]] = {
    # Przeszkody
    'Wall': Wall,
    'Tree': Tree,
    'AntiTankSpike': AntiTankSpike,
    # Tereny
    'Grass': Grass,
    'Road': Road,
    'Swamp': Swamp,
    'PotholeRoad': PotholeRoad,
    'Water': Water,
}


class MapLoader:
    """Wczytuje mapę z pliku CSV i tworzy obiekty gry."""

    def __init__(self, maps_directory: str = None):
        """
        Inicjalizuje MapLoader.

        Args:
            maps_directory (str, optional): Ścieżka do katalogu z mapami. 
                                            Jeśli nie podano, używa domyślnej lokalizacji
                                            względem tego pliku.
        """
        if maps_directory:
            self.maps_directory = os.path.abspath(maps_directory)
        else:
            base_dir = os.path.dirname(__file__)
            self.maps_directory = os.path.abspath(os.path.join(base_dir, '..', 'maps'))
        
        if not os.path.exists(self.maps_directory):
            raise FileNotFoundError(f"Katalog z mapami nie został znaleziony: {self.maps_directory}")

    def get_available_maps(self) -> List[str]:
        """Zwraca listę dostępnych plików map."""
        if not os.path.isdir(self.maps_directory):
            return []
        return [f for f in os.listdir(self.maps_directory) if f.endswith('.csv')]

    def load_map(self, map_filename: str, tile_size: int = 10) -> MapInfo:
        """
        Wczytuje mapę z podanego pliku CSV i zwraca obiekt MapInfo.

        Args:
            map_filename (str): Nazwa pliku mapy (np. 'map1.csv').
            tile_size (int): Rozmiar boku pojedynczego kafelka mapy.

        Returns:
            MapInfo: Obiekt zawierający wszystkie informacje o wczytanej mapie.
        
        Raises:
            FileNotFoundError: Jeśli plik mapy nie zostanie znaleziony.
        """
        map_path = os.path.join(self.maps_directory, map_filename)
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Plik mapy nie został znaleziony: {map_path}")

        obstacle_list: List[ObstacleUnion] = []
        terrain_list: List[TerrainUnion] = []
        map_width = 0
        map_height = 0

        with open(map_path, 'r', newline='') as csvfile:
            map_reader = csv.reader(csvfile)
            rows = list(map_reader)
            map_height = len(rows)

            for y, row in enumerate(rows):
                if y == 0:
                    map_width = len(row)
                
                for x, tile_name in enumerate(row):
                    tile_name = tile_name.strip()
                    if not tile_name:
                        continue

                    tile_class = TILE_CLASSES.get(tile_name)
                    if not tile_class:
                        print(f"Ostrzeżenie: Nieznany typ kafelka '{tile_name}' na pozycji ({x}, {y}). Pomijanie.")
                        continue

                    # Pozycja w centrum kafelka
                    pos_x = x * tile_size + tile_size / 2
                    pos_y = y * tile_size + tile_size / 2
                    pos = Position(pos_x, pos_y)
                    tile_id = str(uuid.uuid4())

                    instance = tile_class(_id=tile_id, _position=pos)

                    if isinstance(instance, Obstacle):
                        obstacle_list.append(instance)
                    elif isinstance(instance, Terrain):
                        terrain_list.append(instance)

        map_total_width = map_width * tile_size
        map_total_height = map_height * tile_size

        map_info = MapInfo(
            _map_seed=map_filename,
            _obstacle_list=obstacle_list,
            _terrain_list=terrain_list,
            _powerup_list=[],  # Lista power-upów jest na razie pusta
            _all_tanks=[],     # Lista czołgów jest na razie pusta
            _size=[map_total_width, map_total_height]
        )

        return map_info
