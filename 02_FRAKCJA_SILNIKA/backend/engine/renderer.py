"""
Game Renderer Module - wizualizacja stanu gry za pomocą Pygame.
"""

import os
import sys
from typing import Any, Dict, List, Optional

# Dodajemy ścieżkę projektu, aby umożliwić importy
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except Exception:
    sys.path.append(os.path.abspath('.'))

# Warunkowy import Pygame, aby silnik działał bez niego w trybie headless
try:
    import pygame
    import pygame.math
except ImportError:
    pygame = None

from backend.structures.map_info import MapInfo
from backend.tank.base_tank import Tank
from backend.utils.config import TankType

# --- Stałe ---
ASSETS_PATH = os.path.join(project_root, 'frontend', 'assets')
TILE_SIZE = 16  # Rozmiar kafelka w pikselach
BACKGROUND_COLOR = (20, 20, 30)
TEAM_COLORS = {
    1: (227, 51, 36),   # Czerwony
    2: (36, 122, 227),  # Niebieski
    0: (150, 150, 150)  # Neutralny/Domyślny
}


class GameRenderer:
    """
    Obsługuje cały proces renderowania gry przy użyciu Pygame.
    """

    def __init__(self, map_width: int = 500, map_height: int = 500):
        if not pygame:
            raise ImportError("Pygame jest wymagany do trybu graficznego, ale nie jest zainstalowany.")

        self.screen = None
        self.clock = pygame.time.Clock()
        self.map_surface = None
        self.grid_data: Optional[List[List[str]]] = None

        self.screen_width = map_width
        self.screen_height = map_height

        # Słownik na wczytane i przetworzone zasoby graficzne
        self.assets: Dict[str, Dict] = {
            'tiles': {},
            'tanks': {},
            'powerups': {},
            'projectiles': {}
        }

    def initialize(self, map_size: Optional[tuple] = None, grid_data: Optional[List[List[str]]] = None) -> bool:
        """Inicjalizuje okno Pygame i wczytuje wszystkie zasoby."""
        try:
            pygame.init()
            if map_size:
                self.screen_width, self.screen_height = map_size

            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("MSI Tanks - Widok Silnika")

            self.grid_data = grid_data
            self._load_all_assets()

            if self.grid_data:
                self._pre_render_map()

            return True
        except Exception as e:
            print(f"Błąd podczas inicjalizacji renderera: {e}")
            return False

    def render(self, tanks: Dict[str, Tank], projectiles: List[Any], powerups: Dict[str, Any]) -> bool:
        """Renderuje pojedynczą klatkę gry. Zwraca False, jeśli użytkownik zamknie okno."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill(BACKGROUND_COLOR)

        if self.map_surface:
            self.screen.blit(self.map_surface, (0, 0))

        # TODO: Rysowanie power-upów i pocisków

        for tank in tanks.values():
            # if tank.is_alive():  # Rysuj tylko żywe czołgi (gdy metoda będzie dostępna)
            self._draw_tank(tank)

        pygame.display.flip()
        self.clock.tick(60)  # Ograniczenie do 60 FPS
        return True

    def cleanup(self):
        """Zamyka Pygame."""
        if pygame:
            pygame.quit()

    # --- Konwersja współrzędnych i kątów ---

    def _convert_coords(self, engine_pos: Any) -> pygame.math.Vector2:
        """Konwertuje współrzędne silnika (Y w górę) na współrzędne ekranu (Y w dół)."""
        return pygame.math.Vector2(engine_pos.x, self.screen_height - engine_pos.y)

    def _convert_angle(self, engine_angle: float) -> float:
        """Konwertuje kąt silnika (0=góra, zgodnie z zegarem) na kąt Pygame (0=prawo, przeciwnie do zegara)."""
        return -engine_angle + 90

    # --- Wczytywanie zasobów ---

    def _load_all_assets(self):
        """Wczytuje wszystkie zasoby gry do pamięci."""
        print("Wczytywanie zasobów graficznych...")
        tile_names = ['Grass', 'Road', 'Wall', 'Tree', 'Water', 'Swamp', 'PotholeRoad', 'AntiTankSpike']
        self._load_tile_assets(tile_names)
        self._load_tank_assets()
        print("Zasoby wczytane.")

    def _load_tile_assets(self, tile_names: List[str]):
        """Wczytuje grafiki kafelków."""
        path = os.path.join(ASSETS_PATH, 'tiles')
        for name in tile_names:
            file_path = os.path.join(path, f"{name}.png")
            try:
                image = pygame.image.load(file_path).convert_alpha()
                self.assets['tiles'][name] = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
            except (pygame.error, FileNotFoundError):
                placeholder = pygame.Surface((TILE_SIZE, TILE_SIZE))
                placeholder.fill((50, 50, 50))
                pygame.draw.rect(placeholder, (80, 80, 80), placeholder.get_rect(), 1)
                self.assets['tiles'][name] = placeholder

    def _load_tank_assets(self):
        """Wczytuje i wstępnie przetwarza zasoby czołgów dla wszystkich typów i drużyn."""
        tank_types_map = {
            TankType.LIGHT: 'light_tank',
            TankType.HEAVY: 'heavy_tank',
            TankType.SNIPER: 'sniper_tank',
        }

        for tank_type_enum, folder_name in tank_types_map.items():
            tank_path = os.path.join(ASSETS_PATH, 'tanks', folder_name)
            try:
                hull_img = pygame.image.load(os.path.join(tank_path, 'tnk1.png')).convert_alpha()
                hull_mask = pygame.image.load(os.path.join(tank_path, 'msk1.png')).convert_alpha()
                turret_img = pygame.image.load(os.path.join(tank_path, 'tnk2.png')).convert_alpha()
                turret_mask = pygame.image.load(os.path.join(tank_path, 'msk2.png')).convert_alpha()

                self.assets['tanks'][tank_type_enum] = {}
                for team_id, color in TEAM_COLORS.items():
                    self.assets['tanks'][tank_type_enum][team_id] = {
                        'hull': hull_img,
                        'turret': turret_img,
                        'hull_colored': self._colorize(hull_mask, color),
                        'turret_colored': self._colorize(turret_mask, color),
                    }
            except (pygame.error, FileNotFoundError) as e:
                print(f"Ostrzeżenie: Nie można wczytać zasobów dla '{folder_name}': {e}")

    def _colorize(self, mask: pygame.Surface, color: tuple) -> pygame.Surface:
        """Nakłada kolor na obraz maski."""
        color_layer = pygame.Surface(mask.get_size(), pygame.SRCALPHA)
        color_layer.fill(color)
        color_layer.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        return color_layer

    # --- Pomocnicy rysowania ---

    def _pre_render_map(self):
        print("zaczynamy prerenderowanie mapy")
        """Renderuje statyczne tło mapy na osobną powierzchnię dla optymalizacji."""
        if not self.grid_data:
            return

        self.map_surface = pygame.Surface((self.screen_width, self.screen_height))
        self.map_surface.fill(BACKGROUND_COLOR)

        for y, row in enumerate(self.grid_data):
            for x, tile_name in enumerate(row):
                if tile_name and (asset := self.assets['tiles'].get(tile_name)):
                    # Siatka mapy (grid_data) ma (0,0) w lewym górnym rogu, więc nie trzeba tu odwracać osi Y
                    self.map_surface.blit(asset, (x * TILE_SIZE, y * TILE_SIZE))
        print("kończymy prerenderowanie mapy")


    def _draw_tank(self, tank: Tank):
        """Rysuje pojedynczy czołg z kadłubem i wieżą."""
        # Po refaktoryzacji zakładamy publiczne atrybuty. Używamy getattr dla bezpieczeństwa.
        tank_type = getattr(tank, 'tank_type', TankType.LIGHT)
        team = getattr(tank, 'team', 0)

        position = getattr(tank, 'position', None)
        heading = getattr(tank, 'heading', 0.0)
        barrel_angle = getattr(tank, 'barrel_angle', 0.0)

        if not position: return

        tank_assets = self.assets['tanks'].get(tank_type, {}).get(team)
        if not tank_assets:
            pos = self._convert_coords(position)
            pygame.draw.circle(self.screen, TEAM_COLORS.get(team, (255, 255, 255)), pos, 10)
            return

        # Konwersja kątów
        hull_angle_pg = self._convert_angle(heading)
        turret_absolute_angle_engine = heading + barrel_angle
        turret_angle_pg = self._convert_angle(turret_absolute_angle_engine)

        # Konwersja pozycji
        screen_pos = self._convert_coords(position)

        # Rysowanie kadłuba
        self._blit_rotated(tank_assets['hull'], screen_pos, hull_angle_pg)
        self._blit_rotated(tank_assets['hull_colored'], screen_pos, hull_angle_pg)

        # Rysowanie wieży
        # TODO: Dodać logikę pivot point dla wieży, jeśli nie jest wycentrowana
        self._blit_rotated(tank_assets['turret'], screen_pos, turret_angle_pg)
        self._blit_rotated(tank_assets['turret_colored'], screen_pos, turret_angle_pg)

    def _blit_rotated(self, surface: pygame.Surface, pos: pygame.math.Vector2, angle: float):
        """Obraca powierzchnię i rysuje ją, centrując na danej pozycji."""
        rotated_surface = pygame.transform.rotate(surface, angle)
        new_rect = rotated_surface.get_rect(center=pos)
        self.screen.blit(rotated_surface, new_rect.topleft)