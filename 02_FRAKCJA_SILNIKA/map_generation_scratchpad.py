"""
Roboczy skrypt do generowania i wizualizacji mapy.

Ten plik służy jako "piaskownica" do testowania generowania, wczytywania
i renderowania mapy.
"""

import pygame
import os
import sys
from typing import Dict, List

# --- Konfiguracja Ścieżek ---
# Dodajemy odpowiednie katalogi do ścieżki Pythona, aby umożliwić importy
# modułów z `backend`, niezależnie od miejsca uruchomienia skryptu.
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Zakładamy, że ten skrypt jest w '02_FRAKCJA_SILNIKA', a 'backend' jest w tym samym katalogu.
    # Dodajemy ten katalog do ścieżki.
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Główny katalog projektu ('MSI_project')
    project_root = os.path.abspath(os.path.join(current_dir, '..'))

    from backend.engine.map_loader import MapLoader, TILE_CLASSES
    # Importujemy funkcję do generowania mapy
    from generate_map import generate_map, OBSTACLE_TYPES, TERRAIN_TYPES

except ImportError as e:
    print(f"Błąd importu: {e}")
    print("Upewnij się, że skrypt jest uruchamiany z katalogu '02_FRAKCJA_SILNIKA' lub że struktura projektu jest poprawna.")
    sys.exit(1)


# --- Stałe ---
TILE_SIZE = 32  # Rozmiar kafelka w pikselach (zwiększony dla lepszej widoczności)

# --- Opcje generowania mapy ---
GENERATE_NEW_MAP = True  # Ustaw na True, aby wygenerować nową mapę przed wyświetleniem
GENERATED_MAP_FILENAME = "scratchpad_generated.csv"
MAP_WIDTH = 25
MAP_HEIGHT = 20
FALLBACK_MAP_FILENAME = 'map1.csv'  # Używana, gdy GENERATE_NEW_MAP = False

# WAŻNE: Ścieżka do assetów. Musisz dostosować tę ścieżkę, jeśli masz inną strukturę projektu.
ASSETS_PATH = os.path.join(current_dir, 'frontend', 'assets', 'tiles')
BACKGROUND_COLOR = (20, 20, 30) # Ciemnoniebieskie tło

# --- Funkcje pomocnicze ---

def load_tile_assets(tile_names: List[str], asset_path: str, tile_size: int) -> Dict[str, pygame.Surface]:
    """
    Wczytuje grafiki kafelków z podanej ścieżki.
    Jeśli grafika nie istnieje, tworzy biały kwadrat.
    """
    print(f"Ładowanie assetów z: {asset_path}")
    assets = {}
    
    # Domyślny biały kafelek na wypadek braku grafiki
    white_tile = pygame.Surface((tile_size, tile_size))
    white_tile.fill((255, 255, 255))

    for name in tile_names:
        # Nazwa pliku to nazwa klasy, np. "Wall.png"
        file_path = os.path.join(asset_path, f"{name}.png")
        try:
            # Wczytaj obraz i przeskaluj do rozmiaru kafelka
            image = pygame.image.load(file_path).convert_alpha()
            assets[name] = pygame.transform.scale(image, (tile_size, tile_size))
            print(f"  [OK] Wczytano asset: {name}.png")
        except (pygame.error, FileNotFoundError):
            print(f"  [!] Ostrzeżenie: Nie znaleziono assetu dla '{name}' w '{file_path}'. Używam białego kafelka.")
            assets[name] = white_tile
            
    return assets

def main():
    """Główna funkcja programu."""

    # --- Inicjalizacja Pygame ---
    pygame.init()
    pygame.display.set_caption("Podgląd Mapy")

    # --- Generowanie lub wybór mapy do wczytania ---
    if GENERATE_NEW_MAP:
        print(f"--- Generowanie nowej mapy: {GENERATED_MAP_FILENAME} ---")
        # Proste, równe proporcje dla wszystkich typów kafelków
        all_tile_types = OBSTACLE_TYPES + TERRAIN_TYPES
        tile_ratios = {tile: 1.0 / len(all_tile_types) for tile in all_tile_types}
        
        generate_map(MAP_WIDTH, MAP_HEIGHT, GENERATED_MAP_FILENAME, tile_ratios)
        map_to_load = GENERATED_MAP_FILENAME
    else:
        map_to_load = FALLBACK_MAP_FILENAME

    # --- Wczytywanie Mapy ---
    try:
        map_loader = MapLoader()
        map_info = map_loader.load_map(map_to_load, tile_size=TILE_SIZE)

        print(
            f"Wczytano mapę '{map_info.map_seed}' "
            f"o wymiarach: {map_info.size[0]}x{map_info.size[1]}px"
        )
    except FileNotFoundError as e:
        print(f"BŁĄD: Nie udało się wczytać mapy '{map_to_load}'")
        print(e)
        return
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd podczas ładowania mapy '{map_to_load}':")
        print(e)
        return

    # --- Ustawienia Ekranu ---
    # Rozmiar ekranu jest teraz obliczany na podstawie wymiarów siatki i rozmiaru kafelka
    grid_width, grid_height = map_info.grid_dimensions
    screen_width, screen_height = grid_width * TILE_SIZE, grid_height * TILE_SIZE
    screen = pygame.display.set_mode((screen_width, screen_height))

    # --- Wczytywanie Assetów ---
    tile_assets = load_tile_assets(list(TILE_CLASSES.keys()), ASSETS_PATH, TILE_SIZE) # type: ignore

    # --- Główna Pętla ---
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False

        screen.fill(BACKGROUND_COLOR)

        # Rysowanie mapy na podstawie siatki (grid_data)
        for y, row in enumerate(map_info.grid_data):
            for x, tile_name in enumerate(row):
                if not tile_name:
                    continue
                
                asset = tile_assets.get(tile_name)
                if asset:
                    # Współrzędne rysowania to po prostu indeksy siatki pomnożone przez rozmiar kafelka
                    screen.blit(asset, (x * TILE_SIZE, y * TILE_SIZE))

        pygame.display.flip()
        clock.tick(60)

    print("\nZamykanie podglądu mapy.")
    pygame.quit()


if __name__ == '__main__':
    if not os.path.isdir(ASSETS_PATH):
        print("-" * 60 + f"\n!!! OSTRZEŻENIE !!!\nNie znaleziono katalogu z grafikami w: '{ASSETS_PATH}'")
        print("Wszystkie kafelki będą wyświetlane jako białe kwadraty.")
        print("Upewnij się, że ścieżka ASSETS_PATH jest poprawna i że folder istnieje.\n" + "-" * 60)
    
    main()
