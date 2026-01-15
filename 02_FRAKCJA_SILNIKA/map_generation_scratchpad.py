"""
Roboczy skrypt do generowania i wizualizacji mapy.

Ten plik służy jako "piaskownica" do testowania generowania, wczytywania
i renderowania mapy.
"""
import random

import pygame
import pygame.math
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
POWERUPS_ASSETS_PATH = os.path.join(current_dir, 'frontend', 'assets', 'power-ups')

# --- DODANE: Opcje generowania power-upów ---
POWERUP_TYPES = ['Medkit', 'Shield', 'Overcharge', 'AmmoBox_Heavy', 'AmmoBox_Light', 'AmmoBox_Sniper']
POWERUP_SIZE = int(TILE_SIZE * 0.75)
POWERUP_SPAWN_START_TICK = 50
POWERUP_SPAWN_INTERVAL = 180  # Co 3 sekundy przy 60 FPS
MAX_POWERUPS_ON_MAP = 10

BACKGROUND_COLOR = (20, 20, 30) # Ciemnoniebieskie tło
ROTATION_SPEED = 3 # Prędkość obrotu czołgu w stopniach na klatkę
TEAM_COLOR = (255, 0, 0) # Czerwony kolor drużyny (prosty do zmiany)

# --- DODANE: Statystyki symulujące silnik fizyki ---
# Maksymalna prędkość obrotu na klatkę (w stopniach)
HEADING_SPIN_RATE = 2.5  # Kadłub
BARREL_SPIN_RATE = 4.0   # Wieżyczka


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

def load_powerup_assets(powerup_names: List[str], asset_path: str, size: int) -> Dict[str, pygame.Surface]:
    """
    Wczytuje grafiki power-upów z podanej ścieżki.
    Jeśli grafika nie istnieje, tworzy różowy kwadrat.
    """
    print(f"Ładowanie assetów power-upów z: {asset_path}")
    assets = {}
    
    # Domyślny różowy kwadrat na wypadek braku grafiki
    pink_square = pygame.Surface((size, size))
    pink_square.fill((255, 0, 255))
    pink_square.set_colorkey((0, 0, 0)) # Ustawienie czarnego jako przezroczystego

    for name in powerup_names:
        # Nazwa pliku to nazwa z listy, np. "Medkit.png"
        file_path = os.path.join(asset_path, f"{name}.png")
        try:
            image = pygame.image.load(file_path).convert_alpha()
            assets[name] = pygame.transform.scale(image, (size, size))
            print(f"  [OK] Wczytano asset power-upu: {name}.png")
        except (pygame.error, FileNotFoundError):
            print(f"  [!] Ostrzeżenie: Nie znaleziono assetu dla '{name}' w '{file_path}'. Używam różowego kwadratu.")
            assets[name] = pink_square
            
    return assets

def normalize_angle(angle: float) -> float:
    """Normalizuje kąt do zakresu [-180, 180]."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

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
    # Rozmiar ekranu jest teraz pobierany bezpośrednio z obiektu MapInfo
    screen_width, screen_height = map_info.size
    screen = pygame.display.set_mode((screen_width, screen_height))

    # --- Wczytywanie Assetów ---
    tile_assets = load_tile_assets(list(TILE_CLASSES.keys()), ASSETS_PATH, TILE_SIZE) # type: ignore
    powerup_assets = load_powerup_assets(POWERUP_TYPES, POWERUPS_ASSETS_PATH, POWERUP_SIZE)

    # --- DODANE: Wczytywanie grafiki czołgu ---
    tank_image = None
    colored_mask_image = None
    turret_image = None
    colored_turret_mask_image = None
    pivot_offset = pygame.math.Vector2(0, 0)  # Domyślny pivot, jeśli wczytanie się nie powiedzie
    try:
        tank_asset_path = os.path.join(current_dir, 'frontend', 'assets', 'tanks', 'light_tank', 'tnk1.png')
        raw_tank_image = pygame.image.load(tank_asset_path).convert_alpha()
        tank_image = pygame.transform.scale(raw_tank_image, (TILE_SIZE, TILE_SIZE))
        print(f"  [OK] Wczytano asset czołgu: tnk1.png")

        # --- DODANE: Wczytywanie i kolorowanie maski ---
        mask_asset_path = os.path.join(current_dir, 'frontend', 'assets', 'tanks', 'light_tank', 'msk1.png')
        raw_mask_image = pygame.image.load(mask_asset_path).convert_alpha()
        mask_image = pygame.transform.scale(raw_mask_image, (TILE_SIZE, TILE_SIZE))

        # Stwórz warstwę koloru o rozmiarze maski
        color_layer = pygame.Surface(mask_image.get_size())
        color_layer.fill(TEAM_COLOR)

        # Nałóż maskę na warstwę koloru w trybie mnożenia (BLEND_RGB_MULT).
        # Białe piksele maski (wartość 1) przyjmą kolor TEAM_COLOR.
        # Czarne piksele maski (wartość 0) pozostaną czarne.
        color_layer.blit(mask_image, (0, 0), special_flags=pygame.BLEND_RGB_MULT)

        # Ustaw czarny jako kolor przezroczysty, aby widoczny był tylko pokolorowany wzór.
        color_layer.set_colorkey((0, 0, 0))
        colored_mask_image = color_layer
        print(f"  [OK] Wczytano i pokolorowano maskę: msk1.png")

        # --- DODANE: Wczytywanie grafiki wieżyczki ---
        turret_asset_path = os.path.join(current_dir, 'frontend', 'assets', 'tanks', 'light_tank', 'tnk2.png')
        raw_turret_image = pygame.image.load(turret_asset_path).convert_alpha()        
        original_turret_size = raw_turret_image.get_size()
        turret_image = pygame.transform.scale(raw_turret_image, (TILE_SIZE, TILE_SIZE))
        print(f"  [OK] Wczytano asset wieżyczki: tnk2.png")

        # --- DODANE: Wczytywanie i kolorowanie maski wieżyczki ---
        turret_mask_asset_path = os.path.join(current_dir, 'frontend', 'assets', 'tanks', 'light_tank', 'msk2.png')
        raw_turret_mask_image = pygame.image.load(turret_mask_asset_path).convert_alpha()
        turret_mask_image = pygame.transform.scale(raw_turret_mask_image, (TILE_SIZE, TILE_SIZE))

        turret_color_layer = pygame.Surface(turret_mask_image.get_size())
        turret_color_layer.fill(TEAM_COLOR)
        turret_color_layer.blit(turret_mask_image, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
        turret_color_layer.set_colorkey((0, 0, 0))
        colored_turret_mask_image = turret_color_layer
        print(f"  [OK] Wczytano i pokolorowano maskę wieżyczki: msk2.png")        

        # --- DODANE: Obliczenie pivotu wieżyczki ---
        # Oryginalny pivot to (65, 48) w obrazku o oryginalnym rozmiarze
        original_pivot = pygame.math.Vector2(75, 46)
        scale_factor_x = TILE_SIZE / original_turret_size[0]
        scale_factor_y = TILE_SIZE / original_turret_size[1]
        scaled_pivot = pygame.math.Vector2(original_pivot.x * scale_factor_x, original_pivot.y * scale_factor_y)

        # Wektor od środka przeskalowanego obrazka do jego pivotu
        turret_center_in_surface = pygame.math.Vector2(turret_image.get_rect().center)
        pivot_offset = scaled_pivot - turret_center_in_surface

    except (pygame.error, FileNotFoundError) as e:
        print(f"  [!] Ostrzeżenie: Nie udało się wczytać grafiki czołgu lub maski: {e}")

    # Pozycja czołgu na siatce (np. 5 kolumna, 5 wiersz)
    tank_grid_pos = (5, 5)
    # --- ZMIENIONE: Zmienne kątów zgodne z silnikiem ---
    hull_heading = 0.0  # Kąt kadłuba
    barrel_angle = 0.0  # Kąt lufy (względem kadłuba)

    # --- DODANE: Stan gry dla power-upów ---
    powerups = []
    current_tick = 0

    # --- Główna Pętla ---
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False

        current_tick += 1

        # --- DODANE: Spawnowanie power-upów ---
        if len(powerups) < MAX_POWERUPS_ON_MAP and \
           current_tick > POWERUP_SPAWN_START_TICK and \
           (current_tick - POWERUP_SPAWN_START_TICK) % POWERUP_SPAWN_INTERVAL == 0:
            
            spawn_successful = False
            for _ in range(50): # 50 prób na znalezienie miejsca
                # Losowa pozycja na mapie (współrzędne pixelowe)
                pos_x = random.uniform(POWERUP_SIZE / 2, screen_width - POWERUP_SIZE / 2)
                pos_y = random.uniform(POWERUP_SIZE / 2, screen_height - POWERUP_SIZE / 2)
                
                candidate_rect = pygame.Rect(0, 0, POWERUP_SIZE, POWERUP_SIZE)
                candidate_rect.center = (pos_x, pos_y)

                # Sprawdź kolizję z przeszkodami
                collision = False
                for obstacle in map_info.obstacle_list:
                    # obstacle.position to środek kafelka
                    obstacle_rect = pygame.Rect(0, 0, obstacle.size[0], obstacle.size[1])
                    obstacle_rect.center = (obstacle.position.x, obstacle.position.y)
                    if candidate_rect.colliderect(obstacle_rect):
                        collision = True
                        break
                if collision: continue

                # Sprawdź kolizję z czołgiem
                tank_center_x = tank_grid_pos[0] * TILE_SIZE + TILE_SIZE / 2
                tank_center_y = tank_grid_pos[1] * TILE_SIZE + TILE_SIZE / 2
                tank_rect = pygame.Rect(0, 0, TILE_SIZE, TILE_SIZE)
                tank_rect.center = (tank_center_x, tank_center_y)
                if candidate_rect.colliderect(tank_rect):
                    continue

                # Sprawdź kolizję z innymi power-upami
                collision = False
                for p in powerups:
                    if candidate_rect.colliderect(p['rect']):
                        collision = True
                        break
                if collision: continue

                # Miejsce jest dobre, tworzymy power-up
                powerup_type = random.choice(POWERUP_TYPES)
                powerup_asset = powerup_assets.get(powerup_type)
                if powerup_asset:
                    powerups.append({
                        'type': powerup_type,
                        'surface': powerup_asset,
                        'rect': candidate_rect
                    })
                    print(f"  [+] Zespawnowano power-up: {powerup_type} na pozycji {candidate_rect.center}")
                    spawn_successful = True
                    break # Wyjdź z pętli prób
            
            if not spawn_successful:
                print("  [!] Nie udało się znaleźć miejsca na spawn power-upa.")

        # --- ZMIENIONE: Obsługa klawiszy zgodna z logiką silnika (żądanie zmiany kąta) ---
        heading_delta_request = 0.0
        barrel_delta_request = 0.0

        # Kadłub: A/D
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            heading_delta_request = ROTATION_SPEED  # Żądanie obrotu w lewo (CCW)
        if keys[pygame.K_d]:
            heading_delta_request = -ROTATION_SPEED # Żądanie obrotu w prawo (CW)
        # Wieżyczka: Strzałki
        if keys[pygame.K_LEFT]:
            barrel_delta_request = ROTATION_SPEED   # Żądanie obrotu w lewo (CCW)
        if keys[pygame.K_RIGHT]:
            barrel_delta_request = -ROTATION_SPEED  # Żądanie obrotu w prawo (CW)

        # --- DODANE: Symulacja logiki z physics.py ---
        # Ogranicz żądany obrót do maksymalnej prędkości (spin rate)
        actual_heading_delta = max(-HEADING_SPIN_RATE, min(heading_delta_request, HEADING_SPIN_RATE))
        actual_barrel_delta = max(-BARREL_SPIN_RATE, min(barrel_delta_request, BARREL_SPIN_RATE))

        # Zastosuj obrót i znormalizuj kąt
        hull_heading = normalize_angle(hull_heading + actual_heading_delta)
        barrel_angle = normalize_angle(barrel_angle + actual_barrel_delta)


        screen.fill(BACKGROUND_COLOR)

        # --- ZMIENIONE: Rysowanie mapy na podstawie list obiektów z MapInfo ---
        # Łączymy listy terenu i przeszkód, aby narysować wszystko za jednym razem
        all_map_objects = map_info.terrain_list + map_info.obstacle_list
        for obj in all_map_objects:
            # Pobieramy nazwę klasy obiektu (np. "Wall", "Grass")
            obj_class_name = obj.__class__.__name__
            asset = tile_assets.get(obj_class_name)
            if asset:
                # Pozycja obiektu to jego środek. Musimy obliczyć lewy górny róg.
                pos_x = obj._position.x
                pos_y = obj._position.y
                top_left = (pos_x - asset.get_width() / 2, pos_y - asset.get_height() / 2)
                screen.blit(asset, top_left)

        # --- DODANE: Rysowanie power-upów ---
        for powerup in powerups:
            screen.blit(powerup['surface'], powerup['rect'].topleft)

        # --- DODANE: Rysowanie czołgu na wierzchu mapy ---
        if tank_image:
            # Obracamy oryginalny obraz, aby uniknąć utraty jakości
            rotated_tank = pygame.transform.rotate(tank_image, hull_heading)
            # Obliczamy nową pozycję, aby obrót odbywał się wokół środka
            tank_center_x = tank_grid_pos[0] * TILE_SIZE + TILE_SIZE / 2
            tank_center_y = tank_grid_pos[1] * TILE_SIZE + TILE_SIZE / 2
            new_rect = rotated_tank.get_rect(center=(tank_center_x, tank_center_y))

            # Najpierw rysujemy bazowy czołg
            screen.blit(rotated_tank, new_rect.topleft)

            # --- DODANE: Rysowanie pokolorowanej maski ---
            if colored_mask_image:
                # Obracamy również maskę
                rotated_mask = pygame.transform.rotate(colored_mask_image, hull_heading)
                # Rysujemy ją na tej samej pozycji co czołg, domyślny tryb mieszania nałoży kolor
                screen.blit(rotated_mask, new_rect.topleft)
            
            # --- DODANE: Rysowanie wieżyczki i jej maski ---
            if turret_image:
                # --- MODIFIED: Wieżyczka obraca się razem z kadłubem ---
                final_turret_display_angle = hull_heading + barrel_angle

                # Obracamy wieżyczkę
                rotated_turret = pygame.transform.rotate(turret_image, final_turret_display_angle)
                
                # --- MODIFIED: Obliczenie pozycji z uwzględnieniem pivotu ---
                # Obracamy wektor od środka do pivotu
                rotated_offset = pivot_offset.rotate(-final_turret_display_angle)
                
                # Nowy środek do blitowania to środek czołgu przesunięty o obrócony wektor
                blit_center_pos = pygame.math.Vector2(tank_center_x, tank_center_y) - rotated_offset
                new_turret_rect = rotated_turret.get_rect(center=blit_center_pos)

                # Rysujemy wieżyczkę
                screen.blit(rotated_turret, new_turret_rect.topleft)

                # Rysowanie pokolorowanej maski wieżyczki
                if colored_turret_mask_image:
                    rotated_turret_mask = pygame.transform.rotate(colored_turret_mask_image, final_turret_display_angle)
                    # Maska ma te same wymiary i pivot, więc używamy tego samego rect
                    screen.blit(rotated_turret_mask, new_turret_rect.topleft)

        pygame.display.flip()
        clock.tick(60)

    print("\nZamykanie podglądu mapy.")
    pygame.quit()


if __name__ == '__main__':
    if not os.path.isdir(ASSETS_PATH):
        print("-" * 60 + f"\n!!! OSTRZEŻENIE !!!\nNie znaleziono katalogu z grafikami w: '{ASSETS_PATH}'")
        print("Wszystkie kafelki będą wyświetlane jako białe kwadraty.")
        print("Upewnij się, że ścieżka ASSETS_PATH jest poprawna i że folder istnieje.\n" + "-" * 60)
    
    if not os.path.isdir(POWERUPS_ASSETS_PATH):
        print("-" * 60 + f"\n!!! OSTRZEŻENIE !!!\nNie znaleziono katalogu z grafikami power-upów w: '{POWERUPS_ASSETS_PATH}'")
        print("Wszystkie power-upy będą wyświetlane jako różowe kwadraty.")
        print("Upewnij się, że ścieżka POWERUPS_ASSETS_PATH jest poprawna i że folder istnieje.\n" + "-" * 60)

    main()
