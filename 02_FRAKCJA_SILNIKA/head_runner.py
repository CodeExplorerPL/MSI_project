"""
Skrypt do uruchamiania pełnej symulacji gry w trybie graficznym (headful).

Ten skrypt automatycznie:
1. Uruchamia wymaganą liczbę serwerów agentów w osobnych procesach.
2. Inicjalizuje Pygame i ładuje zasoby graficzne.
3. Uruchamia główną pętlę gry, która łączy logikę silnika z renderowaniem w Pygame.
4. Wyświetla na bieżąco stan gry: pozycje czołgów, strzały, power-upy.
5. Po zakończeniu gry zamyka okno i serwery agentów.
"""

import subprocess
import sys
import os
import time
import pygame
import math
from typing import Dict, Any, List

# --- Konfiguracja Ścieżek ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    from backend.engine.game_loop import GameLoop, TEAM_A_NBR, TEAM_B_NBR, AGENT_BASE_PORT, TankScoreboard
    from backend.utils.logger import set_log_level
    from backend.engine.physics import process_physics_tick
    from controller.api import ActionCommand, AmmoType
    from backend.tank.base_tank import Tank
    from backend.tank.light_tank import LightTank
    from backend.structures.position import Position

except ImportError as e:
    print(f"Błąd importu: {e}")
    print("Upewnij się, że skrypt jest uruchamiany z katalogu '02_FRAKCJA_SILNIKA' lub że struktura projektu jest poprawna.")
    sys.exit(1)

# --- Stałe Konfiguracyjne Grafiki ---
LOG_LEVEL = "INFO"
MAP_SEED = "map1.csv"
TARGET_FPS = 60
SCALE = 2  # Współczynnik skalowania grafiki (wszystko będzie 2x większe)
TILE_SIZE = 10  # To MUSI być zgodne z domyślną wartością w map_loader.py

ASSETS_BASE_PATH = os.path.join(current_dir, 'frontend', 'assets')
TILE_ASSETS_PATH = os.path.join(ASSETS_BASE_PATH, 'tiles')
POWERUP_ASSETS_PATH = os.path.join(ASSETS_BASE_PATH, 'power-ups')
TANK_ASSETS_PATH = os.path.join(ASSETS_BASE_PATH, 'tanks')

BACKGROUND_COLOR = (20, 20, 30)
TEAM_COLORS = {
    1: (50, 150, 255),  # Niebieski
    2: (255, 50, 50)    # Czerwony
}

TANK_ASSET_MAP = {
    "LightTank": "light_tank",
    "HeavyTank": "heavy_tank",
    "SniperTank": "sniper_tank"
}

# --- Funkcje Pomocnicze Renderowania ---

def load_assets():
    """Ładuje wszystkie potrzebne zasoby graficzne."""
    assets = {
        'tiles': {},
        'powerups': {},
        'tanks': {}
    }
    print("--- Ładowanie zasobów graficznych ---")

    # Kafelki
    tile_names = ['Wall', 'Tree', 'AntiTankSpike', 'Grass', 'Road', 'Swamp', 'PotholeRoad', 'Water']
    for name in tile_names:
        try:
            path = os.path.join(TILE_ASSETS_PATH, f"{name}.png")
            img = pygame.image.load(path).convert_alpha()
            # Skalujemy asset do docelowego rozmiaru
            assets['tiles'][name] = pygame.transform.scale(img, (TILE_SIZE * SCALE, TILE_SIZE * SCALE))
        except pygame.error:
            print(f"[!] Nie znaleziono assetu dla kafelka: {name}")

    # Power-upy
    powerup_names = ['Medkit', 'Shield', 'Overcharge', 'AmmoBox_Heavy', 'AmmoBox_Light', 'AmmoBox_Sniper']
    powerup_render_size = (int(TILE_SIZE * SCALE * 0.8), int(TILE_SIZE * SCALE * 0.8))
    for name in powerup_names:
        try:
            path = os.path.join(POWERUP_ASSETS_PATH, f"{name}.png")
            img = pygame.image.load(path).convert_alpha()
            assets['powerups'][name] = pygame.transform.scale(img, powerup_render_size)
        except pygame.error:
            print(f"[!] Nie znaleziono assetu dla power-upa: {name}")

    # Czołgi
    tank_render_size = (TILE_SIZE * SCALE, TILE_SIZE * SCALE)
    for tank_type, folder_name in TANK_ASSET_MAP.items():
        try:
            base_path = os.path.join(TANK_ASSETS_PATH, folder_name)
            # ROZWIĄZANIE: Pre-rotacja assetów o -90 stopni, aby dopasować je do systemu fizyki (0 = Wschód)
            # Dzięki temu Północne grafiki zachowują się jak Wschodnie.
            assets['tanks'][tank_type] = {
                'body': pygame.transform.rotate(pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'tnk1.png')).convert_alpha(), tank_render_size), -90),
                'mask_body': pygame.transform.rotate(pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'msk1.png')).convert_alpha(), tank_render_size), -90),
                'turret': pygame.transform.rotate(pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'tnk2.png')).convert_alpha(), tank_render_size), -90),
                'mask_turret': pygame.transform.rotate(pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'msk2.png')).convert_alpha(), tank_render_size), -90),
            }
        except pygame.error:
            print(f"[!] Nie znaleziono assetów dla czołgu: {tank_type}")

    print("--- Ładowanie zakończone ---")
    return assets

def draw_tank(surface: pygame.Surface, tank: Tank, assets: Dict, scale: int):
    """Rysuje pojedynczy czołg na ekranie z uwzględnieniem skali."""
    tank_assets = assets['tanks'].get(tank._tank_type)
    if not tank_assets:
        return

    team_color = TEAM_COLORS.get(tank.team, (255, 255, 255))

    # Przeskalowana pozycja środka czołgu
    center_pos = (tank.position.x * scale, tank.position.y * scale)

    # --- Kadłub ---
    body_img = tank_assets['body']
    rotated_body = pygame.transform.rotate(body_img, tank.heading)
    body_rect = rotated_body.get_rect(center=center_pos)
    surface.blit(rotated_body, body_rect.topleft)

    # Maska koloru kadłuba
    mask_body_img = tank_assets['mask_body']
    color_layer = pygame.Surface(mask_body_img.get_size(), pygame.SRCALPHA)
    color_layer.fill(team_color) # Użyj koloru drużyny
    color_layer.blit(mask_body_img, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    rotated_mask = pygame.transform.rotate(color_layer, tank.heading)
    surface.blit(rotated_mask, body_rect.topleft)

    # --- Wieża ---
    turret_img = tank_assets['turret']
    turret_angle = tank.heading + tank.barrel_angle
    rotated_turret = pygame.transform.rotate(turret_img, turret_angle)
    turret_rect = rotated_turret.get_rect(center=center_pos)
    surface.blit(rotated_turret, turret_rect.topleft)

    # Maska koloru wieży
    mask_turret_img = tank_assets['mask_turret']
    turret_color_layer = pygame.Surface(mask_turret_img.get_size(), pygame.SRCALPHA)
    turret_color_layer.fill(team_color) # Użyj koloru drużyny
    turret_color_layer.blit(mask_turret_img, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    rotated_turret_mask = pygame.transform.rotate(turret_color_layer, turret_angle)
    surface.blit(rotated_turret_mask, turret_rect.topleft)

    # --- Pasek HP ---
    hp_bar_width = 40
    hp_bar_height = 5
    hp_ratio = max(0, tank.hp / tank._max_hp)
    # Pozycjonowanie paska HP nad czołgiem
    hp_bar_x = center_pos[0] - hp_bar_width / 2
    hp_bar_y = center_pos[1] - (tank_assets['body'].get_height() / 2) - 10
    pygame.draw.rect(surface, (50, 50, 50), (hp_bar_x, hp_bar_y, hp_bar_width, hp_bar_height))
    pygame.draw.rect(surface, (0, 255, 0), (hp_bar_x, hp_bar_y, hp_bar_width * hp_ratio, hp_bar_height))

def draw_shot_effect(surface: pygame.Surface, start_pos: Dict, end_pos: Dict, life: int, scale: int):
    """Rysuje linię symbolizującą strzał z uwzględnieniem skali."""
    if life > 0:
        alpha = int(255 * (life / 10.0)) # Efekt zanikania
        color = (255, 255, 0, alpha)
        line_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        # Skalowanie pozycji
        scaled_start = (start_pos.x * scale, start_pos.y * scale)
        scaled_end = (end_pos.x * scale, end_pos.y * scale)
        pygame.draw.line(line_surface, color, scaled_start, scaled_end, 2)
        surface.blit(line_surface, (0, 0))

def create_background_surface(map_info: Any, assets: Dict, scale: int, width: int, height: int) -> pygame.Surface:
    """Tworzy i zwraca powierzchnię z narysowaną statyczną mapą (teren + przeszkody)."""
    print("--- Tworzenie pre-renderowanego tła mapy ---")
    background = pygame.Surface((width, height))
    background.fill(BACKGROUND_COLOR)

    # Rysowanie terenu i przeszkód
    all_map_objects = map_info.terrain_list + map_info.obstacle_list
    for obj in all_map_objects:
        obj_class_name = obj.__class__.__name__
        asset = assets['tiles'].get(obj_class_name)
        if asset:
            # Pozycja obiektu to jego środek. Skalujemy ją.
            pos_x = obj._position.x * scale
            pos_y = obj._position.y * scale
            # Obliczamy lewy górny róg na podstawie przeskalowanego środka i rozmiaru assetu
            top_left = (pos_x - asset.get_width() / 2, pos_y - asset.get_height() / 2)
            background.blit(asset, top_left)
    
    print("--- Tło mapy utworzone ---")
    return background

def draw_ui(screen: pygame.Surface, font: pygame.font.Font, game_loop: GameLoop, window_width: int, map_rect: pygame.Rect):
    """Rysuje interfejs użytkownika na bocznych panelach."""
    
    # Statystyki drużyn
    team1_alive = sum(1 for t in game_loop.tanks.values() if t.team == 1 and t.is_alive())
    team1_kills = sum(s.tanks_killed for s in game_loop.scoreboards.values() if s.team == 1)
    
    team2_alive = sum(1 for t in game_loop.tanks.values() if t.team == 2 and t.is_alive())
    team2_kills = sum(s.tanks_killed for s in game_loop.scoreboards.values() if s.team == 2)

    # --- Panel lewy (Team 1) ---
    panel1_x = map_rect.left / 2
    
    title1_surf = font.render("TEAM 1", True, TEAM_COLORS[1])
    title1_rect = title1_surf.get_rect(center=(panel1_x, 100))
    screen.blit(title1_surf, title1_rect)

    alive1_surf = font.render(f"Alive: {team1_alive}", True, (200, 200, 200))
    alive1_rect = alive1_surf.get_rect(center=(panel1_x, 150))
    screen.blit(alive1_surf, alive1_rect)

    kills1_surf = font.render(f"Kills: {team1_kills}", True, (200, 200, 200))
    kills1_rect = kills1_surf.get_rect(center=(panel1_x, 180))
    screen.blit(kills1_surf, kills1_rect)

    # --- Panel prawy (Team 2) ---
    panel2_x = map_rect.right + (window_width - map_rect.right) / 2

    title2_surf = font.render("TEAM 2", True, TEAM_COLORS[2])
    title2_rect = title2_surf.get_rect(center=(panel2_x, 100))
    screen.blit(title2_surf, title2_rect)

    alive2_surf = font.render(f"Alive: {team2_alive}", True, (200, 200, 200))
    alive2_rect = alive2_surf.get_rect(center=(panel2_x, 150))
    screen.blit(alive2_surf, alive2_rect)

    kills2_surf = font.render(f"Kills: {team2_kills}", True, (200, 200, 200))
    kills2_rect = kills2_surf.get_rect(center=(panel2_x, 180))
    screen.blit(kills2_surf, kills2_rect)

def draw_debug_info(screen: pygame.Surface, font: pygame.font.Font, clock: pygame.time.Clock, current_tick: int):
    """Rysuje informacje debugowe (FPS, Tick) w lewym górnym rogu."""
    # Użyj mniejszej czcionki dla informacji debugowych
    debug_font = pygame.font.Font(None, 24)
    
    fps_text = f"FPS: {clock.get_fps():.1f}"
    tick_text = f"Tick: {current_tick}"
    
    fps_surf = debug_font.render(fps_text, True, (255, 255, 0))
    tick_surf = debug_font.render(tick_text, True, (255, 255, 0))
    
    screen.blit(fps_surf, (10, 10))
    screen.blit(tick_surf, (10, 30))



def main():
    """Główna funkcja uruchamiająca symulację z grafiką."""
    print("--- Uruchamianie symulacji w trybie graficznym ---")
    set_log_level(LOG_LEVEL)

    agent_processes = []
    total_tanks = TEAM_A_NBR + TEAM_B_NBR
    controller_script_path = os.path.join(current_dir, 'controller', 'server.py')

    if not os.path.exists(controller_script_path):
        print(f"BŁĄD: Nie znaleziono skryptu serwera agenta w: {controller_script_path}")
        return

    # --- Inicjalizacja Gry ---
    game_loop = GameLoop(headless=False)

    try:
        # 1. Uruchomienie serwerów agentów
        print(f"Uruchamianie {total_tanks} serwerów agentów...")
        for i in range(total_tanks):
            port = AGENT_BASE_PORT + i
            command = [sys.executable, controller_script_path, "--port", str(port)]
            proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            agent_processes.append(proc)
            print(f"  -> Agent {i+1} uruchomiony na porcie {port} (PID: {proc.pid})")

        print("\nOczekiwanie 3 sekundy na start serwerów agentów...")
        time.sleep(3)

        # 2. Inicjalizacja silnika gry
        if not game_loop.initialize_game(map_seed=MAP_SEED):
            raise RuntimeError("Inicjalizacja pętli gry nie powiodła się!")

        # --- DODAWANIE CZOŁGU TESTOWEGO ---
        print("\n--- Dodawanie czołgu testowego ---")
        test_tank_id = "tank_test_1"
        test_tank_pos = Position(_x=1.0, _y=1.0)
        # Używamy konstruktora z `start_pos` tak jak w `game_loop.py`
        test_tank = LightTank(_id=test_tank_id, team=1, start_pos=test_tank_pos)
        
        game_loop.tanks[test_tank_id] = test_tank
        if game_loop.map_info:
            game_loop.map_info._all_tanks.append(test_tank)
        game_loop.scoreboards[test_tank_id] = TankScoreboard(tank_id=test_tank_id, team=1)
        print(f"  -> Dodano czołg testowy: {test_tank_id} na pozycji ({test_tank_pos.x}, {test_tank_pos.y})")

        # WAŻNE: Zaktualizuj stan GameCore o nowo stworzone czołgi PRZED pętlą gry!
        game_loop._update_team_counts()

        # 3. Inicjalizacja Pygame i okna 16:9
        pygame.init()
        map_engine_width, map_engine_height = game_loop.map_info._size
        map_render_size = map_engine_width * SCALE

        # Ustaw rozmiar okna w proporcjach 16:9, aby zmieścić mapę i panele boczne
        window_height = map_render_size + 100
        window_width = int(window_height * 16 / 9)

        screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Symulator Walk Czołgów")
        clock = pygame.time.Clock()
        assets = load_assets()
        font = pygame.font.Font(None, 36)

        # Utworzenie powierzchni do rysowania samej mapy
        map_surface = pygame.Surface((map_render_size, map_render_size))
        map_rect = map_surface.get_rect(center=(window_width / 2, window_height / 2))

        # OPTYMALIZACJA: Pre-renderowanie statycznego tła mapy
        background_surface = create_background_surface(game_loop.map_info, assets, SCALE, map_render_size, map_render_size)

        # --- Wyświetlanie informacji o spawnie ---
        print("\n--- Informacje o Spawnie ---")
        print("Zespawnowane czołgi:")
        if game_loop.tanks:
            # Sortowanie dla czytelności
            sorted_tanks = sorted(game_loop.tanks.values(), key=lambda t: t._id)
            for tank in sorted_tanks:
                print(f"  - Czołg: {tank._id} (Team: {tank.team}, Typ: {tank._tank_type}) na pozycji ({tank.position.x:.1f}, {tank.position.y:.1f})")
        else:
            print("  Brak czołgów.")

        print("\nZespawnowane power-upy:")
        if game_loop.map_info and game_loop.map_info.powerup_list:
            for powerup in game_loop.map_info.powerup_list:
                print(f"  - Power-up: {powerup.powerup_type.name} na pozycji ({powerup.position.x:.1f}, {powerup.position.y:.1f})")
        else:
            print("  Brak power-upów na mapie.")

        # --- TEST DIAGNOSTYCZNY: Wyświetlenie zamrożonej mapy i UI ---
        print("\n--- TEST: Wyświetlanie statycznej mapy i interfejsu ---")
        print("--- Naciśnij SPACJĘ, aby rozpocząć symulację ---")

        running = True
        waiting_for_start = True
        while waiting_for_start:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting_for_start = False
                    running = False  # Ustaw flagę wyjścia z gry
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    waiting_for_start = False

            if not running:  # Jeśli użytkownik zamknął okno, wyjdź z pętli oczekiwania
                break

            screen.fill(BACKGROUND_COLOR)
            map_surface.blit(background_surface, (0, 0)) # Narysuj tło na powierzchni mapy
            screen.blit(map_surface, map_rect) # Narysuj powierzchnię mapy na ekranie
            draw_ui(screen, font, game_loop, window_width, map_rect) # Narysuj UI
            draw_debug_info(screen, font, clock, 0) # Pokaż info debugowe
            pygame.display.flip()
            clock.tick(30)

        # Jeśli użytkownik zamknął okno w menu startowym, nie kontynuuj
        if not running:
            raise SystemExit("Wyjście z programu na życzenie użytkownika.")

        print("--- Rozpoczynanie właściwej symulacji... ---")

        # 4. Start pętli w GameCore - kluczowy krok pominięty wcześniej
        if not game_loop.game_core.start_game_loop():
            raise RuntimeError("Nie udało się uruchomić pętli w GameCore!")

        shot_effects = [] # Lista do przechowywania aktywnych efektów strzałów

        # --- Główna Pętla Gry i Renderowania ---
        print("\n--- Rozpoczynanie pętli gry ---")
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Sprawdzenie warunku końca gry
            if not game_loop.game_core.can_continue_game():
                running = False
                continue

            # --- KROK 1: Logika jednego ticka (replikacja z game_loop._process_game_tick) ---

            tick_info = game_loop.game_core.process_tick()
            current_tick = tick_info["tick"]

            if tick_info["sudden_death"]:
                game_loop._apply_sudden_death_damage()

            if tick_info["powerup_spawned"]:
                game_loop._spawn_powerups()

            sensor_data_map = game_loop._prepare_sensor_data()
            agent_actions = game_loop._query_agents(sensor_data_map, current_tick)

            # --- KROK 2: Fizyka (replikacja z game_loop._process_physics) ---

            physics_results = {}
            if game_loop.map_info:
                actions_converted = {}
                for tank_id, action_dict in agent_actions.items():
                    try:
                        ammo_to_load_str = action_dict.get("ammo_to_load")
                        ammo_to_load_type = AmmoType[ammo_to_load_str] if ammo_to_load_str else None
                        actions_converted[tank_id] = ActionCommand(
                            barrel_rotation_angle=action_dict.get("barrel_rotation_angle", 0.0),
                            heading_rotation_angle=action_dict.get("heading_rotation_angle", 0.0),
                            move_speed=action_dict.get("move_speed", 0.0),
                            ammo_to_load=ammo_to_load_type,
                            should_fire=action_dict.get("should_fire", False)
                        )
                    except (KeyError, TypeError):
                        # Ignoruj błędne akcje
                        pass

                all_tanks_list = list(game_loop.tanks.values())
                delta_time = 1.0 / TARGET_FPS

                physics_results = process_physics_tick(
                    all_tanks=all_tanks_list,
                    actions=actions_converted,
                    map_info=game_loop.map_info,
                    delta_time=delta_time
                )

                # Przetwarzanie trafień do scoreboardu
                for hit in physics_results.get("projectile_hits", []):
                    if hit.hit_tank_id:
                        # Znajdź strzelca (uproszczone, jak w oryginale)
                        for tank_id, action in actions_converted.items():
                            if action.should_fire:
                                if tank_id in game_loop.scoreboards:
                                    game_loop.scoreboards[tank_id].damage_dealt += hit.damage_dealt
                                game_loop.last_attacker[hit.hit_tank_id] = tank_id
                                # Dodaj efekt strzału do narysowania
                                shooter_tank = game_loop.tanks.get(tank_id)
                                if shooter_tank:
                                    shot_effects.append({
                                        "start": shooter_tank.position,
                                        "end": hit.hit_position,
                                        "life": 10 # Czas życia efektu w klatkach
                                    })
                                break

            # --- KROK 3: Sprawdzenie zniszczeń i aktualizacja stanu ---
            game_loop._check_death_conditions()
            game_loop._update_team_counts()

            # --- KROK 4: Renderowanie ---
            screen.fill(BACKGROUND_COLOR)

            # Rysuj tło na powierzchni mapy (czyści poprzednią klatkę)
            map_surface.blit(background_surface, (0, 0))

            # Rysowanie power-upów
            for powerup in game_loop.map_info.powerup_list:
                asset = assets['powerups'].get(powerup.powerup_type.name)
                if asset:
                    pos_x = powerup.position.x * SCALE # Używamy SCALE, a nie map_render_size
                    pos_y = powerup.position.y * SCALE
                    top_left = (pos_x - asset.get_width() / 2, pos_y - asset.get_height() / 2)
                    map_surface.blit(asset, top_left)

            # Rysowanie czołgów
            for tank in game_loop.tanks.values():
                draw_tank(map_surface, tank, assets, SCALE)

            # Rysowanie i aktualizacja efektów strzałów
            remaining_shots = []
            for shot in shot_effects:
                # Rysujemy na powierzchni mapy
                draw_shot_effect(map_surface, shot['start'], shot['end'], shot['life'], SCALE)
                shot['life'] -= 1
                if shot['life'] > 0:
                    remaining_shots.append(shot)
            shot_effects = remaining_shots

            # Rysowanie finalnej mapy na środku ekranu i UI po bokach
            screen.blit(map_surface, map_rect)
            draw_ui(screen, font, game_loop, window_width, map_rect)
            draw_debug_info(screen, font, clock, current_tick)

            pygame.display.flip()
            clock.tick(TARGET_FPS)

        # --- Koniec Pętli ---
        print("--- Pętla gry zakończona ---")

        # Wyświetl wyniki w konsoli
        game_results = game_loop.game_core.end_game("normal")
        game_results["scoreboards"] = game_loop._get_final_scoreboards()

        print("\n--- Wyniki Gry ---")
        if game_results.get("winner_team"):
            print(f"🏆 Zwycięzca: Drużyna {game_results.get('winner_team')}")
        else:
            print("🤝 Remis")
        print(f"Całkowita liczba ticków: {game_results.get('total_ticks')}")

        scoreboards = game_results.get("scoreboards", [])
        if scoreboards:
            scoreboards.sort(key=lambda x: (x.get('team', 0), -x.get('tanks_killed', 0)))
            for score in scoreboards:
                print(f"  - Czołg: {score.get('tank_id')}, Drużyna: {score.get('team')}, "
                      f"Zabójstwa: {score.get('tanks_killed')}, Obrażenia: {score.get('damage_dealt', 0):.0f}")

        # Daj chwilę na przeczytanie wyników przed zamknięciem
        time.sleep(5)

    except Exception as e:
        print(f"\n--- KRYTYCZNY BŁĄD W PĘTLI GRY ---")
        import traceback
        traceback.print_exc()

    finally:
        # --- Sprzątanie ---
        print("\n--- Zamykanie zasobów ---")
        game_loop.cleanup_game()

        print("Zamykanie serwerów agentów...")
        for proc in agent_processes:
            proc.terminate()
            print(f"  -> Zatrzymano proces agenta (PID: {proc.pid})")

        pygame.quit()
        print("\n--- Zakończono symulację ---")

if __name__ == "__main__":
    main()
