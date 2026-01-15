"""
Game Loop Module - Main game loop implementation
Główna pętla gry, razem z initem
Refactored to use existing structures and improve compatibility
"""

import time
from typing import Any, Dict, List, Optional

from ..structures.map_info import MapInfo
from ..tank.base_tank import Tank
from ..utils.config import TankType
from ..utils.logger import GameEventType, get_logger
from .game_core import GameCore, create_default_game


class GameLoop:
    """Główna klasa pętli gry z fazami inicjalizacji, loop i końca."""

    def __init__(self, config=None, headless: bool = False):
        """
        Inicjalizacja GameLoop.

        Args:
            config: Konfiguracja gry (opcjonalna)
            headless: Czy uruchomić w trybie bez interfejsu graficznego
        """
        self.game_core = GameCore(config) if config else create_default_game()
        self.logger = get_logger()
        self.headless = headless
        self.renderer = None

        if not self.headless:
            try:
                from .renderer import GameRenderer
                self.renderer = GameRenderer()
            except ImportError:
                self.logger.warning(
                    "Pygame nie jest zainstalowany lub renderer nie mógł zostać zaimportowany. Wymuszanie trybu headless."
                )
                self.headless = True

        # Komponenty silnika - placeholders for future implementation
        self.map_loader = None
        self.physics_engine = None
        self.visibility_engine = None

        # Stan gry
        self.map_info: Optional[MapInfo] = None
        self.tanks: Dict[str, Tank] = {}
        self.agents: Dict[str, Any] = {}  # Agent controllers
        self.powerups: Dict[str, Any] = {}
        self.projectiles: List[Any] = []

        # Metryki wydajności
        self.tick_start_time = 0.0
        self.performance_data = {
            "total_ticks": 0,
            "avg_tick_time": 0.0,
            "agent_response_times": {},
        }

    def initialize_game(
        self, map_seed: Optional[str] = None, agent_modules: Optional[List] = None
    ) -> bool:
        """
        Faza 1: Inicjalizacja gry.

        Args:
            map_seed: Seed dla generacji mapy
            agent_modules: Lista modułów agentów

        Returns:
            True jeśli inicjalizacja się powiodła
        """
        try:
            self.logger.info("Starting game initialization...")

            self.logger.debug("Step 1: Initializing GameCore...")
            init_result = self.game_core.initialize_game(map_seed)
            if not init_result["success"]:
                self.logger.error(
                    f"Game core initialization failed: {init_result.get('error')}"
                )
                return False
            self.logger.debug("Step 1: GameCore initialized.")

            self.logger.debug("Step 2: Initializing engine components...")
            self._initialize_engines()
            self.logger.debug("Step 2: Engine components initialized.")

            self.logger.debug("Step 3: Loading map...")
            if not self._load_map(map_seed):
                self.logger.error("Map loading failed")
                return False
            self.logger.debug("Step 3: Map loaded.")

            self.logger.debug("Step 4: Initializing renderer...")
            if not self.headless and self.renderer:
                # Pobieramy dane potrzebne rendererowi, unikając przekazywania całego obiektu MapInfo,
                # który ma niespójną strukturę (dynamicznie dodawane pole grid_data).
                grid = getattr(self.map_info, 'grid_data', None)
                size = tuple(self.map_info.size) if self.map_info and hasattr(self.map_info, 'size') else (500, 500)
                if not self.renderer.initialize(map_size=size, grid_data=grid):
                    self.logger.error("Renderer initialization failed.")
                    return False
            self.logger.debug("Step 4: Renderer initialized.")

            self.logger.debug("Step 5: Spawning tanks...")
            if not self._spawn_tanks():
                self.logger.error("Tank spawning failed")
                return False
            self.logger.debug("Step 5: Tanks spawned.")

            self.logger.debug("Step 6: Loading agents...")
            if agent_modules and not self._load_agents(agent_modules):
                self.logger.error("Agent loading failed")
                return False
            self.logger.debug("Step 6: Agents loaded.")

            self.logger.info("Game initialization completed successfully")
            return True

        except Exception as e:
            import traceback
            self.logger.error(f"Game initialization failed with an unhandled exception: {e}\n{traceback.format_exc()}")
            return False

    def run_game_loop(self) -> Dict[str, Any]:
        """
        Faza 2: Główna pętla gry.

        Returns:
            Wyniki gry
        """
        self.logger.info("Starting main game loop...")

        # Rozpoczęcie pętli gry
        if not self.game_core.start_game_loop():
            return {"success": False, "error": "Failed to start game loop"}

        game_results = {"success": True}

        try:
            # Główna pętla: While(one team alive)
            while self.game_core.can_continue_game():
                tick_start_time = time.time()

                # Przetworzenie tick'a
                tick_info = self._process_game_tick()

                # Pomiar wydajności
                tick_duration = time.time() - tick_start_time
                self.logger.log_tick_end(
                    self.game_core.get_current_tick(), tick_duration
                )
                self._update_performance_metrics(tick_duration)

                # Sprawdzenie warunków zakończenia
                if not tick_info["game_continues"]:
                    break

                # Renderowanie klatki, jeśli nie jesteśmy w trybie headless
                if not self.headless:
                    if not self.renderer.render(
                        tanks=self.tanks,
                        projectiles=self.projectiles,
                        powerups=self.powerups,
                    ):
                        self.logger.info("Okno renderera zostało zamknięte przez użytkownika.")
                        # Przerywamy grę, tak jakby użytkownik wcisnął Ctrl+C
                        raise KeyboardInterrupt
                else:  # W trybie headless możemy chcieć małego opóźnienia, by nie zużywać 100% CPU
                    self._limit_fps(tick_duration)

            # Zakończenie gry
            game_results = self.game_core.end_game("normal")
            self.logger.info(
                f"Game completed after {game_results['total_ticks']} ticks"
            )

        except KeyboardInterrupt:
            self.logger.info("Game interrupted by user")
            game_results = self.game_core.end_game("interrupted")
        except Exception as e:
            self.logger.error(f"Game loop failed with exception: {e}")
            game_results = self.game_core.end_game("error")
            game_results["error"] = str(e)

        return game_results

    def cleanup_game(self):
        """
        Faza 3: Zakończenie i sprzątanie.
        """
        self.logger.info("Starting game cleanup...")

        try:
            # Zakończenie agentów
            self._cleanup_agents()

            # Czyszczenie zasobów
            self._cleanup_resources()

            # Generowanie raportu wydajności
            self._generate_performance_report()

            # Sprzątanie renderera
            if self.renderer:
                self.renderer.cleanup()

            self.logger.info("Game cleanup completed")

        except Exception as e:
            import traceback
            self.logger.error(f"Error during cleanup: {e}\n{traceback.format_exc()}")

    def _process_game_tick(self) -> Dict[str, Any]:
        """
        Przetworzenie pojedynczego tick'a zgodnie ze specyfikacją.

        Kolejność zgodna z Main game loop (Logika Silnika):
        1. Inkrementacja tick'a
        2. Sudden death check
        3. Spawn power-upów
        4. Przygotowanie sensor_data
        5. Wysłanie zapytań do agentów
        6. Odebranie komend
        7. Przetworzenie logiki w kolejności:
           a. Obrót wieżyczek i kadłubów
           b. Zmiana amunicji / Przeładowanie
           c. Strzały
           d. Detekcja trafień pocisków
           e. Ruch czołgów i kolizje
           f. Sprawdzenie warunków śmierci
        """
        # 1. Przetworzenie tick'a w game core
        tick_info = self.game_core.process_tick()
        current_tick = tick_info["tick"]

        self.logger.log_tick_start(current_tick)

        # 2. Sudden death - obrażenia dla wszystkich czołgów
        if tick_info["sudden_death"]:
            self._apply_sudden_death_damage()

        # 3. Spawn power-upów
        if tick_info["powerup_spawned"]:
            self._spawn_powerups()

        # 4. Przygotowanie sensor_data dla każdego czołgu
        sensor_data_map = self._prepare_sensor_data()

        # 5. Wysłanie zapytań do agentów i odebranie odpowiedzi
        agent_actions = self._query_agents(sensor_data_map)

        # 6. Przetworzenie logiki w określonej kolejności
        self._process_tank_rotations(agent_actions)  # a. Obroty
        self._process_ammo_changes(agent_actions)  # b. Amunicja/Przeładowanie
        self._process_shooting(agent_actions)  # c. Strzały
        self._process_projectile_hits()  # d. Detekcja trafień
        self._process_tank_movement(agent_actions)  # e. Ruch i kolizje
        self._check_death_conditions()  # f. Warunki śmierci

        # 7. Aktualizacja liczników zespołów
        self._update_team_counts()

        return tick_info

    def _initialize_engines(self):
        """Inicjalizacja komponentów silnika."""
        self.logger.debug("Initializing game engines...")

        # TODO: Inicjalizacja rzeczywistych komponentów gdy będą dostępne
        # self.physics_engine = PhysicsEngine(self.game_core.config)
        # self.visibility_engine = VisibilityEngine(self.game_core.config)
        # self.map_loader = MapLoader()

        self.logger.debug("Game engines initialized")

    def _load_map(self, map_seed: Optional[str] = None) -> bool:
        """
        Ładowanie i tworzenie mapy.

        Args:
            map_seed: Seed dla generacji mapy

        Returns:
            True jeśli ładowanie się powiodło
        """
        try:
            self.logger.info(f"Loading map with seed: {map_seed}")

            # TODO: Implementacja ładowania mapy gdy MapLoader będzie dostępny
            # self.map_info = self.map_loader.load_map(map_seed)

            # --- TYMCZASOWE TWORZENIE MAPY DLA DEMO UI ---
            map_width = self.game_core.config.map_config.width
            map_height = self.game_core.config.map_config.height
            tile_count_x = map_width // 16
            tile_count_y = map_height // 16

            # Prosta mapa: trawa z ramką ze ścian
            grid = [['Wall' if x == 0 or x == tile_count_x - 1 or y == 0 or y == tile_count_y - 1 else 'Grass' for x in range(tile_count_x)] for y in range(tile_count_y)]

            # Tworzymy obiekt MapInfo zgodnie z nową, uproszczoną definicją
            self.map_info = MapInfo(
                map_seed=map_seed or "default_seed",
                size=(map_width, map_height),
                grid_data=grid
            )
            self.logger.info("Utworzono tymczasową mapę na potrzeby demonstracji UI.")

            try:
                self.logger.log_game_event(
                    GameEventType.MAP_LOAD,
                    f"Map loaded successfully with seed: {map_seed}",
                    map_seed=map_seed,
                )
            except (ImportError, AttributeError):
                self.logger.info(f"Map loaded successfully with seed: {map_seed}")

            return True

        except Exception as e:
            import traceback
            self.logger.error(f"Failed to load map: {e}\n{traceback.format_exc()}")
            return False

    def _spawn_tanks(self) -> bool:
        """
        Spawn czołgów zgodnie z zasadami.

        Returns:
            True jeśli spawn się powiódł
        """
        try:
            self.logger.info("Spawning tanks...")
            
            try:
                self.logger.debug("Importing tank classes for spawning...")
                from ..tank.light_tank import LightTank
                from ..tank.heavy_tank import HeavyTank
                from ..structures.position import Position
                self.logger.debug("Tank classes imported successfully.")
            except ImportError:
                import traceback
                self.logger.error(f"CRITICAL: Failed to import tank classes.\n{traceback.format_exc()}")
                return False

            # --- TYMCZASOWE TWORZENIE CZOŁGÓW DLA DEMO UI (z rozszerzonym debugowaniem) ---
            try:
                self.logger.info("Attempting to spawn LightTank...")
                tank1_id = "player1"
                tank1 = LightTank(_id=tank1_id, _team=1, position=Position(x=100, y=100))
                tank1.heading = 45.0
                tank1.barrel_angle = -15.0
                self.tanks[tank1_id] = tank1
                self.logger.info(f"Successfully spawned temporary tank: {tank1_id}")
            except Exception as e:
                import traceback
                self.logger.error(f"CRITICAL: Failed to spawn LightTank. Error: {e}\n{traceback.format_exc()}")
                return False

            try:
                self.logger.info("Attempting to spawn HeavyTank...")
                tank2_id = "player2"
                tank2 = HeavyTank(_id=tank2_id, _team=2, position=Position(x=400, y=400))
                tank2.heading = 225.0
                tank2.barrel_angle = 30.0
                self.tanks[tank2_id] = tank2
                self.logger.info(f"Successfully spawned temporary tank: {tank2_id}")
            except Exception as e:
                import traceback
                self.logger.error(f"CRITICAL: Failed to spawn HeavyTank. Error: {e}\n{traceback.format_exc()}")
                return False

            return True

        except Exception as e:
            import traceback
            self.logger.error(f"CRITICAL: An unexpected error occurred in _spawn_tanks.\n{traceback.format_exc()}")
            return False

    def _load_agents(self, agent_modules: List) -> bool:
        """
        Ładowanie modułów agentów.

        Args:
            agent_modules: Lista modułów agentów do załadowania

        Returns:
            True jeśli ładowanie się powiodło
        """
        try:
            self.logger.info(f"Loading {len(agent_modules)} agent modules...")

            for i, agent_module in enumerate(agent_modules):
                agent_id = f"agent_{i + 1}"

                # TODO: Implementacja ładowania agentów gdy API będzie gotowe
                # agent_controller = self._load_agent_module(agent_module)
                # self.agents[agent_id] = agent_controller

                self.logger.debug(f"Loaded agent: {agent_id}")

            self.logger.info("All agents loaded successfully")
            return True

        except Exception as e:
            import traceback
            self.logger.error(f"Failed to load agents: {e}\n{traceback.format_exc()}")
            return False

    def _apply_sudden_death_damage(self):
        """Aplikuje obrażenia nagłej śmierci wszystkim czołgom."""
        damage = abs(self.game_core.get_sudden_death_damage())  # Convert to positive

        for tank_id, tank in self.tanks.items():
            # TODO: Aplikacja obrażeń gdy Tank.take_damage będzie dostępne
            # tank.take_damage(damage)
            pass

        self.logger.debug(f"Applied sudden death damage: {damage} to all tanks")

    def _spawn_powerups(self):
        """Spawn power-upów zgodnie z zasadami."""
        powerup_config = self.game_core.get_powerup_config()

        # Sprawdzenie czy nie przekroczono limitu power-upów
        if len(self.powerups) >= powerup_config["max_powerups"]:
            return

        # TODO: Implementacja spawnu power-upów gdy będą gotowe
        # powerup = self._create_random_powerup()
        # self.powerups[powerup.id] = powerup

        self.logger.log_powerup_action(
            "powerup_new", "spawn", {"type": "random", "count": len(self.powerups)}
        )

    def _prepare_sensor_data(self) -> Dict[str, Any]:
        """
        Przygotowanie danych sensorycznych dla każdego czołgu.

        Returns:
            Mapa sensor_data dla każdego czołgu
        """
        sensor_data_map = {}

        for tank_id, tank in self.tanks.items():
            # TODO: Implementacja przygotowania sensor_data gdy VisibilityEngine będzie gotowy
            # sensor_data = self.visibility_engine.prepare_sensor_data(
            #     tank, self.map_info, self.tanks, self.powerups
            # )
            # sensor_data_map[tank_id] = sensor_data
            pass

        return sensor_data_map

    def _query_agents(self, sensor_data_map: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wysłanie zapytań do agentów i odebranie odpowiedzi.

        Args:
            sensor_data_map: Dane sensoryczne dla każdego czołgu

        Returns:
            Mapa akcji od agentów
        """
        agent_actions = {}
        for tank_id, sensor_data in sensor_data_map.items():
            agent_id = self._get_agent_for_tank(tank_id)
            if agent_id not in self.agents:
                continue

            try:
                # Pomiar czasu odpowiedzi
                request_start = time.time()

                self.logger.log_agent_interaction(agent_id, "request", tank_id=tank_id)

                # TODO: Implementacja zapytania do agenta gdy API będzie gotowe
                # tank_status = self._get_tank_status(tank_id)
                # enemies_remaining = self._count_enemies(tank_id)
                # action = self.agents[agent_id].get_action(
                #     current_tick, tank_status, sensor_data, enemies_remaining
                # )

                response_time = time.time() - request_start

                # agent_actions[tank_id] = action
                self.logger.log_agent_interaction(
                    agent_id, "response", response_time=response_time, tank_id=tank_id
                )

            except Exception as e:
                self.logger.log_agent_interaction(
                    agent_id, "timeout", error=str(e), tank_id=tank_id,
                    
                )

        return agent_actions

    def _process_tank_rotations(self, agent_actions: Dict[str, Any]):
        """Przetworzenie obrotów wieżyczek i kadłubów."""
        for tank_id, action in agent_actions.items():
            # TODO: Implementacja obrotów gdy Tank będzie miał odpowiednie metody
            # if tank_id in self.tanks:
            #     tank = self.tanks[tank_id]
            #     if hasattr(action, 'barrel_rotation'):
            #         tank.rotate_barrel(action.barrel_rotation, 1.0/60.0)  # Assuming 60 FPS
            #     if hasattr(action, 'heading_rotation'):
            #         tank.rotate_heading(action.heading_rotation, 1.0/60.0)
            pass

    def _process_ammo_changes(self, agent_actions: Dict[str, Any]):
        """Przetworzenie zmiany amunicji i przeładowania."""
        for tank_id, action in agent_actions.items():
            # TODO: Implementacja zmiany amunicji
            # if tank_id in self.tanks:
            #     tank = self.tanks[tank_id]
            #     if hasattr(action, 'ammo_type'):
            #         tank.ammo_loaded = action.ammo_type
            pass

    def _process_shooting(self, agent_actions: Dict[str, Any]):
        """Przetworzenie strzałów."""
        for tank_id, action in agent_actions.items():
            # TODO: Implementacja strzałów
            # if tank_id in self.tanks:
            #     tank = self.tanks[tank_id]
            #     if hasattr(action, 'shoot') and action.shoot:
            #         damage = tank.shoot()
            #         if damage:
            #             projectile = self._create_projectile(tank, damage)
            #             self.projectiles.append(projectile)
            pass

    def _process_projectile_hits(self):
        """Detekcja trafień pocisków."""
        # TODO: Implementacja detekcji trafień gdy fizyka będzie gotowa
        # for projectile in self.projectiles[:]:
        #     hit_target = self.physics_engine.check_projectile_collision(projectile)
        #     if hit_target:
        #         hit_target.take_damage(projectile.damage)
        #         self.projectiles.remove(projectile)
        pass

    def _process_tank_movement(self, agent_actions: Dict[str, Any]):
        """Przetworzenie ruchu czołgów i kolizji."""
        for tank_id, action in agent_actions.items():
            # TODO: Implementacja ruchu i kolizji
            # if tank_id in self.tanks:
            #     tank = self.tanks[tank_id]
            #     if hasattr(action, 'move_speed'):
            #         tank.set_move_speed(action.move_speed)
            #         # Move tank and check collisions
            #         collision = self.physics_engine.move_tank_with_collision_check(tank)
            #         if collision:
            #             self._handle_collision(tank, collision)
            pass

    def _check_death_conditions(self):
        """Sprawdzenie warunków śmierci czołgów."""
        tanks_to_remove = []

        for tank_id, tank in self.tanks.items():
            # TODO: Sprawdzenie HP <= 0 gdy Tank będzie miał odpowiednie metody
            # if not tank.is_alive():
            #     tanks_to_remove.append(tank_id)
            #     self.logger.log_tank_action(tank_id, "death", {'final_hp': tank.hp})
            pass

        # Usunięcie martwych czołgów
        for tank_id in tanks_to_remove:
            del self.tanks[tank_id]

    def _update_team_counts(self):
        """Aktualizacja liczby żywych czołgów w zespołach."""
        team_counts = {}

        for tank_id, tank in self.tanks.items():
            # Zakładamy, że po refaktoryzacji czołg ma publiczny atrybut 'team'
            team = getattr(tank, 'team', None)
            if team is not None:
                team_counts[team] = team_counts.get(team, 0) + 1

        # Aktualizacja w game core
        for team, count in team_counts.items():
            self.game_core.update_team_count(team, count)

    def _select_random_tank_type(self, tank_types: List[TankType]) -> TankType:
        """Losowy wybór typu czołgu."""
        import random

        return random.choice(tank_types)

    def _get_agent_for_tank(self, tank_id: str) -> str:
        """Pobranie ID agenta dla danego czołgu."""
        # TODO: Implementacja mapowania czołg -> agent
        return f"agent_{tank_id.split('_')[1]}"

    def _count_enemies(self, tank_id: str) -> int:
        """Liczenie wrogich czołgów dla danego czołgu."""
        # TODO: Implementacja liczenia wrogów gdy Tank będzie miał team property
        return len(self.tanks) - 1

    def _cleanup_agents(self):
        """Zakończenie pracy agentów."""
        for agent_id, agent in self.agents.items():
            try:
                # TODO: Wywołanie agent.end() gdy API będzie gotowe
                # agent.end()
                pass
            except Exception as e:
                import traceback
                self.logger.error(f"Error ending agent {agent_id}: {e}\n{traceback.format_exc()}")

    def _cleanup_resources(self):
        """Czyszczenie zasobów gry."""
        self.tanks.clear()
        self.agents.clear()
        self.powerups.clear()
        self.projectiles.clear()

    def _update_performance_metrics(self, tick_duration: float):
        """Aktualizacja metryk wydajności."""
        self.performance_data["total_ticks"] += 1

        # Obliczanie średniego czasu tick'a
        if self.performance_data["avg_tick_time"] == 0:
            self.performance_data["avg_tick_time"] = tick_duration
        else:
            # Średnia ruchoma
            alpha = 0.1
            self.performance_data["avg_tick_time"] = (
                alpha * tick_duration
                + (1 - alpha) * self.performance_data["avg_tick_time"]
            )

    def _generate_performance_report(self):
        """Generowanie raportu wydajności."""
        try:
            # Try to get performance report from logger if available
            report = self.logger.get_performance_report()
            self.logger.info(f"Performance report: {report}")
        except (AttributeError, Exception):
            # Fallback to basic performance data
            self.logger.info(f"Performance data: {self.performance_data}")

    def _limit_fps(self, tick_duration: float, target_fps: int = 200):
        """Ograniczenie FPS w trybie headless, aby nie zużywać 100% CPU."""
        target_tick_time = 1.0 / target_fps
        if tick_duration < target_tick_time:
            time.sleep(target_tick_time - tick_duration)


def run_game(
    config=None,
    map_seed: Optional[str] = None,
    agent_modules: Optional[List] = None,
    headless: bool = False,
) -> Dict[str, Any]:
    """
    Główna funkcja uruchamiająca pełną grę.

    Args:
        config: Konfiguracja gry
        map_seed: Seed mapy
        agent_modules: Lista modułów agentów
        headless: Tryb bez GUI

    Returns:
        Wyniki gry
    """
    game_loop = GameLoop(config, headless)

    try:
        # Faza 1: Inicjalizacja
        if not game_loop.initialize_game(map_seed, agent_modules):
            return {"success": False, "error": "Initialization failed"}

        # Faza 2: Główna pętla
        results = game_loop.run_game_loop()

        # Faza 3: Zakończenie
        game_loop.cleanup_game()

        return results

    except Exception as e:
        import traceback
        game_loop.logger.error(f"Game execution failed: {e}\n{traceback.format_exc()}")
        game_loop.cleanup_game()
        return {"success": False, "error": str(e)}
