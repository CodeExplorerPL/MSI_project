"""
Game Loop Module - Main game loop implementation
Główna pętla gry, razem z initem
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from ..structures.map_info import MapInfo
from ..structures.position import Position
from ..tank.base_tank import Tank
from ..tank.sensor_data import SensorData
from ..utils.config import PowerUpType, TankType, game_config
from ..utils.logger import GameEventType, get_logger
from .game_core import GameCore, create_default_game
from .map_loader import MapLoader
from .physics import PhysicsEngine
from .visibility import VisibilityEngine


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

        # Komponenty silnika
        self.map_loader = None
        self.physics_engine = None
        self.visibility_engine = None

        # Stan gry
        self.map_info: Optional[MapInfo] = None
        self.tanks: Dict[str, Tank] = {}
        self.agents: Dict[str, Any] = {}  # Agent controllers
        self.powerups: Dict[str, Any] = {}

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

            # 1. Inicjalizacja game core
            init_result = self.game_core.initialize_game(map_seed)
            if not init_result["success"]:
                self.logger.error(
                    f"Game core initialization failed: {init_result.get('error')}"
                )
                return False

            # 2. Inicjalizacja komponentów silnika
            self._initialize_engines()

            # 3. Wybór i ładowanie mapy
            if not self._load_map(map_seed):
                self.logger.error("Map loading failed")
                return False

            # 4. Spawn czołgów
            if not self._spawn_tanks():
                self.logger.error("Tank spawning failed")
                return False

            # 5. Ładowanie agentów
            if agent_modules and not self._load_agents(agent_modules):
                self.logger.error("Agent loading failed")
                return False

            # 6. Finalizacja inicjalizacji
            self.logger.info("Game initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Game initialization failed with exception: {e}")
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

                # Ograniczenie FPS jeśli potrzebne (opcjonalne)
                if not self.headless:
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

            self.logger.info("Game cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

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

        # TODO: Inicjalizacja rzeczywistych komponentów
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

            # TODO: Implementacja ładowania mapy
            # self.map_info = self.map_loader.load_map(map_seed)

            # Tymczasowe - stwórz pustą mapę
            self.map_info = None  # Placeholder

            self.logger.log_game_event(
                GameEventType.MAP_LOAD,
                f"Map loaded successfully with seed: {map_seed}",
                map_seed=map_seed,
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to load map: {e}")
            return False

    def _spawn_tanks(self) -> bool:
        """
        Spawn czołgów zgodnie z zasadami.

        Returns:
            True jeśli spawn się powiódł
        """
        try:
            self.logger.info("Spawning tanks...")

            spawn_positions = self.game_core.get_tank_spawn_positions()
            tank_types = self.game_core.get_available_tank_types()

            tank_id_counter = 1

            for team in range(1, self.game_core.config.tank_config.team_count + 1):
                for tank_in_team in range(self.game_core.config.tank_config.team_size):
                    # Losowy wybór typu czołgu
                    tank_type = self._select_random_tank_type(tank_types)

                    # Tworzenie ID czołgu
                    tank_id = f"tank_{team}_{tank_in_team + 1}"

                    # Pozycja spawnu
                    position_index = (
                        team - 1
                    ) * self.game_core.config.tank_config.team_size + tank_in_team
                    if position_index < len(spawn_positions):
                        spawn_pos = spawn_positions[position_index]
                    else:
                        # Fallback pozycja
                        spawn_pos = (
                            50 + tank_id_counter * 20,
                            50 + tank_id_counter * 10,
                        )

                    # TODO: Tworzenie rzeczywistego obiektu czołgu
                    # tank = self._create_tank(tank_id, team, tank_type, spawn_pos)
                    # self.tanks[tank_id] = tank

                    self.logger.log_tank_action(
                        tank_id,
                        "spawn",
                        {
                            "team": team,
                            "tank_type": tank_type.value,
                            "position": spawn_pos,
                        },
                    )

                    tank_id_counter += 1

            self.logger.info(f"Successfully spawned {len(spawn_positions)} tanks")
            return True

        except Exception as e:
            self.logger.error(f"Failed to spawn tanks: {e}")
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

                # TODO: Implementacja ładowania agentów
                # agent_controller = self._load_agent_module(agent_module)
                # self.agents[agent_id] = agent_controller

                self.logger.debug(f"Loaded agent: {agent_id}")

            self.logger.info("All agents loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load agents: {e}")
            return False

    def _apply_sudden_death_damage(self):
        """Aplikuje obrażenia nagłej śmierci wszystkim czołgom."""
        damage = self.game_core.get_sudden_death_damage()

        for tank_id, tank in self.tanks.items():
            # TODO: Aplikacja obrażeń
            # tank.apply_damage(damage)
            pass

        self.logger.debug(f"Applied sudden death damage: {damage} to all tanks")

    def _spawn_powerups(self):
        """Spawn power-upów zgodnie z zasadami."""
        powerup_config = self.game_core.get_powerup_config()

        # Sprawdzenie czy nie przekroczono limitu power-upów
        if len(self.powerups) >= powerup_config["max_powerups"]:
            return

        # TODO: Implementacja spawnu power-upów
        # powerup = self._create_random_powerup()
        # self.powerups[powerup.id] = powerup

        self.logger.log_powerup_action(
            "powerup_new", "spawn", {"type": "random", "count": len(self.powerups)}
        )

    def _prepare_sensor_data(self) -> Dict[str, SensorData]:
        """
        Przygotowanie danych sensorycznych dla każdego czołgu.

        Returns:
            Mapa sensor_data dla każdego czołgu
        """
        sensor_data_map = {}

        for tank_id, tank in self.tanks.items():
            # TODO: Implementacja przygotowania sensor_data
            # sensor_data = self.visibility_engine.prepare_sensor_data(tank, self.map_info, self.tanks, self.powerups)
            # sensor_data_map[tank_id] = sensor_data
            pass

        return sensor_data_map

    def _query_agents(self, sensor_data_map: Dict[str, SensorData]) -> Dict[str, Any]:
        """
        Wysłanie zapytań do agentów i odebranie odpowiedzi.

        Args:
            sensor_data_map: Dane sensoryczne dla każdego czołgu

        Returns:
            Mapa akcji od agentów
        """
        agent_actions = {}
        current_tick = self.game_core.get_current_tick()

        for tank_id, sensor_data in sensor_data_map.items():
            agent_id = self._get_agent_for_tank(tank_id)
            if agent_id not in self.agents:
                continue

            try:
                # Pomiar czasu odpowiedzi
                request_start = time.time()

                self.logger.log_agent_interaction(agent_id, "request", tank_id=tank_id)

                # TODO: Implementacja zapytania do agenta
                # tank_status = self._get_tank_status(tank_id)
                # enemies_remaining = self._count_enemies(tank_id)
                # action = self.agents[agent_id].get_action(current_tick, tank_status, sensor_data, enemies_remaining)

                response_time = time.time() - request_start

                # agent_actions[tank_id] = action
                self.logger.log_agent_interaction(
                    agent_id, "response", response_time=response_time, tank_id=tank_id
                )

            except Exception as e:
                self.logger.log_agent_interaction(
                    agent_id, "timeout", error=str(e), tank_id=tank_id
                )

        return agent_actions

    def _process_tank_rotations(self, agent_actions: Dict[str, Any]):
        """Przetworzenie obrotów wieżyczek i kadłubów."""
        for tank_id, action in agent_actions.items():
            # TODO: Implementacja obrotów
            pass

    def _process_ammo_changes(self, agent_actions: Dict[str, Any]):
        """Przetworzenie zmiany amunicji i przeładowania."""
        for tank_id, action in agent_actions.items():
            # TODO: Implementacja zmiany amunicji
            pass

    def _process_shooting(self, agent_actions: Dict[str, Any]):
        """Przetworzenie strzałów."""
        for tank_id, action in agent_actions.items():
            # TODO: Implementacja strzałów
            pass

    def _process_projectile_hits(self):
        """Detekcja trafień pocisków."""
        # TODO: Implementacja detekcji trafień
        pass

    def _process_tank_movement(self, agent_actions: Dict[str, Any]):
        """Przetworzenie ruchu czołgów i kolizji."""
        for tank_id, action in agent_actions.items():
            # TODO: Implementacja ruchu i kolizji
            pass

    def _check_death_conditions(self):
        """Sprawdzenie warunków śmierci czołgów."""
        tanks_to_remove = []

        for tank_id, tank in self.tanks.items():
            # TODO: Sprawdzenie HP <= 0
            # if tank.hp <= 0:
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
            # TODO: Pobranie drużyny z czołgu
            # team = tank.team
            # team_counts[team] = team_counts.get(team, 0) + 1
            pass

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
        # TODO: Implementacja liczenia wrogów
        return len(self.tanks) - 1

    def _cleanup_agents(self):
        """Zakończenie pracy agentów."""
        for agent_id, agent in self.agents.items():
            try:
                # TODO: Wywołanie agent.end()
                # agent.end()
                pass
            except Exception as e:
                self.logger.error(f"Error ending agent {agent_id}: {e}")

    def _cleanup_resources(self):
        """Czyszczenie zasobów gry."""
        self.tanks.clear()
        self.agents.clear()
        self.powerups.clear()

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
        report = (
            self.game_core.game_core.get_performance_report()
            if hasattr(self.game_core, "get_performance_report")
            else {}
        )
        self.logger.info(f"Performance report: {report}")

    def _limit_fps(self, tick_duration: float, target_fps: int = 60):
        """Ograniczenie FPS jeśli potrzebne."""
        target_tick_time = 1.0 / target_fps
        if tick_duration < target_tick_time:
            time.sleep(target_tick_time - tick_duration)


def run_game(
    config=None,
    map_seed: str = None,
    agent_modules: List = None,
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
        game_loop.logger.error(f"Game execution failed: {e}")
        game_loop.cleanup_game()
        return {"success": False, "error": str(e)}
