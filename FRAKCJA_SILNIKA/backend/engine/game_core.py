"""
Game Core Module - Core game management and configuration
Zasady gry (Kto z kim, ile, co i jak) ala config
Refactored to use existing structures
"""

import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..utils.config import (
    GameConfig,
    TankType,
    game_config,
)
from ..utils.logger import get_logger


@dataclass
class GameState:
    """Aktualny stan gry."""

    current_tick: int = 0
    game_active: bool = False
    game_started: bool = False
    sudden_death_active: bool = False
    winner_team: Optional[int] = None

    # Statystyki gry
    teams_alive: List[int] = field(default_factory=lambda: [1, 2])
    tanks_alive_per_team: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0})

    # Spawning tracking
    powerup_spawn_timer: int = 0
    last_powerup_spawn_tick: int = 0


class GameCore:
    """Główna klasa zarządzająca logiką gry i konfiguracją."""

    def __init__(self, config: Optional[GameConfig] = None):
        """
        Inicjalizacja rdzenia gry.

        Args:
            config: Konfiguracja gry (opcjonalna, użyje domyślnej jeśli nie podana)
        """
        self.config = config or game_config
        self.logger = get_logger()
        self.game_state = GameState()

        # Walidacja konfiguracji
        self.config.validate_config()

        # Identyfikator sesji gry
        self.session_id = str(uuid.uuid4())[:8]

        self.logger.info(f"GameCore initialized with session ID: {self.session_id}")

    def initialize_game(self, map_seed: Optional[str] = None) -> Dict:
        """
        Inicjalizacja nowej gry.

        Args:
            map_seed: Seed dla generowania mapy (opcjonalny)

        Returns:
            Słownik z informacjami o inicjalizacji
        """
        if self.game_state.game_started:
            self.logger.warning("Próba inicjalizacji już rozpoczętej gry")
            return {"success": False, "error": "Game already started"}

        # Ustawienie seed'a jeśli podany
        if map_seed:
            random.seed(map_seed)

        # Reset stanu gry
        self._reset_game_state()

        # Przygotowanie informacji o inicjalizacji
        init_info = {
            "session_id": self.session_id,
            "map_size": (self.config.map_config.width, self.config.map_config.height),
            "team_count": self.config.tank_config.team_count,
            "team_size": self.config.tank_config.team_size,
            "total_tanks": self.config.tank_config.team_count
            * self.config.tank_config.team_size,
            "friendly_fire": self.config.game_rules.friendly_fire,
            "sudden_death_tick": self.config.game_rules.sudden_death_tick,
            "map_seed": map_seed,
            "spawn_positions": self.config.get_tank_spawn_positions(),
        }

        self.logger.start_game(**init_info)
        self.game_state.game_started = True

        return {"success": True, "init_info": init_info}

    def start_game_loop(self):
        """Rozpoczęcie głównej pętli gry."""
        if not self.game_state.game_started:
            self.logger.error("Próba rozpoczęcia gry bez inicjalizacji")
            return False

        self.game_state.game_active = True
        self.logger.info("Game loop started")
        return True

    def can_continue_game(self) -> bool:
        """
        Sprawdza czy gra może być kontynuowana.

        Returns:
            True jeśli gra może trwać dalej, False w przeciwnym razie
        """
        if not self.game_state.game_active:
            return False

        # Sprawdzenie warunku wygranej (team annihilation)
        alive_teams = [
            team
            for team, count in self.game_state.tanks_alive_per_team.items()
            if count > 0
        ]

        if len(alive_teams) <= 1:
            if len(alive_teams) == 1:
                self.game_state.winner_team = alive_teams[0]
                self.logger.info(f"Team {self.game_state.winner_team} wins!")
            else:
                self.logger.info("Draw - no teams left alive!")

            return False

        return True

    def process_tick(self) -> Dict:
        """
        Przetwarzanie pojedynczego tick'a gry.

        Returns:
            Informacje o przetworzonym tick'u
        """
        self.game_state.current_tick += 1
        self.logger.set_current_tick(self.game_state.current_tick)

        tick_info = {
            "tick": self.game_state.current_tick,
            "sudden_death": False,
            "powerup_spawned": False,
            "game_continues": True,
        }

        # Sprawdzenie nagłej śmierci
        if self.game_state.current_tick >= self.config.game_rules.sudden_death_tick:
            if not self.game_state.sudden_death_active:
                self.game_state.sudden_death_active = True
                try:
                    # Try to use the existing GameEventType from logger
                    from ..utils.logger import GameEventType

                    self.logger.log_game_event(
                        GameEventType.SUDDEN_DEATH,
                        f"Sudden death activated at tick {self.game_state.current_tick}",
                    )
                except (ImportError, AttributeError):
                    # Fallback if GameEventType is not available
                    self.logger.info(
                        f"Sudden death activated at tick {self.game_state.current_tick}"
                    )
            tick_info["sudden_death"] = True

        # Sprawdzenie spawnu power-upów
        should_spawn_powerup = self._should_spawn_powerup()
        if should_spawn_powerup:
            tick_info["powerup_spawned"] = True
            self.game_state.last_powerup_spawn_tick = self.game_state.current_tick

        # Sprawdzenie czy gra może być kontynuowana
        tick_info["game_continues"] = self.can_continue_game()

        return tick_info

    def end_game(self, reason: str = "normal") -> Dict:
        """
        Zakończenie gry.

        Args:
            reason: Powód zakończenia gry

        Returns:
            Wyniki gry
        """
        self.game_state.game_active = False

        game_results = {
            "session_id": self.session_id,
            "total_ticks": self.game_state.current_tick,
            "winner_team": self.game_state.winner_team,
            "reason": reason,
            "sudden_death_reached": self.game_state.sudden_death_active,
            "final_team_counts": self.game_state.tanks_alive_per_team.copy(),
        }

        self.logger.end_game(**game_results)
        return game_results

    def update_team_count(self, team: int, alive_count: int):
        """
        Aktualizacja liczby żywych czołgów w drużynie.

        Args:
            team: Numer drużyny
            alive_count: Liczba żywych czołgów
        """
        self.game_state.tanks_alive_per_team[team] = alive_count
        self.logger.debug(f"Team {team} has {alive_count} tanks alive")

    def get_current_tick(self) -> int:
        """Pobieranie aktualnego tick'a."""
        return self.game_state.current_tick

    def is_sudden_death_active(self) -> bool:
        """Sprawdzenie czy aktywna jest nagła śmierć."""
        return self.game_state.sudden_death_active

    def get_sudden_death_damage(self) -> int:
        """Pobieranie obrażeń nagłej śmierci."""
        return self.config.game_rules.sudden_death_damage_per_tick

    def get_collision_damage(self, collision_type: str, participant: str) -> int:
        """
        Pobieranie obrażeń od kolizji.

        Args:
            collision_type: Typ kolizji (np. "tank_vs_wall")
            participant: Uczestnik kolizji (np. "tank", "moving", "stationary")

        Returns:
            Wartość obrażeń
        """
        collision_data = self.config.physics_config.collision_damage.get(
            collision_type, {}
        )
        return collision_data.get(participant, 0)

    def get_tank_spawn_positions(self) -> List[Tuple[float, float]]:
        """Pobieranie pozycji spawnu czołgów."""
        return self.config.get_tank_spawn_positions()

    def get_available_tank_types(self) -> List[TankType]:
        """Pobieranie dostępnych typów czołgów."""
        return list(self.config.tank_config.tank_stats.keys())

    def get_tank_stats_for_type(self, tank_type: TankType) -> Dict:
        """Pobieranie statystyk dla danego typu czołgu."""
        return self.config.tank_config.tank_stats.get(tank_type, {})

    def _reset_game_state(self):
        """Reset stanu gry do wartości początkowych."""
        self.game_state = GameState()
        self.game_state.tanks_alive_per_team = {
            i + 1: self.config.tank_config.team_size
            for i in range(self.config.tank_config.team_count)
        }
        self.game_state.teams_alive = list(
            range(1, self.config.tank_config.team_count + 1)
        )

    def _should_spawn_powerup(self) -> bool:
        """
        Sprawdzenie czy należy zespawnować power-up.

        Returns:
            True jeśli należy zespawnować power-up
        """
        current_tick = self.game_state.current_tick

        # Sprawdzenie czy minął czas startowy
        if current_tick < self.config.powerup_config.spawn_start_tick:
            return False

        # Sprawdzenie interwału
        ticks_since_start = current_tick - self.config.powerup_config.spawn_start_tick
        return ticks_since_start % self.config.powerup_config.spawn_interval == 0

    def get_powerup_config(self) -> Dict:
        """Pobieranie konfiguracji power-upów."""
        return {
            "spawn_start_tick": self.config.powerup_config.spawn_start_tick,
            "spawn_interval": self.config.powerup_config.spawn_interval,
            "max_powerups": self.config.powerup_config.max_powerups_on_map,
            "min_distance": self.config.powerup_config.min_distance_between_powerups,
            "despawn_time": self.config.powerup_config.despawn_time,
        }

    def get_map_config(self) -> Dict:
        """Pobieranie konfiguracji mapy."""
        return {
            "width": self.config.map_config.width,
            "height": self.config.map_config.height,
            "layouts": self.config.map_config.available_layouts,
            "obstacle_size": self.config.map_config.obstacle_size,
            "terrain_size": self.config.map_config.terrain_size,
            "powerup_size": self.config.map_config.powerup_size,
        }

    def get_coordinate_system_info(self) -> Dict:
        """Pobieranie informacji o systemie współrzędnych."""
        cs = self.config.coordinate_system
        return {
            "origin": cs.origin,
            "x_direction": cs.x_direction,
            "y_direction": cs.y_direction,
            "angle_zero": cs.angle_zero,
            "angle_direction": cs.angle_direction,
            "position_reference": cs.position_reference,
        }


# Factory functions
def create_game_core(config: Optional[GameConfig] = None) -> GameCore:
    """
    Factory function do tworzenia instancji GameCore.

    Args:
        config: Opcjonalna konfiguracja gry

    Returns:
        Nowa instancja GameCore
    """
    return GameCore(config)


def create_default_game() -> GameCore:
    """
    Tworzenie gry z domyślną konfiguracją.

    Returns:
        GameCore z domyślną konfiguracją
    """
    return GameCore()
