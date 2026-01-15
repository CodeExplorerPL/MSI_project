"""
Game Loop Module - Main game loop implementation
Główna pętla gry, razem z initem
Refactored to use existing structures and improve compatibility

============================================================================
USAGE INSTRUCTIONS / INSTRUKCJA URUCHOMIENIA
============================================================================

To run the game engine:
    python run_game.py --headless --log-level DEBUG

Agent Configuration:
    Each tank connects to an agent server. Agents are FastAPI servers that
    the engine communicates with via HTTP POST requests.

    Agent URL format: http://{host}:{port}
    Default base port: 8001 (increments per tank: 8001, 8002, 8003, ...)

    To start an agent server:
        cd controller
        python server.py --host 0.0.0.0 --port 8001

    For multiple tanks, start multiple agents on different ports:
        python server.py --port 8001  # Tank 1
        python server.py --port 8002  # Tank 2
        ...

Agent API Endpoints (called by engine):
    POST /agent/action   - Get tank action for current tick
    POST /agent/destroy  - Notify agent that tank was destroyed
    POST /agent/end      - Send final scoreboard when game ends

Configurable Team Sizes:
    Set TEAM_A_NBR and TEAM_B_NBR below to control number of tanks per team.
    Each tank type is randomly selected (1=Light, 2=Heavy, 3=Sniper).

============================================================================
"""

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import httpx

from ..structures import MapInfo, Position
from ..tank.base_tank import Tank
from ..tank.heavy_tank import HeavyTank
from ..tank.light_tank import LightTank
from ..tank.sniper_tank import SniperTank
from ..utils.config import GameConfig, TankType, game_config
from ..utils.logger import GameEventType, get_logger

from .game_core import GameCore, create_default_game
from .map_loader import MapLoader
from .physics import process_physics_tick, apply_damage
from .visibility import check_visibility

# Type alias for tank union
TankUnion = Union[LightTank, HeavyTank, SniperTank]

# ============================================================================
# CONFIGURABLE TEAM SIZES / KONFIGUROWALNA LICZBA CZOŁGÓW
# ============================================================================
# Set number of tanks per team. Each tank type is randomly chosen (1-3).
# Tank types: 1 = LightTank, 2 = HeavyTank, 3 = SniperTank

TEAM_A_NBR = 5  # Number of tanks in Team A (Team 1)
TEAM_B_NBR = 5  # Number of tanks in Team B (Team 2)

# Base port for agent servers (tank_1_1 -> 8001, tank_1_2 -> 8002, etc.)
AGENT_BASE_PORT = 8001
AGENT_HOST = "127.0.0.1"
AGENT_TIMEOUT = 1.0  # Seconds to wait for agent response


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AgentConnection:
    """Configuration for agent HTTP connection."""
    tank_id: str
    host: str
    port: int
    base_url: str = field(init=False)
    
    def __post_init__(self):
        self.base_url = f"http://{self.host}:{self.port}"


@dataclass
class TankScoreboard:
    """Scoreboard tracking damage and kills for a tank."""
    tank_id: str
    team: int
    damage_dealt: float = 0.0
    tanks_killed: int = 0


# ============================================================================
# GAME LOOP CLASS
# ============================================================================

class GameLoop:
    """Główna klasa pętli gry z fazami inicjalizacji, loop i końca."""

    def __init__(self, config: Optional[GameConfig] = None, headless: bool = False):
        """
        Inicjalizacja GameLoop.

        Args:
            config: Konfiguracja gry (opcjonalna)
            headless: Czy uruchomić w trybie bez interfejsu graficznego
        """
        self.game_core = GameCore(config) if config else create_default_game()
        self.logger = get_logger()
        self.headless = headless

        # Engine components
        self.map_loader = MapLoader()
        
        # Game state
        self.map_info: Optional[MapInfo] = None
        self.tanks: Dict[str, TankUnion] = {}
        self.agent_connections: Dict[str, AgentConnection] = {}
        
        # Scoreboard tracking
        self.scoreboards: Dict[str, TankScoreboard] = {}
        self.last_attacker: Dict[str, str] = {}  # Maps tank_id -> last attacker tank_id
        
        # HTTP client for agent communication
        self.http_client: Optional[httpx.Client] = None

        # Performance metrics
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
            agent_modules: Lista modułów agentów (host:port or just port)

        Returns:
            True jeśli inicjalizacja się powiodła
        """
        try:
            self.logger.info("Starting game initialization...")

            # 1. Initialize game core
            init_result = self.game_core.initialize_game(map_seed)
            if not init_result["success"]:
                self.logger.error(
                    f"Game core initialization failed: {init_result.get('error')}"
                )
                return False

            # 2. Initialize HTTP client
            self.http_client = httpx.Client(timeout=AGENT_TIMEOUT)

            # 3. Load map
            if not self._load_map(map_seed):
                self.logger.error("Map loading failed")
                return False

            # 4. Spawn tanks
            if not self._spawn_tanks():
                self.logger.error("Tank spawning failed")
                return False

            # 5. Load agents (connect to agent servers)
            if not self._load_agents(agent_modules):
                self.logger.warning("Agent loading failed - continuing without agents")
                # Continue anyway - game can run without agents for testing

            # 6. Finalize initialization
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

        # Start game loop
        if not self.game_core.start_game_loop():
            return {"success": False, "error": "Failed to start game loop"}

        game_results = {"success": True}

        try:
            # Main loop: While(one team alive)
            while self.game_core.can_continue_game():
                tick_start_time = time.time()

                # Process tick
                tick_info = self._process_game_tick()

                # Performance measurement
                tick_duration = time.time() - tick_start_time
                self.logger.log_tick_end(
                    self.game_core.get_current_tick(), tick_duration
                )
                self._update_performance_metrics(tick_duration)

                # Check end conditions
                if not tick_info["game_continues"]:
                    break

                # FPS limiting if not headless
                if not self.headless:
                    self._limit_fps(tick_duration)

            # End game
            game_results = self.game_core.end_game("normal")
            game_results["scoreboards"] = self._get_final_scoreboards()
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
            # Send end signal to all agents with scoreboard
            self._cleanup_agents()

            # Close HTTP client
            if self.http_client:
                self.http_client.close()
                self.http_client = None

            # Clean resources
            self._cleanup_resources()

            # Generate performance report
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
        7. Przetworzenie logiki (physics)
        """
        # 1. Process tick in game core
        tick_info = self.game_core.process_tick()
        current_tick = tick_info["tick"]

        self.logger.log_tick_start(current_tick)

        # 2. Sudden death - damage to all tanks
        if tick_info["sudden_death"]:
            self._apply_sudden_death_damage()

        # 3. Spawn power-ups (handled by physics engine via map_info)
        if tick_info["powerup_spawned"]:
            self._spawn_powerups()

        # 4. Prepare sensor_data for each tank
        sensor_data_map = self._prepare_sensor_data()

        # 5. Query agents and get actions
        agent_actions = self._query_agents(sensor_data_map, current_tick)

        # 6. Process physics using physics.py
        self._process_physics(agent_actions)

        # 7. Check death conditions and notify agents
        self._check_death_conditions()

        # 8. Update team counts
        self._update_team_counts()

        return tick_info

    def _load_map(self, map_seed: Optional[str] = None) -> bool:
        """
        Ładowanie i tworzenie mapy.

        Args:
            map_seed: Seed dla generacji mapy (używany jako nazwa pliku)

        Returns:
            True jeśli ładowanie się powiodło
        """
        try:
            self.logger.info(f"Loading map with seed: {map_seed}")

            # Try to load map from file
            available_maps = self.map_loader.get_available_maps()
            
            if available_maps:
                # Use first available map or specified seed
                map_file = map_seed if map_seed and map_seed in available_maps else available_maps[0]
                try:
                    self.map_info = self.map_loader.load_map(map_file)
                    self.logger.info(f"Loaded map from file: {map_file}")
                except FileNotFoundError:
                    self.logger.warning(f"Map file not found: {map_file}, creating empty map")
                    self._create_empty_map()
            else:
                self.logger.warning("No map files found, creating empty map")
                self._create_empty_map()

            self.logger.log_game_event(
                GameEventType.MAP_LOAD,
                f"Map loaded successfully with seed: {map_seed}",
                map_seed=map_seed,
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to load map: {e}")
            return False

    def _create_empty_map(self):
        """Create an empty map with default size."""
        self.map_info = MapInfo(
            _map_seed="empty",
            _obstacle_list=[],
            _terrain_list=[],
            _powerup_list=[],
            _all_tanks=[],
            _size=[
                self.game_core.config.map_config.width,
                self.game_core.config.map_config.height
            ]
        )

    def _spawn_tanks(self) -> bool:
        """
        Spawn czołgów zgodnie z konfiguracją TEAM_A_NBR i TEAM_B_NBR.
        Typ czołgu jest losowany (1-3 = Light/Heavy/Sniper).

        Returns:
            True jeśli spawn się powiódł
        """
        try:
            self.logger.info(f"Spawning tanks: Team A={TEAM_A_NBR}, Team B={TEAM_B_NBR}")

            tank_port = AGENT_BASE_PORT

            # Team 1 tanks
            for i in range(TEAM_A_NBR):
                tank_id = f"tank_1_{i + 1}"
                tank_type = random.randint(1, 3)  # 1=Light, 2=Heavy, 3=Sniper
                spawn_pos = self._get_spawn_position(1, i)
                
                tank = self._create_tank(tank_id, 1, tank_type, spawn_pos)
                self.tanks[tank_id] = tank
                
                # Create agent connection
                self.agent_connections[tank_id] = AgentConnection(
                    tank_id=tank_id,
                    host=AGENT_HOST,
                    port=tank_port
                )
                
                # Initialize scoreboard
                self.scoreboards[tank_id] = TankScoreboard(
                    tank_id=tank_id,
                    team=1
                )
                
                self.logger.log_tank_action(
                    tank_id,
                    "spawn",
                    {
                        "team": 1,
                        "tank_type": tank.tank_type,
                        "position": (spawn_pos.x, spawn_pos.y),
                        "agent_port": tank_port,
                    },
                )
                tank_port += 1

            # Team 2 tanks
            for i in range(TEAM_B_NBR):
                tank_id = f"tank_2_{i + 1}"
                tank_type = random.randint(1, 3)  # 1=Light, 2=Heavy, 3=Sniper
                spawn_pos = self._get_spawn_position(2, i)
                
                tank = self._create_tank(tank_id, 2, tank_type, spawn_pos)
                self.tanks[tank_id] = tank
                
                # Create agent connection
                self.agent_connections[tank_id] = AgentConnection(
                    tank_id=tank_id,
                    host=AGENT_HOST,
                    port=tank_port
                )
                
                # Initialize scoreboard
                self.scoreboards[tank_id] = TankScoreboard(
                    tank_id=tank_id,
                    team=2
                )
                
                self.logger.log_tank_action(
                    tank_id,
                    "spawn",
                    {
                        "team": 2,
                        "tank_type": tank.tank_type,
                        "position": (spawn_pos.x, spawn_pos.y),
                        "agent_port": tank_port,
                    },
                )
                tank_port += 1

            # Update map_info with tanks
            if self.map_info:
                self.map_info._all_tanks = list(self.tanks.values())

            self.logger.info(f"Successfully spawned {len(self.tanks)} tanks")
            return True

        except Exception as e:
            self.logger.error(f"Failed to spawn tanks: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _create_tank(
        self, tank_id: str, team: int, tank_type: int, position: Position
    ) -> TankUnion:
        """
        Create a tank of the specified type.

        Args:
            tank_id: Unique tank identifier
            team: Team number (1 or 2)
            tank_type: Tank type (1=Light, 2=Heavy, 3=Sniper)
            position: Spawn position

        Returns:
            Created tank object
        """
        if tank_type == 1:
            return LightTank(_id=tank_id, team=team, start_pos=position)
        elif tank_type == 2:
            return HeavyTank(_id=tank_id, team=team, start_pos=position)
        else:  # tank_type == 3
            return SniperTank(_id=tank_id, team=team, start_pos=position)

    def _get_spawn_position(self, team: int, index: int) -> Position:
        """
        Get spawn position for a tank.

        Args:
            team: Team number (1 or 2)
            index: Tank index within team

        Returns:
            Spawn position
        """
        map_width = self.game_core.config.map_config.width
        
        if team == 1:
            # Team 1 spawns on left side
            x = 50 + (index * 15)
            y = 50 + (index * 20)
        else:
            # Team 2 spawns on right side
            x = map_width - 100 + (index * 15)
            y = 50 + (index * 20)
        
        return Position(x, y)

    def _load_agents(self, agent_modules: Optional[List]) -> bool:
        """
        Ładowanie modułów agentów (connecting to agent HTTP servers).

        Args:
            agent_modules: Lista modułów agentów (can be port numbers or host:port)

        Returns:
            True jeśli co najmniej jeden agent został załadowany
        """
        try:
            self.logger.info(f"Connecting to agent servers...")
            
            connected_count = 0
            
            for tank_id, connection in self.agent_connections.items():
                # Try to ping the agent server
                try:
                    response = self.http_client.get(
                        f"{connection.base_url}/",
                        timeout=0.5
                    )
                    if response.status_code == 200:
                        self.logger.info(
                            f"Connected to agent for {tank_id} at {connection.base_url}"
                        )
                        connected_count += 1
                    else:
                        self.logger.warning(
                            f"Agent for {tank_id} returned status {response.status_code}"
                        )
                except httpx.ConnectError:
                    self.logger.warning(
                        f"No agent running for {tank_id} at {connection.base_url}"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to connect to agent for {tank_id}: {e}"
                    )

            self.logger.info(f"Connected to {connected_count}/{len(self.agent_connections)} agents")
            return connected_count > 0

        except Exception as e:
            self.logger.error(f"Failed to load agents: {e}")
            return False

    def _apply_sudden_death_damage(self):
        """Aplikuje obrażenia nagłej śmierci wszystkim czołgom."""
        damage = abs(self.game_core.get_sudden_death_damage())

        for tank_id, tank in self.tanks.items():
            if tank.is_alive():
                apply_damage(tank, damage)

        self.logger.debug(f"Applied sudden death damage: {damage} to all tanks")

    def _spawn_powerups(self):
        """Spawn power-upów zgodnie z zasadami."""
        powerup_config = self.game_core.get_powerup_config()

        # Check if we hit max powerups
        if self.map_info and len(self.map_info.powerup_list) >= powerup_config["max_powerups"]:
            return

        # TODO: Implement actual powerup spawning
        # For now, powerups are not spawned dynamically

        self.logger.log_powerup_action(
            "powerup_new", "spawn", {"type": "random", "count": len(self.map_info.powerup_list) if self.map_info else 0}
        )

    def _prepare_sensor_data(self) -> Dict[str, Any]:
        """
        Przygotowanie danych sensorycznych dla każdego czołgu.
        Uses visibility.py check_visibility function.

        Returns:
            Mapa sensor_data dla każdego czołgu
        """
        sensor_data_map = {}

        all_tanks_list = list(self.tanks.values())
        obstacles = self.map_info.obstacle_list if self.map_info else []
        terrains = self.map_info.terrain_list if self.map_info else []
        powerups = self.map_info.powerup_list if self.map_info else []

        for tank_id, tank in self.tanks.items():
            if not tank.is_alive():
                continue

            # Use visibility system to get sensor data
            sensor_data = check_visibility(
                tank=tank,
                all_tanks=all_tanks_list,
                obstacles=obstacles,
                terrains=terrains,
                powerups=powerups
            )
            sensor_data_map[tank_id] = sensor_data

        return sensor_data_map

    def _query_agents(
        self, sensor_data_map: Dict[str, Any], current_tick: int
    ) -> Dict[str, Any]:
        """
        Wysłanie zapytań do agentów i odebranie odpowiedzi.
        Calls /agent/action endpoint for each tank.

        Args:
            sensor_data_map: Dane sensoryczne dla każdego czołgu
            current_tick: Current game tick

        Returns:
            Mapa akcji od agentów (tank_id -> ActionCommand dict)
        """
        agent_actions = {}

        for tank_id, sensor_data in sensor_data_map.items():
            if tank_id not in self.agent_connections:
                continue

            tank = self.tanks.get(tank_id)
            if not tank or not tank.is_alive():
                continue

            connection = self.agent_connections[tank_id]

            try:
                # Measure response time
                request_start = time.time()

                self.logger.log_agent_interaction(connection.base_url, "request", tank_id=tank_id)

                # Build payload for agent
                payload = {
                    "current_tick": current_tick,
                    "my_tank_status": self._tank_to_dict(tank),
                    "sensor_data": self._sensor_data_to_dict(sensor_data),
                    "enemies_remaining": self._count_enemies(tank_id)
                }

                # POST to /agent/action
                response = self.http_client.post(
                    f"{connection.base_url}/agent/action",
                    json=payload
                )

                response_time = time.time() - request_start

                if response.status_code == 200:
                    action_data = response.json()
                    agent_actions[tank_id] = action_data
                    self.logger.log_agent_interaction(
                        connection.base_url, "response",
                        response_time=response_time, tank_id=tank_id
                    )
                else:
                    self.logger.warning(
                        f"Agent {tank_id} returned error: {response.status_code}"
                    )

            except httpx.ConnectError:
                # Agent not running, skip
                pass
            except httpx.TimeoutException:
                self.logger.log_agent_interaction(
                    connection.base_url, "timeout", tank_id=tank_id
                )
            except Exception as e:
                self.logger.log_agent_interaction(
                    connection.base_url, "error", error=str(e), tank_id=tank_id
                )

        return agent_actions

    def _tank_to_dict(self, tank: TankUnion) -> Dict[str, Any]:
        """Convert tank object to dictionary for API."""
        return {
            "_id": tank._id,
            "_team": tank._team,
            "_tank_type": tank._tank_type,
            "hp": tank.hp,
            "shield": tank.shield,
            "_max_hp": tank._max_hp,
            "_max_shield": tank._max_shield,
            "position": {"x": tank.position.x, "y": tank.position.y},
            "heading": tank.heading,
            "barrel_angle": tank.barrel_angle,
            "move_speed": tank.move_speed,
            "_top_speed": tank._top_speed,
            "_vision_range": tank._vision_range,
            "_vision_angle": tank._vision_angle,
            "ammo_loaded": tank.ammo_loaded.name if tank.ammo_loaded else None,
            "is_overcharged": tank.is_overcharged,
            "_reload_timer": tank._reload_timer,
            "ammo": {
                k.name: {"count": v.count, "_ammo_type": k.name}
                for k, v in tank.ammo.items()
            }
        }

    def _sensor_data_to_dict(self, sensor_data) -> Dict[str, Any]:
        """Convert sensor data to dictionary for API."""
        return {
            "seen_tanks": [
                {
                    "id": st.id,
                    "team": st.team,
                    "tank_type": st.tank_type,
                    "position": {"x": st.position.x, "y": st.position.y},
                    "is_damaged": st.is_damaged,
                    "heading": st.heading,
                    "barrel_angle": st.barrel_angle,
                    "distance": st.distance,
                }
                for st in sensor_data.seen_tanks
            ],
            "seen_powerups": [
                {
                    "id": getattr(p, "id", ""),
                    "position": {"x": p.position.x, "y": p.position.y},
                    "powerup_type": str(p.powerup_type),
                }
                for p in sensor_data.seen_powerups
            ],
            "seen_obstacles": [
                {
                    "id": getattr(o, "_id", ""),
                    "position": {"x": o._position.x, "y": o._position.y},
                    "is_destructible": o.is_destructible,
                }
                for o in sensor_data.seen_obstacles
            ],
            "seen_terrains": [
                {
                    "position": {"x": t._position.x, "y": t._position.y},
                }
                for t in sensor_data.seen_terrains
            ],
        }

    def _process_physics(self, agent_actions: Dict[str, Any]):
        """
        Process physics using physics.py process_physics_tick.
        
        Args:
            agent_actions: Dictionary of tank_id -> action dict
        """
        if not self.map_info:
            return

        # Convert action dicts to ActionCommand-like objects
        from controller.api import ActionCommand
        
        actions_converted = {}
        for tank_id, action_dict in agent_actions.items():
            try:
                actions_converted[tank_id] = ActionCommand(
                    barrel_rotation_angle=action_dict.get("barrel_rotation_angle", 0.0),
                    heading_rotation_angle=action_dict.get("heading_rotation_angle", 0.0),
                    move_speed=action_dict.get("move_speed", 0.0),
                    ammo_to_load=None,  # TODO: Parse ammo type
                    should_fire=action_dict.get("should_fire", False)
                )
            except Exception as e:
                self.logger.warning(f"Failed to parse action for {tank_id}: {e}")

        # Process physics tick
        all_tanks_list = list(self.tanks.values())
        delta_time = 1.0 / 60.0  # Assuming 60 FPS

        try:
            results = process_physics_tick(
                all_tanks=all_tanks_list,
                actions=actions_converted,
                map_info=self.map_info,
                delta_time=delta_time
            )

            # Update scoreboards based on projectile hits
            for hit in results.get("projectile_hits", []):
                if hit.hit_tank_id:
                    # Find who fired (the tank whose action caused this)
                    for tank_id, action in actions_converted.items():
                        if action.should_fire:
                            # Credit damage to attacker
                            if tank_id in self.scoreboards:
                                self.scoreboards[tank_id].damage_dealt += hit.damage_dealt
                            # Track last attacker for kill credit
                            self.last_attacker[hit.hit_tank_id] = tank_id
                            break

            # Log destroyed tanks
            for tank_id in results.get("destroyed_tanks", []):
                self.logger.log_tank_action(tank_id, "destroyed", {})

        except Exception as e:
            self.logger.error(f"Physics processing failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _check_death_conditions(self):
        """Sprawdzenie warunków śmierci czołgów i powiadamianie agentów."""
        tanks_to_remove = []

        for tank_id, tank in self.tanks.items():
            if not tank.is_alive():
                tanks_to_remove.append(tank_id)
                
                # Credit kill to last attacker
                attacker_id = self.last_attacker.get(tank_id)
                if attacker_id and attacker_id in self.scoreboards:
                    self.scoreboards[attacker_id].tanks_killed += 1
                    self.logger.info(
                        f"Tank {tank_id} destroyed by {attacker_id}"
                    )

                # Notify agent of destruction
                self._notify_agent_destroyed(tank_id)

                self.logger.log_tank_action(
                    tank_id, "death", {"final_hp": tank.hp}
                )

        # Remove dead tanks
        for tank_id in tanks_to_remove:
            del self.tanks[tank_id]
            # Also remove from map_info
            if self.map_info:
                self.map_info.all_tanks = [t for t in self.map_info.all_tanks if t._id != tank_id]

    def _notify_agent_destroyed(self, tank_id: str):
        """
        Notify agent that its tank was destroyed.
        Calls POST /agent/destroy endpoint.
        """
        if tank_id not in self.agent_connections:
            return

        connection = self.agent_connections[tank_id]

        try:
            self.http_client.post(
                f"{connection.base_url}/agent/destroy",
                timeout=0.5
            )
            self.logger.debug(f"Notified agent {tank_id} of destruction")
        except Exception as e:
            self.logger.debug(f"Failed to notify agent {tank_id} of destruction: {e}")

    def _update_team_counts(self):
        """Aktualizacja liczby żywych czołgów w zespołach."""
        team_counts = {}

        for tank_id, tank in self.tanks.items():
            if tank.is_alive():
                team = tank.team
                team_counts[team] = team_counts.get(team, 0) + 1

        # Update in game core
        for team, count in team_counts.items():
            self.game_core.update_team_count(team, count)

    def _count_enemies(self, tank_id: str) -> int:
        """Liczenie wrogich czołgów dla danego czołgu."""
        if tank_id not in self.tanks:
            return 0
        
        my_team = self.tanks[tank_id].team
        return sum(
            1 for t in self.tanks.values()
            if t.is_alive() and t.team != my_team
        )

    def _get_final_scoreboards(self) -> List[Dict[str, Any]]:
        """Get final scoreboards for all tanks."""
        return [
            {
                "tank_id": sb.tank_id,
                "team": sb.team,
                "damage_dealt": sb.damage_dealt,
                "tanks_killed": sb.tanks_killed,
            }
            for sb in self.scoreboards.values()
        ]

    def _cleanup_agents(self):
        """
        Zakończenie pracy agentów.
        Sends scoreboard to each agent via POST /agent/end.
        """
        for tank_id, connection in self.agent_connections.items():
            scoreboard = self.scoreboards.get(tank_id)
            if not scoreboard:
                continue

            try:
                payload = {
                    "damage_dealt": scoreboard.damage_dealt,
                    "tanks_killed": scoreboard.tanks_killed,
                }
                
                self.http_client.post(
                    f"{connection.base_url}/agent/end",
                    json=payload,
                    timeout=0.5
                )
                self.logger.debug(f"Sent end signal to agent {tank_id}")
            except Exception as e:
                self.logger.debug(f"Failed to send end signal to agent {tank_id}: {e}")

    def _cleanup_resources(self):
        """Czyszczenie zasobów gry."""
        self.tanks.clear()
        self.agent_connections.clear()
        self.scoreboards.clear()
        self.last_attacker.clear()

    def _update_performance_metrics(self, tick_duration: float):
        """Aktualizacja metryk wydajności."""
        self.performance_data["total_ticks"] += 1

        # Calculate moving average
        if self.performance_data["avg_tick_time"] == 0:
            self.performance_data["avg_tick_time"] = tick_duration
        else:
            alpha = 0.1
            self.performance_data["avg_tick_time"] = (
                alpha * tick_duration
                + (1 - alpha) * self.performance_data["avg_tick_time"]
            )

    def _generate_performance_report(self):
        """Generowanie raportu wydajności."""
        try:
            report = self.logger.get_performance_report()
            self.logger.info(f"Performance report: {report}")
        except (AttributeError, Exception):
            self.logger.info(f"Performance data: {self.performance_data}")

    def _limit_fps(self, tick_duration: float, target_fps: int = 60):
        """Ograniczenie FPS jeśli potrzebne."""
        target_tick_time = 1.0 / target_fps
        if tick_duration < target_tick_time:
            time.sleep(target_tick_time - tick_duration)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def run_game(
    config: Optional[GameConfig] = None,
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
        Wyniki gry z scoreboardem
    """
    game_loop = GameLoop(config, headless)

    try:
        # Phase 1: Initialization
        if not game_loop.initialize_game(map_seed, agent_modules):
            return {"success": False, "error": "Initialization failed"}

        # Phase 2: Main loop
        results = game_loop.run_game_loop()

        # Phase 3: Cleanup
        game_loop.cleanup_game()

        return results

    except Exception as e:
        game_loop.logger.error(f"Game execution failed: {e}")
        game_loop.cleanup_game()
        return {"success": False, "error": str(e)}
