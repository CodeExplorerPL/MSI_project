"""Comprehensive Logging System for Tank Battle Game Engine"""

import logging
import os
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class LogLevel(Enum):
    """Log levels for different types of messages."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    GAME_EVENT = "GAME_EVENT"
    PERFORMANCE = "PERFORMANCE"


class GameEventType(Enum):
    """Specific game event types for tracking."""

    GAME_START = "GAME_START"
    GAME_END = "GAME_END"
    TICK_START = "TICK_START"
    TICK_END = "TICK_END"
    TANK_SPAWN = "TANK_SPAWN"
    TANK_DEATH = "TANK_DEATH"
    TANK_MOVE = "TANK_MOVE"
    TANK_SHOOT = "TANK_SHOOT"
    TANK_HIT = "TANK_HIT"
    TANK_COLLISION = "TANK_COLLISION"
    POWERUP_SPAWN = "POWERUP_SPAWN"
    POWERUP_COLLECTED = "POWERUP_COLLECTED"
    POWERUP_DESPAWN = "POWERUP_DESPAWN"
    MAP_LOAD = "MAP_LOAD"
    AGENT_REQUEST = "AGENT_REQUEST"
    AGENT_RESPONSE = "AGENT_RESPONSE"
    AGENT_TIMEOUT = "AGENT_TIMEOUT"
    SUDDEN_DEATH = "SUDDEN_DEATH"


class GameLogger:
    """Main logging class for the game engine."""

    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        """
        Initialize the game logger.

        Args:
            log_dir: Directory to store log files
            log_level: Minimum log level to record
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.log_level = getattr(logging, log_level.upper())
        self.current_tick = 0
        self.game_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Performance tracking
        self.performance_metrics = {
            "tick_times": [],
            "agent_response_times": {},
            "total_ticks": 0,
            "game_start_time": None,
            "game_end_time": None,
        }

        # Game statistics
        self.game_stats = {
            "tanks_spawned": 0,
            "tanks_killed": 0,
            "shots_fired": 0,
            "hits_landed": 0,
            "powerups_spawned": 0,
            "powerups_collected": 0,
            "collisions": 0,
        }

        self._setup_loggers()

    def _setup_loggers(self):
        """Setup different loggers for different purposes."""
        # Main game logger
        self.main_logger = self._create_logger(
            "game_main",
            f"game_{self.game_session_id}.log",
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | TICK:%(tick)04d | %(message)s",
                datefmt="%H:%M:%S",
            ),
        )

        # Event logger for game events
        self.event_logger = self._create_logger(
            "game_events",
            f"events_{self.game_session_id}.log",
            logging.Formatter(
                "%(asctime)s | %(event_type)-15s | TICK:%(tick)04d | %(message)s",
                datefmt="%H:%M:%S",
            ),
        )

        # Performance logger
        self.performance_logger = self._create_logger(
            "game_performance",
            f"performance_{self.game_session_id}.log",
            logging.Formatter(
                "%(asctime)s | %(metric_type)-12s | %(message)s", datefmt="%H:%M:%S"
            ),
        )

        # Error logger
        self.error_logger = self._create_logger(
            "game_errors",
            f"errors_{self.game_session_id}.log",
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
                datefmt="%H:%M:%S",
            ),
        )

    def _create_logger(
        self, name: str, filename: str, formatter: logging.Formatter
    ) -> logging.Logger:
        """Create and configure a logger."""
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)

        # Clear any existing handlers
        logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(self.log_dir / filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler (only for main logger if debug mode)
        if name == "game_main" and self.log_level <= logging.DEBUG:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    def set_current_tick(self, tick: int):
        """Set the current game tick for logging context."""
        self.current_tick = tick

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.main_logger.debug(message, extra={"tick": self.current_tick, **kwargs})

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.main_logger.info(message, extra={"tick": self.current_tick, **kwargs})

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.main_logger.warning(message, extra={"tick": self.current_tick, **kwargs})

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.main_logger.error(message, extra={"tick": self.current_tick, **kwargs})
        self.error_logger.error(message, extra={"tick": self.current_tick, **kwargs})

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.main_logger.critical(message, extra={"tick": self.current_tick, **kwargs})
        self.error_logger.critical(message, extra={"tick": self.current_tick, **kwargs})

    def log_game_event(self, event_type: GameEventType, message: str, **kwargs):
        """Log a specific game event."""
        self.event_logger.info(
            message,
            extra={"tick": self.current_tick, "event_type": event_type.value, **kwargs},
        )

        # Update statistics
        self._update_stats(event_type, kwargs)

    def log_performance(self, metric_type: str, value: Any, **kwargs):
        """Log performance metrics."""
        message = f"{metric_type}: {value}"
        self.performance_logger.info(
            message, extra={"metric_type": metric_type, **kwargs}
        )

        # Store performance data
        if metric_type == "tick_time":
            self.performance_metrics["tick_times"].append(float(value))
        elif metric_type == "agent_response_time":
            agent_id = kwargs.get("agent_id", "unknown")
            if agent_id not in self.performance_metrics["agent_response_times"]:
                self.performance_metrics["agent_response_times"][agent_id] = []
            self.performance_metrics["agent_response_times"][agent_id].append(
                float(value)
            )

    def start_game(self, **game_info):
        """Log game start."""
        self.performance_metrics["game_start_time"] = datetime.now()
        self.log_game_event(
            GameEventType.GAME_START,
            f"Game started with session ID: {self.game_session_id}",
            **game_info,
        )
        self.info(f"Game configuration: {game_info}")

    def end_game(self, **game_results):
        """Log game end and generate summary."""
        self.performance_metrics["game_end_time"] = datetime.now()
        self.performance_metrics["total_ticks"] = self.current_tick

        self.log_game_event(
            GameEventType.GAME_END,
            f"Game ended after {self.current_tick} ticks",
            **game_results,
        )

        # Generate game summary
        self._generate_game_summary(game_results)

    def log_tick_start(self, tick: int):
        """Log start of a game tick."""
        self.set_current_tick(tick)
        self.debug(f"Tick {tick} started")

    def log_tick_end(self, tick: int, tick_duration: float):
        """Log end of a game tick."""
        self.debug(f"Tick {tick} completed in {tick_duration:.4f}s")
        self.log_performance("tick_time", tick_duration, tick=tick)

    def log_tank_action(self, tank_id: str, action_type: str, details: Dict[str, Any]):
        """Log tank actions."""
        message = f"Tank {tank_id} performed {action_type}"
        if action_type == "spawn":
            self.log_game_event(
                GameEventType.TANK_SPAWN, message, tank_id=tank_id, **details
            )
        elif action_type == "death":
            self.log_game_event(
                GameEventType.TANK_DEATH, message, tank_id=tank_id, **details
            )
        elif action_type == "move":
            self.log_game_event(
                GameEventType.TANK_MOVE, message, tank_id=tank_id, **details
            )
        elif action_type == "shoot":
            self.log_game_event(
                GameEventType.TANK_SHOOT, message, tank_id=tank_id, **details
            )
        elif action_type == "hit":
            self.log_game_event(
                GameEventType.TANK_HIT, message, tank_id=tank_id, **details
            )
        elif action_type == "collision":
            self.log_game_event(
                GameEventType.TANK_COLLISION, message, tank_id=tank_id, **details
            )

    def log_powerup_action(
        self, powerup_id: str, action_type: str, details: Dict[str, Any]
    ):
        """Log power-up related actions."""
        message = f"PowerUp {powerup_id} {action_type}"
        if action_type == "spawn":
            self.log_game_event(
                GameEventType.POWERUP_SPAWN, message, powerup_id=powerup_id, **details
            )
        elif action_type == "collected":
            self.log_game_event(
                GameEventType.POWERUP_COLLECTED,
                message,
                powerup_id=powerup_id,
                **details,
            )
        elif action_type == "despawn":
            self.log_game_event(
                GameEventType.POWERUP_DESPAWN, message, powerup_id=powerup_id, **details
            )

    def log_agent_interaction(
        self,
        agent_id: str,
        action_type: str,
        response_time: Optional[float] = None,
        **details,
    ):
        """Log agent interactions."""
        message = f"Agent {agent_id} {action_type}"
        if action_type == "request":
            self.log_game_event(
                GameEventType.AGENT_REQUEST, message, agent_id=agent_id, **details
            )
        elif action_type == "response":
            self.log_game_event(
                GameEventType.AGENT_RESPONSE, message, agent_id=agent_id, **details
            )
            if response_time:
                self.log_performance(
                    "agent_response_time", response_time, agent_id=agent_id
                )
        elif action_type == "timeout":
            self.log_game_event(
                GameEventType.AGENT_TIMEOUT, message, agent_id=agent_id, **details
            )

    def _update_stats(self, event_type: GameEventType, kwargs: Dict[str, Any]):
        """Update game statistics based on events."""
        if event_type == GameEventType.TANK_SPAWN:
            self.game_stats["tanks_spawned"] += 1
        elif event_type == GameEventType.TANK_DEATH:
            self.game_stats["tanks_killed"] += 1
        elif event_type == GameEventType.TANK_SHOOT:
            self.game_stats["shots_fired"] += 1
        elif event_type == GameEventType.TANK_HIT:
            self.game_stats["hits_landed"] += 1
        elif event_type == GameEventType.POWERUP_SPAWN:
            self.game_stats["powerups_spawned"] += 1
        elif event_type == GameEventType.POWERUP_COLLECTED:
            self.game_stats["powerups_collected"] += 1
        elif event_type == GameEventType.TANK_COLLISION:
            self.game_stats["collisions"] += 1

    def _generate_game_summary(self, game_results: Dict[str, Any]):
        """Generate a comprehensive game summary."""
        if (
            self.performance_metrics["game_start_time"]
            and self.performance_metrics["game_end_time"]
        ):
            total_time = (
                self.performance_metrics["game_end_time"]
                - self.performance_metrics["game_start_time"]
            ).total_seconds()
        else:
            total_time = 0

        avg_tick_time = (
            (
                sum(self.performance_metrics["tick_times"])
                / len(self.performance_metrics["tick_times"])
            )
            if self.performance_metrics["tick_times"]
            else 0
        )

        summary = f"""
=== GAME SUMMARY ===
Session ID: {self.game_session_id}
Total Game Time: {total_time:.2f} seconds
Total Ticks: {self.current_tick}
Average Tick Time: {avg_tick_time:.4f} seconds

=== STATISTICS ===
Tanks Spawned: {self.game_stats["tanks_spawned"]}
Tanks Killed: {self.game_stats["tanks_killed"]}
Shots Fired: {self.game_stats["shots_fired"]}
Hits Landed: {self.game_stats["hits_landed"]}
Hit Accuracy: {(self.game_stats["hits_landed"] / self.game_stats["shots_fired"] * 100) if self.game_stats["shots_fired"] > 0 else 0:.1f}%
PowerUps Spawned: {self.game_stats["powerups_spawned"]}
PowerUps Collected: {self.game_stats["powerups_collected"]}
Collisions: {self.game_stats["collisions"]}

=== RESULTS ===
{game_results}
"""

        self.info(summary)

        # Save summary to separate file
        summary_file = self.log_dir / f"summary_{self.game_session_id}.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary)

    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        return {
            "session_id": self.game_session_id,
            "total_ticks": self.current_tick,
            "average_tick_time": sum(self.performance_metrics["tick_times"])
            / len(self.performance_metrics["tick_times"])
            if self.performance_metrics["tick_times"]
            else 0,
            "min_tick_time": min(self.performance_metrics["tick_times"])
            if self.performance_metrics["tick_times"]
            else 0,
            "max_tick_time": max(self.performance_metrics["tick_times"])
            if self.performance_metrics["tick_times"]
            else 0,
            "agent_response_times": {
                agent_id: {
                    "avg": sum(times) / len(times) if times else 0,
                    "min": min(times) if times else 0,
                    "max": max(times) if times else 0,
                    "count": len(times),
                }
                for agent_id, times in self.performance_metrics[
                    "agent_response_times"
                ].items()
            },
            "game_stats": self.game_stats.copy(),
        }


# Global logger instance
game_logger = GameLogger()


def set_log_level(level: str):
    """Set the global log level."""
    global game_logger
    game_logger = GameLogger(log_level=level)


def get_logger() -> GameLogger:
    """Get the global logger instance."""
    return game_logger
