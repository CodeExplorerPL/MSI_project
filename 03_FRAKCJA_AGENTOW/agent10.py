"""
Fuzzy DQN agent for 5v5 tank battles.

Key features:
- ANFIS-style Q-network (fuzzy rules + Sugeno consequents),
- replay buffer + target network,
- reward shaping focused on useful combat behavior,
- optional online training mode,
- FastAPI endpoints compatible with the game engine.
"""

from __future__ import annotations

try:
    from comet_ml import Experiment
except ImportError:
    Experiment = None

import matplotlib.pyplot as plt
import argparse
import copy
import math
import os
import random
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import Body, FastAPI
from pydantic import BaseModel
import uvicorn
import matplotlib
matplotlib.use("Agg")       # zeby uniknac bledow

from dependencyAgent10.ANFISDQN import ANFISDQN

# Add engine paths for local runs.
current_dir = os.path.dirname(os.path.abspath(__file__))
controller_dir = os.path.join(os.path.dirname(current_dir), "02_FRAKCJA_SILNIKA", "controller")
engine_dir = os.path.join(os.path.dirname(current_dir), "02_FRAKCJA_SILNIKA")
sys.path.insert(0, controller_dir)
sys.path.insert(0, engine_dir)


STATE_DIM = 23
DEFAULT_MODEL_PATH = os.path.join(current_dir, "./dependencyAgent10/fuzzy_dqn_model_best.pt")
MAP_WIDTH = 200.0
MAP_HEIGHT = 200.0
COMET_LOG_EVERY = 20
ACTION_DIM = 2
# TODO: Scale actions using per-tank limits from my_tank_status (_top_speed, _heading_spin_rate, _barrel_spin_rate)
MAX_MOVE_SPEED = 5.0
MAX_HEADING_DELTA = 5.0
MAX_BARREL_DELTA = 5.0


class ActionCommand(BaseModel):
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: Optional[str] = None
    should_fire: bool = False


@dataclass(frozen=True)
class ActionSpec:
    name: str
    move_speed: float
    heading_rotation_angle: float
    barrel_rotation_angle: float
    should_fire: bool


@dataclass
class TargetInfo:
    distance: float
    hull_error: float
    barrel_error: float
    position: Dict[str, float]


class ReplayBuffer:
    """Simple ring buffer for off-policy learning."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

        self.index = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = float(reward)
        self.next_states[self.index] = next_state
        self.dones[self.index] = float(done)

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)

        states = torch.from_numpy(self.states[idx]).to(device)
        actions = torch.from_numpy(self.actions[idx]).to(device)
        rewards = torch.from_numpy(self.rewards[idx]).to(device)
        next_states = torch.from_numpy(self.next_states[idx]).to(device)
        dones = torch.from_numpy(self.dones[idx]).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size


@dataclass
class Observation:
    vector: np.ndarray
    enemy_visible: bool
    enemy_dist: float
    enemy_barrel_error: float
    shot_blocked: bool
    enemy_hull_error: float
    ally_fire_risk: bool
    obstacle_ahead: bool
    danger_ahead: bool
    powerup_visible: bool
    powerup_dist: float
    hp_ratio: float
    shield_ratio: float
    can_fire: bool
    reload_norm: float
    vision_range: float = 40.0
    ammo_range: float = 50.0


class StateEncoder:
    """Converts engine payload into a normalized feature vector and helper signals."""

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(value, hi))

    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    @staticmethod
    def _distance(a: Dict[str, float], b: Dict[str, float]) -> float:
        return math.hypot(b["x"] - a["x"], b["y"] - a["y"])

    @staticmethod
    def _angle_to(a: Dict[str, float], b: Dict[str, float]) -> float:
        return math.degrees(math.atan2(b["y"] - a["y"], b["x"] - a["x"]))

    def _can_fire(self, my_status: Dict[str, Any], reload_timer: float) -> bool:
        if reload_timer > 0.0:
            return False

        ammo = my_status.get("ammo", {})
        for slot in ammo.values():
            count = int((slot or {}).get("count", 0) or 0)
            if count > 0:
                return True
        return False

    def _nearest_enemy(
        self,
        my_pos: Dict[str, float],
        my_team: Optional[int],
        seen_tanks: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        enemies = [tank for tank in seen_tanks if tank.get("team") != my_team]
        if not enemies:
            return None

        return min(
            enemies,
            key=lambda tank: float(
                tank.get("distance")
                if tank.get("distance") is not None
                else self._distance(my_pos, tank.get("position", {"x": 0.0, "y": 0.0}))
            ),
        )

    def _ally_in_fire_line(
        self,
        my_pos: Dict[str, float],
        my_team: Optional[int],
        barrel_abs: float,
        seen_tanks: List[Dict[str, Any]],
    ) -> bool:
        for tank in seen_tanks:
            if tank.get("team") != my_team:
                continue
            ally_pos = tank.get("position", {"x": 0.0, "y": 0.0})
            angle = self._angle_to(my_pos, ally_pos)
            error = abs(self.normalize_angle(angle - barrel_abs))
            if error < 4.0:
                return True
        return False

    def _friendly_blocks_shot(
        self,
        my_pos: Dict[str, float],
        my_team: Optional[int],
        barrel_abs: float,
        seen_tanks: List[Dict[str, Any]],
        target_distance: float,
        max_range: float = 25.0,
        half_angle: float = 5.0,
    ) -> bool:
        max_dist = min(max_range, max(target_distance, 0.0))
        for tank in seen_tanks:
            if tank.get("team") != my_team:
                continue
            ally_pos = tank.get("position")
            if not ally_pos:
                raise ValueError(f"Missing ally position in sensor data: {tank}")
            ally_dist = self._distance(my_pos, ally_pos)
            if ally_dist >= target_distance:
                continue
            if ally_dist > max_dist:
                continue
            angle = self._angle_to(my_pos, ally_pos)
            rel = abs(self.normalize_angle(angle - barrel_abs))
            if rel <= half_angle:
                return True
        return False

    def _has_object_ahead(
        self,
        my_pos: Dict[str, float],
        my_heading: float,
        objects: List[Dict[str, Any]],
        max_dist: float,
        half_angle: float,
    ) -> bool:
        for item in objects:
            pos = item.get("position")
            if not pos:
                raise ValueError(f"Missing position for object in sensor data: {item}")

            distance = self._distance(my_pos, pos)
            if distance > max_dist:
                continue

            angle = self._angle_to(my_pos, pos)
            rel = abs(self.normalize_angle(angle - my_heading))
            if rel < half_angle:
                return True

        return False

    def _shot_blocked_by_obstacle(
        self,
        my_pos: Dict[str, float],
        barrel_abs: float,
        seen_obstacles: List[Dict[str, Any]],
        target_distance: float,
        max_range: float = 25.0,
        half_angle: float = 5.0,
    ) -> bool:
        max_dist = min(max_range, max(target_distance, 0.0))
        for item in seen_obstacles:
            pos = item.get("position")
            if not pos:
                raise ValueError(f"Missing position for obstacle in sensor data: {item}")
            distance = self._distance(my_pos, pos)
            if distance > max_dist:
                continue
            angle = self._angle_to(my_pos, pos)
            rel = abs(self.normalize_angle(angle - barrel_abs))
            if rel <= half_angle:
                return True
        return False

    def encode(
        self,
        my_status: Dict[str, Any],
        sensor_data: Dict[str, Any],
        enemies_remaining: int,
        enemy_target_pos: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Observation, bool]:
        my_pos = my_status.get("position", {"x": 0.0, "y": 0.0})
        my_team = my_status.get("_team")

        my_heading = float(my_status.get("heading", 0.0) or 0.0)
        my_barrel = float(my_status.get("barrel_angle", 0.0) or 0.0)
        barrel_abs = my_heading + my_barrel

        max_hp = float(my_status.get("_max_hp", 100.0) or 100.0)
        max_shield = float(my_status.get("_max_shield", 100.0) or 100.0)

        hp_ratio = self._clamp(float(my_status.get("hp", 0.0) or 0.0) / max_hp, 0.0, 1.0)
        shield_ratio = self._clamp(float(my_status.get("shield", 0.0) or 0.0) / max_shield, 0.0, 1.0)

        reload_timer = float(my_status.get("_reload_timer", 0.0) or 0.0)
        reload_norm = self._clamp(reload_timer / 10.0, 0.0, 1.0)

        ammo = my_status.get("ammo", {})
        heavy_ratio = self._clamp(float(ammo.get("HEAVY", {}).get("count", 0) or 0) / 5.0, 0.0, 1.0)
        light_ratio = self._clamp(float(ammo.get("LIGHT", {}).get("count", 0) or 0) / 15.0, 0.0, 1.0)
        long_ratio = self._clamp(
            float(ammo.get("LONG_DISTANCE", {}).get("count", 0) or 0) / 10.0,
            0.0,
            1.0,
        )

        seen_tanks = sensor_data.get("seen_tanks", [])
        nearest_enemy = self._nearest_enemy(my_pos, my_team, seen_tanks)

        enemy_visible = nearest_enemy is not None
        enemy_dist = 1.0
        enemy_hull_error = 0.5
        enemy_barrel_error = 0.5
        enemy_distance_raw: Optional[float] = None

        enemy_pos = None
        if nearest_enemy is not None:
            enemy_pos = nearest_enemy.get("position", {"x": 0.0, "y": 0.0})
            distance_raw = nearest_enemy.get("distance")
            if distance_raw is None:
                raise ValueError(f"Missing enemy distance in sensor data: {nearest_enemy}")
            enemy_distance_raw = float(distance_raw)

            vision_range = float(my_status.get("_vision_range", 40.0) or 40.0)
            enemy_dist = self._clamp(float(distance_raw) / max(vision_range, 1.0), 0.0, 1.0)

        else:
            enemy_pos = {"x": float(enemy_target_pos[0]), "y": float(enemy_target_pos[1])}
            distance_raw = self._distance(my_pos, enemy_pos)
            vision_range = float(my_status.get("_vision_range", 40.0) or 40.0)
            enemy_dist = self._clamp(float(distance_raw) / max(vision_range, 1.0), 0.0, 1.0)

        target_angle =  self._angle_to(my_pos, enemy_pos)
        enemy_hull_error = (self.normalize_angle(target_angle - my_heading) / 180.0 + 1.0) * 0.5
        enemy_barrel_error = (self.normalize_angle(target_angle - barrel_abs) / 180.0 + 1.0) * 0.5

        ally_fire_risk = self._ally_in_fire_line(my_pos, my_team, barrel_abs, seen_tanks)

        seen_obstacles = sensor_data.get("seen_obstacles", [])
        shot_blocked = False
        if enemy_visible and enemy_distance_raw is not None:
            obstacle_blocks = self._shot_blocked_by_obstacle(
                my_pos,
                barrel_abs,
                seen_obstacles,
                target_distance=enemy_distance_raw,
                max_range=25.0,
                half_angle=5.0,
            )
            ally_blocks = self._friendly_blocks_shot(
                my_pos,
                my_team,
                barrel_abs,
                seen_tanks,
                target_distance=enemy_distance_raw,
                max_range=25.0,
                half_angle=5.0,
            )
            shot_blocked = obstacle_blocks or ally_blocks
        obstacle_ahead = self._has_object_ahead(
            my_pos,
            my_heading,
            seen_obstacles,
            max_dist=12.0,
            half_angle=28.0,
        )

        seen_terrains = sensor_data.get("seen_terrains", [])
        dangerous_terrains = [
            terrain
            for terrain in seen_terrains
            if terrain.get("type") in {"Water", "PotholeRoad"}
        ]
        danger_ahead = self._has_object_ahead(
            my_pos,
            my_heading,
            dangerous_terrains,
            max_dist=10.0,
            half_angle=35.0,
        )

        seen_powerups = sensor_data.get("seen_powerups", [])
        powerup_visible = len(seen_powerups) > 0
        powerup_dist = 1.0
        if powerup_visible:
            nearest_powerup = min(
                seen_powerups,
                key=lambda p: self._distance(my_pos, p.get("position", {"x": 0.0, "y": 0.0})),
            )
            dist_raw = self._distance(my_pos, nearest_powerup.get("position", {"x": 0.0, "y": 0.0}))
            vision_range = float(my_status.get("_vision_range", 40.0) or 40.0)
            powerup_dist = self._clamp(dist_raw / max(vision_range, 1.0), 0.0, 1.0)

        top_speed = float(my_status.get("_top_speed", 5.0) or 5.0)
        speed = float(my_status.get("move_speed", 0.0) or 0.0)
        speed_ratio = self._clamp(speed / max(top_speed, 1.0), -1.0, 1.0)
        speed_ratio = (speed_ratio + 1.0) * 0.5

        enemies_remaining_norm = self._clamp(float(enemies_remaining) / 5.0, 0.0, 1.0)
        can_fire = self._can_fire(my_status, reload_timer)

        _AMMO_RANGES = {"HEAVY": 25.0, "LIGHT": 50.0, "LONG_DISTANCE": 100.0}
        obs_vision_range = float(my_status.get("_vision_range", 40.0) or 40.0)
        ammo_loaded = str(my_status.get("ammo_loaded") or "LIGHT").upper()
        obs_ammo_range = _AMMO_RANGES.get(ammo_loaded, 50.0) / vision_range
        is_at_range = obs_ammo_range > enemy_dist

        vector = np.array(
            [
                hp_ratio,
                shield_ratio,
                reload_norm,
                heavy_ratio,
                light_ratio,
                long_ratio,
                1.0 if enemy_visible else 0.0,
                enemy_dist,
                enemy_barrel_error,
                enemy_hull_error,
                1.0 if ally_fire_risk else 0.0,
                1.0 if obstacle_ahead else 0.0,
                1.0 if danger_ahead else 0.0,
                1.0 if powerup_visible else 0.0,
                powerup_dist,
                speed_ratio,
                enemies_remaining_norm,
            ],
            dtype=np.float32,
        )

        return Observation(
            vector=vector,
            enemy_visible=enemy_visible,
            enemy_dist=enemy_dist,
            enemy_barrel_error=enemy_barrel_error,
            shot_blocked=shot_blocked,
            enemy_hull_error=enemy_hull_error,
            ally_fire_risk=ally_fire_risk,
            obstacle_ahead=obstacle_ahead,
            danger_ahead=danger_ahead,
            powerup_visible=powerup_visible,
            powerup_dist=powerup_dist,
            hp_ratio=hp_ratio,
            shield_ratio=shield_ratio,
            can_fire=can_fire,
            reload_norm=reload_norm,
        ), is_at_range


@dataclass
class AgentConfig:
    state_dim: int = STATE_DIM
    n_rules: int = 32
    mf_type: str = "gaussian"

    gamma: float = 0.9
    actor_lr: float = 1e-5
    critic_lr: float = 1e-4
    tau: float = 0.01
    action_noise_start: float = 0.05
    action_noise_end: float = 0.05
    action_noise_decay_steps: int = 5_000
    batch_size: int = 512
    replay_capacity: int = 12_500
    warmup_steps: int = 512
    train_every: int = 1
    target_sync_every: int = 1

    frame_skip: int = 1
    save_every_games: int = 1

    model_path: str = DEFAULT_MODEL_PATH
    best_model_path: Optional[str] = None
    mock_barrel_model_path: Optional[str] = None
    mock_shoot_model_path: Optional[str] = None
    seed: int = 1
    map_name: str = ""


class FuzzyDQNAgent:
    def __init__(self, name: str, config: AgentConfig, training_enabled: bool, load_checkpoint: bool = True):
        self.name = name
        self.config = config
        self.config.model_path = self._normalize_optional_path(self.config.model_path)
        self.config.best_model_path = self._normalize_optional_path(self.config.best_model_path)
        self.training_enabled = training_enabled
        self.map_name = str(config.map_name or "")
        self.checkpoints: list
    
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = StateEncoder()

        self.actor = ANFISDQN(
            n_inputs=config.state_dim,
            n_rules=config.n_rules,
            n_actions=ACTION_DIM,
            mf_type=config.mf_type,
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_target.eval()

        self.critic = ANFISDQN(
            n_inputs=config.state_dim + ACTION_DIM,
            n_rules=config.n_rules,
            n_actions=1,
            mf_type=config.mf_type,
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_target.eval()

        # Agent_2 logic state
        self.previous_enemy_state: Dict[str, Dict[str, Any]] = {}
        self.scan_direction = 1.0
        self.scan_speed = 2.5

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.replay = ReplayBuffer(config.replay_capacity, config.state_dim, ACTION_DIM)

        self.total_steps = 0
        self.train_steps = 0
        self.games_played = 0
        self.last_loss: Optional[float] = None
        self.current_episode_score = 0.0
        self.last_episode_score = 0.0

        self.experiment = None
        if self.training_enabled:
            if Experiment:
                threading.Thread(target=self._init_comet, daemon=True).start()
            else:
                print(f"[{self.name}] WARNING: comet_ml module not found. Experiment logging disabled.")

        self.last_observation: Optional[Observation] = None
        self.last_action_vector: Optional[np.ndarray] = None
        self.last_command = ActionCommand()
        self.last_fire_tick = -10_000
        self.prev_enemies_remaining: Optional[int] = None
        self.was_destroyed = False
        self.best_score = float("-inf")
        self.last_status: Optional[Dict[str, Any]] = None
        self.checkpoint_loaded = False
        self.checkpoint_error: Optional[str] = None

        self.lock = threading.Lock()

        self.best_model_path = self._resolve_best_model_path()
        if load_checkpoint:
            self.checkpoint_loaded = self._load_checkpoint_if_available()
        print(
            f"[{self.name}] ready | training={self.training_enabled} "
            f"device={self.device} rules={self.config.n_rules} mf={self.config.mf_type} "
            f"save_path={self.config.model_path} checkpoint_loaded={self.checkpoint_loaded}"
        )
        if self.checkpoint_error:
            print(f"[{self.name}] checkpoint_error={self.checkpoint_error}")

        self.trace_positions: List[Tuple[float, float]] = []
        self.trace_hp: List[float] = []
        self.trace_shots: List[Tuple[float, float, float]] = []
        self.trace_allies: List[Tuple[float, float]] = []
        self.trace_enemies: List[Tuple[float, float]] = []
        self.trace_labels: List[str] = []
        self.pos_history: List[Tuple[float, float]] = []
        self.trace_actor_raw: List[np.ndarray] = []
        self.last_actor_raw: Optional[np.ndarray] = None
        self.trace_mock_actions: List[np.ndarray] = []
        self.last_mock_action: Optional[np.ndarray] = None
        self.trace_damage_taken: List[float] = []
        self.trace_hit_target: List[float] = []
        self.trace_friendly_hit: List[float] = []
        self.trace_should_fire: List[float] = []
        self.episode_reward_total = 0.0
        self.episode_reward_parts: Dict[str, float] = {}
        self.episode_reward_parts_steps = 0
        self.frontier_min_x: Optional[float] = None
        self.frontier_max_x: Optional[float] = None
        self.frontier_min_y: Optional[float] = None
        self.frontier_max_y: Optional[float] = None
        self.reward_parts_history: Dict[str, List[float]] = {}
        self.target: Optional[Tuple[float, float]] = None
        self.last_dangerous_terrain_pos: Optional[Tuple[float, float]] = None
        self.last_min_corner = None
        self.last_max_corner = None

    @staticmethod
    def _normalize_optional_path(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        return os.path.abspath(os.path.expanduser(path))

    def _init_comet(self) -> None:
        try:
            self.experiment = Experiment(
                api_key="L2PzW7c3YM3WqM5hNfCsloeLZ",
                project_name="msi-projekt",
                workspace="kluski777",
                auto_output_logging="simple"
            )
            self.experiment.set_name(self.name)
            self.experiment.log_parameters(vars(self.config))
        except Exception as e:
            print(f"[{self.name}] Failed to initialize Comet ML: {e}")

    def _resolve_best_model_path(self) -> Optional[str]:
        if self.config.best_model_path:
            return self.config.best_model_path
        if not self.config.model_path:
            return None
        root, ext = os.path.splitext(self.config.model_path)
        ext = ext or ".pt"
        return f"{root}_best{ext}"

    def _scale_action_tensor(self, action: torch.Tensor) -> torch.Tensor:
        move = action[:, 0] * MAX_MOVE_SPEED
        heading = action[:, 1] * MAX_HEADING_DELTA
        return torch.stack([move, heading], dim=1)

    def _current_action_noise_std(self) -> float:
        decay_steps = max(1, int(self.config.action_noise_decay_steps))
        t = min(max(self.total_steps, 0), decay_steps)
        frac = 1.0 - (t / float(decay_steps))
        return float(self.config.action_noise_end + (self.config.action_noise_start - self.config.action_noise_end) * frac)

    def _select_action(self, state_vector: np.ndarray, training: bool) -> np.ndarray:
        state = torch.from_numpy(state_vector).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw = self.actor(state)
            self.last_actor_raw = raw.squeeze(0).detach().cpu().numpy()
            action = torch.tanh(raw)
            if training:
                noise_std = self._current_action_noise_std()
                noise = torch.normal(
                    mean=0.0,
                    std=noise_std,
                    size=action.shape,
                    device=action.device,
                )
                action = torch.clamp(action + noise, -1.0, 1.0)
            scaled = self._scale_action_tensor(action)
        return scaled.squeeze(0).cpu().numpy()

    @staticmethod
    def _ammo_counts(my_status: Dict[str, Any]) -> Dict[str, int]:
        ammo_data = my_status.get("ammo", {}) or {}
        counts: Dict[str, int] = {}
        for ammo_name, slot in ammo_data.items():
            key = str(ammo_name).upper()
            count = int((slot or {}).get("count", 0) or 0)
            counts[key] = max(0, count)
        return counts

    @staticmethod
    def _safe_nonnegative_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(parsed):
            return float(default)
        return float(max(0.0, parsed))

    @staticmethod
    def _safe_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "true", "t", "yes", "y"}:
                return True
            if v in {"0", "false", "f", "no", "n"}:
                return False
            return default
        return default
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    @staticmethod
    def angle_to(source: Dict[str, float], target: Dict[str, float]) -> float:
        dx = target["x"] - source["x"]
        dy = target["y"] - source["y"]
        return math.degrees(math.atan2(dy, dx))

    # def _dangerous_terrain_angle_error(
    #     self,
    #     my_status: Dict[str, Any],
    #     sensor_data: Dict[str, Any],
    # ) -> Optional[float]:
    #     my_pos = my_status.get("position", {"x": 0.0, "y": 0.0})
    #     my_heading = float(my_status.get("heading", 0.0) or 0.0)
    #     seen_terrains = sensor_data.get("seen_terrains", [])

    #     dangerous_terrains = []
    #     for terrain in seen_terrains:
    #         pos = terrain.get("position")
    #         if not pos:
    #             continue
    #         dmg = float(terrain.get("dmg", 0.0) or 0.0)
    #         if dmg > 0.0:
    #             dangerous_terrains.append(terrain)

    #     visible_angle_errors: List[float] = []
    #     if dangerous_terrains:
    #         nearest = min(
    #             dangerous_terrains,
    #             key=lambda terrain: self.encoder._distance(
    #                 my_pos,
    #                 terrain.get("position", {"x": 0.0, "y": 0.0}),
    #             ),
    #         )
    #         nearest_pos = nearest.get("position", {"x": 0.0, "y": 0.0})
    #         tx = float(nearest_pos.get("x", 0.0) or 0.0)
    #         ty = float(nearest_pos.get("y", 0.0) or 0.0)
    #         self.last_dangerous_terrain_pos = (tx, ty)

    #         for terrain in dangerous_terrains:
    #             terrain_pos = terrain.get("position", {"x": 0.0, "y": 0.0})
    #             terrain_angle = self.angle_to(my_pos, terrain_pos)
    #             visible_angle_errors.append(abs(self.normalize_angle(terrain_angle - my_heading)))

    #     memory_angle_error: Optional[float] = None
    #     if self.last_dangerous_terrain_pos is not None:
    #         tx, ty = self.last_dangerous_terrain_pos
    #         memory_pos = {"x": tx, "y": ty}
    #         memory_angle = self.angle_to(my_pos, memory_pos)
    #         memory_angle_error = abs(self.normalize_angle(memory_angle - my_heading))

    #     if visible_angle_errors and memory_angle_error is not None:
    #         return min(min(visible_angle_errors), memory_angle_error)
    #     if visible_angle_errors:
    #         return min(visible_angle_errors)
    #     return memory_angle_error

    def _find_nearest_enemy_agent2(
        self,
        my_status: Dict[str, Any],
        sensor_data: Dict[str, Any],
        current_tick: int,
    ) -> Optional[TargetInfo]:
        my_pos = my_status.get("position", {"x": 0.0, "y": 0.0})
        my_team = my_status.get("_team")
        my_heading = my_status.get("heading", 0.0)
        my_barrel = my_status.get("barrel_angle", 0.0)

        enemies = [
            tank
            for tank in sensor_data.get("seen_tanks", [])
            if tank.get("team") != my_team
        ]
        if not enemies:
            return None

        nearest = min(enemies, key=lambda tank: float(tank.get("distance", 1e9)))
        enemy_pos = nearest.get("position", {"x": 0.0, "y": 0.0})

        # --- Predykcja pozycji (Aim Leading / Latency Compensation) ---
        enemy_id = nearest.get("id")
        predicted_pos = enemy_pos.copy()

        if enemy_id:
            if enemy_id in self.previous_enemy_state:
                prev = self.previous_enemy_state[enemy_id]
                dt = current_tick - prev["tick"]
                if 0 < dt < 5:  # Tylko jeśli dane są świeże
                    vx = (enemy_pos["x"] - prev["x"]) / dt
                    vy = (enemy_pos["y"] - prev["y"]) / dt
                    # Przewidujemy pozycję o 1.5 klatki do przodu (kompensacja czasu reakcji)
                    predicted_pos["x"] += vx * 1.5
                    predicted_pos["y"] += vy * 1.5
            
            self.previous_enemy_state[enemy_id] = {
                "x": enemy_pos["x"], "y": enemy_pos["y"], "tick": current_tick
            }

        target_angle = self.angle_to(my_pos, predicted_pos)
        hull_error = self.normalize_angle(target_angle - my_heading)
        barrel_abs = my_heading + my_barrel
        barrel_error = self.normalize_angle(target_angle - barrel_abs)

        return TargetInfo(
            distance=float(nearest.get("distance", 999.0)),
            hull_error=hull_error,
            barrel_error=barrel_error,
            position=enemy_pos,
        )

    def _ally_in_fire_line_agent2(self, my_status: Dict[str, Any], sensor_data: Dict[str, Any]) -> bool:
        my_pos = my_status.get("position", {"x": 0.0, "y": 0.0})
        my_team = my_status.get("_team")
        my_heading = my_status.get("heading", 0.0)
        my_barrel = my_status.get("barrel_angle", 0.0)
        barrel_abs = my_heading + my_barrel

        for tank in sensor_data.get("seen_tanks", []):
            if tank.get("team") != my_team:
                continue
            ally_pos = tank.get("position", {"x": 0.0, "y": 0.0})
            angle = self.angle_to(my_pos, ally_pos)
            error = abs(self.normalize_angle(angle - barrel_abs))
            if error < 4.0:
                return True
        return False

    def _fuzzy_shoot_decision(self, barrel_error: float, distance: float) -> bool:
        abs_error = abs(barrel_error) 
        mu_error_small = max(0.0, 1.0 - (abs_error / 3.0))
        if abs_error < 6.0:
            mu_error_medium = max(0.0, (abs_error - 2.0) / 4.0)
        else:
            mu_error_medium = max(0.0, (10.0 - abs_error) / 4.0)
        mu_dist_close = max(0.0, 1.0 - (distance / 20.0))
        if distance < 30.0:
            mu_dist_medium = max(0.0, (distance - 10.0) / 20.0)
        else:
            mu_dist_medium = max(0.0, (50.0 - distance) / 20.0)
        rule1 = mu_error_small
        rule2 = min(mu_error_medium, mu_dist_close) * 0.6
        rule3 = min(mu_error_medium, mu_dist_medium) * 0.3
        fire_confidence = max(rule1, rule2, rule3)
        return fire_confidence > 0.5

    def _select_ammo_agent2(
        self,
        my_status: Dict[str, Any],
        target: Optional[TargetInfo],
    ) -> Optional[str]:
        counts = self._ammo_counts(my_status)
        if not counts:
            return None
        if target is None:
            current = str(my_status.get("ammo_loaded") or "").upper()
            if current and counts.get(current, 0) > 0:
                return current
            return max(counts.items(), key=lambda item: item[1])[0] if counts else None
        if target.distance > 50.0:
            preferred = ["LONG_DISTANCE", "LIGHT", "HEAVY"]
        elif target.distance > 25.0:
            preferred = ["LIGHT", "HEAVY", "LONG_DISTANCE"]
        else:
            preferred = ["HEAVY", "LIGHT", "LONG_DISTANCE"]
        for ammo_name in preferred:
            if counts.get(ammo_name, 0) > 0:
                return ammo_name
        current = str(my_status.get("ammo_loaded") or "").upper()
        if current and counts.get(current, 0) > 0:
            return current
        return max(counts.items(), key=lambda item: item[1])[0] if counts else None

    def _to_command(
        self,
        action_vec: np.ndarray,
        my_status: Dict[str, Any],
        obs: Observation,
        sensor_data: Dict[str, Any],
        current_tick: int,
        is_in_shooting_range: bool = True
    ) -> ActionCommand:
        move_speed = float(action_vec[0])
        heading_rotation = float(action_vec[1])
        
        # --- Agent_2 Logic for Barrel & Fire ---
        target = self._find_nearest_enemy_agent2(my_status, sensor_data, current_tick)
        
        barrel_rotation = 0.0
        should_fire = False
        ammo_to_load = None

        if target is not None:
            # Combat Action
            max_barrel_step = 3.0
            barrel_rotation = max(-max_barrel_step, min(max_barrel_step, target.barrel_error))
            
            reload_timer = float(my_status.get("_reload_timer", 0.0) or 0.0)
            if (
                self._fuzzy_shoot_decision(target.barrel_error, target.distance)
                and not self._ally_in_fire_line_agent2(my_status, sensor_data)
                and reload_timer <= 0.0
                # and is_in_shooting_range
            ):
                should_fire = True
            
            ammo_to_load = self._select_ammo_agent2(my_status, target)
        else:
            # Exploration Action
            barrel_rotation = self.scan_speed * self.scan_direction
            ammo_to_load = self._select_ammo_agent2(my_status, None)

        # Record fake mock action for plotting consistency
        self.last_mock_action = np.array([barrel_rotation / 5.0, 1.0 if should_fire else 0.0], dtype=np.float32)

        return ActionCommand(
            barrel_rotation_angle=barrel_rotation,
            heading_rotation_angle=heading_rotation,
            move_speed=move_speed,
            should_fire=should_fire,
            ammo_to_load=ammo_to_load,
        )

    def _compute_step_reward(
        self,
        prev_obs: Observation,
        current_obs: Observation,
        action: ActionCommand,
        enemies_remaining: int,
        current_tick: int,
        current_pos: Tuple[float, float],
        is_enemy_at_ammo_range: bool,
        dangerous_terrain_angle_error: Optional[float] = None,
    ) -> float:
        from collections import defaultdict
        parts: Dict[str, float] = defaultdict(float)
        # Observation fields:
            # - vector
                # - hp_ratio
                # - shield_ratio
                # - reload_norm
                # - heavy_ratio
                # - light_ratio
                # - long_ratio
                # - enemy_visible (1.0/0.0)
                # - enemy_dist
                # - enemy_barrel_error
                # - enemy_hull_error
                # - ally_fire_risk (1.0/0.0)
                # - obstacle_ahead (1.0/0.0)
                # - danger_ahead (1.0/0.0)
                # - powerup_visible (1.0/0.0)
                # - powerup_dist
                # - speed_ratio
                # - enemies_remaining_norm
                # - x_norm
                # - y_norm
                # - dx_recent
                # - dy_recent
                # - dx_prev
                # - dy_prev
            # - enemy_visible: nearest enemy is visible
            # - enemy_dist: normalized distance to nearest enemy (0..1)
            # - enemy_barrel_error: normalized barrel angle error to enemy (0..1, 0.5=center)
            # - enemy_hull_error: normalized hull angle error to enemy (0..1, 0.5=center)
            # - ally_fire_risk: ally within firing line
            # - obstacle_ahead: obstacle within forward cone
            # - danger_ahead: dangerous terrain ahead
            # - powerup_visible: any powerup visible
            # - powerup_dist: normalized distance to nearest powerup (0..1)
            # - hp_ratio: hp / max_hp (0..1)
            # - shield_ratio: shield / max_shield (0..1)
            # - can_fire: reloaded + ammo available
            # - reload_norm: normalized reload timer (0..1)
        # Action fields:
            # barrel_rotation_angle: float = 0.0
            # heading_rotation_angle: float = 0.0
            # move_speed: float = 0.0
            # should_fire: bool = False

        if not is_enemy_at_ammo_range or not current_obs.enemy_visible:
            parts["approaching_enemy"] = (0.5 - current_obs.vector[15]) * 40 * abs(current_obs.enemy_hull_error - 0.5)
        else:
            print(f'No strzela! {is_enemy_at_ammo_range=}, {current_obs.enemy_visible=}')

        hp_delta = current_obs.hp_ratio - prev_obs.hp_ratio
        if not np.isclose(hp_delta, 0.0):
            parts["hp_loss"] = hp_delta * 5_000 # moze byc spokojnie powyzej 1.
        if current_obs.danger_ahead: # jak sie nie nauczy to zostanie sie przy tym
            parts["danger_ahead"] = -100
        # if dangerous_terrain_angle_error is not None and current_obs.danger_ahead:
        #     parts["danger_alignment"] = -100 * dangerous_terrain_angle_error
        
        if current_obs.obstacle_ahead:
            parts["obstacle_nearby"] = -7.5
        if prev_obs.obstacle_ahead and current_obs.obstacle_ahead:
            parts["obstacle_sustained"] = -10.0

        if not current_obs.enemy_visible:
            speed_norm = abs(action.move_speed) / MAX_MOVE_SPEED
            if speed_norm < 0.2:
                parts["idle_penalty"] = -0.4 * (1.0 - speed_norm / 0.2)

        delta = action.heading_rotation_angle / MAX_HEADING_DELTA
        parts["rotation"] = -0.75 * (delta) ** 2

        recent = self.pos_history[-200:] or [current_pos]
        prev = self.pos_history[-400:-200] or recent
        spawn = self.pos_history[0] if self.pos_history else current_pos
        
        centroid = lambda pts: (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        
        if current_tick < 200:
            rc = pc = spawn
        else:
            rc = centroid(recent)
            pc = spawn if current_tick < 400 else centroid(prev)

        var_r = sum((p[0] - rc[0]) ** 2 + (p[1] - rc[1]) ** 2 for p in recent) / len(recent)
        var_p = sum((p[0] - pc[0]) ** 2 + (p[1] - pc[1]) ** 2 for p in prev) / len(prev)

        parts["variance_recent"] = var_r / 5
        parts["variance_prev"] = var_p / 5
        parts["centroid_bonus"] = parts["variance_recent"] + parts["variance_prev"]

        reward = sum(parts.values())
        self.episode_reward_total += reward
        for key, value in parts.items():
            self.episode_reward_parts[key] = self.episode_reward_parts.get(key, 0.0) + value
        self.episode_reward_parts_steps += 1

        return reward


    def _maybe_train(self) -> None:
        # min_required = max(self.config.batch_size, self.config.warmup_steps)
        if len(self.replay) < self.config.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.config.batch_size,
            self.device,
        )

        with torch.no_grad():
            next_raw = self.actor_target(next_states)
            next_action = torch.tanh(next_raw)
            next_action = self._scale_action_tensor(next_action)
            next_action = torch.sigmoid(next_action)
            next_q = self.critic_target(torch.cat([next_states, next_action], dim=1)).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.config.gamma * next_q

        actions = torch.sigmoid(actions)
        current_q = self.critic(torch.cat([states, actions], dim=1)).squeeze(1)
        critic_loss = F.mse_loss(current_q, target_q)
        self.last_loss = float(critic_loss.item())

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actor_raw = self.actor(states)
        actor_action = torch.tanh(actor_raw)
        actor_action = self._scale_action_tensor(actor_action)
        actor_action = torch.sigmoid(actor_action)
        raw_penalty = 6e-2 * actor_raw.pow(2).mean()
        actor_loss = -self.critic(torch.cat([states, actor_action], dim=1)).mean() + raw_penalty

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0)
        self.actor_optimizer.step()

        tau = float(self.config.tau)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        if self.experiment and (self.train_steps % COMET_LOG_EVERY == 0):
            self.experiment.log_metric("critic_loss", self.last_loss, step=self.train_steps)
            self.experiment.log_metric("actor_loss", float(actor_loss.item()), step=self.train_steps)
            self.experiment.log_metric("q_mean", current_q.mean().item(), step=self.train_steps)
            self.experiment.log_metric("action_noise_std", self._current_action_noise_std(), step=self.train_steps)

        self.train_steps += 1

    def _checkpoint_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "train_steps": self.train_steps,
            "games_played": self.games_played,
            "best_score": self.best_score,
            "config": {
                "state_dim": self.config.state_dim,
                "n_rules": self.config.n_rules,
                "mf_type": self.config.mf_type,
                "gamma": self.config.gamma,
                "actor_lr": self.config.actor_lr,
                "critic_lr": self.config.critic_lr,
                "tau": self.config.tau,
                "action_noise_start": self.config.action_noise_start,
                "action_noise_end": self.config.action_noise_end,
                "action_noise_decay_steps": self.config.action_noise_decay_steps,
                "batch_size": self.config.batch_size,
                "replay_capacity": self.config.replay_capacity,
                "warmup_steps": self.config.warmup_steps,
                "train_every": self.config.train_every,
                "target_sync_every": self.config.target_sync_every,
                "frame_skip": self.config.frame_skip,
            },
            "reward_parts_history": dict(self.reward_parts_history),
        }
        return payload

    def save_checkpoint(self, path: Optional[str] = None, label: str = "checkpoint") -> None:
        save_path = path or self.config.model_path
        if not save_path:
            return

        model_dir = os.path.dirname(save_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        torch.save(self._checkpoint_payload(), save_path)
        print(f"[{self.name}] {label} saved: {save_path}")

    def _load_checkpoint_if_available(self) -> bool:
        path = self.config.model_path
        primary_path = path
        used_fallback = False
        if not path:
            self.checkpoint_error = "empty model path"
            print(f"[{self.name}] ERROR: checkpoint path is empty")
            return False

        print(
            f"[{self.name}] checkpoint probe: pid={os.getpid()} cwd={os.getcwd()} path={path}",
            flush=True,
        )
        for _ in range(3):
            if os.path.isfile(path):
                break
            time.sleep(0.2)
        if not os.path.isfile(path):
            fallback_path = self.best_model_path or self._resolve_best_model_path()
            fallback_path = self._normalize_optional_path(fallback_path)
            if fallback_path and os.path.isfile(fallback_path):
                used_fallback = True
                self.checkpoint_error = (
                    f"primary checkpoint missing: {primary_path} "
                    f"(fallback used: {fallback_path})"
                )
                print(
                    f"[{self.name}] ERROR: primary checkpoint missing: {primary_path} "
                    f"| using fallback: {fallback_path}",
                    flush=True,
                )
                path = fallback_path
                self.config.model_path = fallback_path

        if not os.path.isfile(path):
            parent_dir = os.path.dirname(path) or "."
            parent_exists = os.path.isdir(parent_dir)
            nearby_pt_files: List[str] = []
            if parent_exists:
                try:
                    nearby_pt_files = sorted(
                        [fname for fname in os.listdir(parent_dir) if fname.endswith(".pt")]
                    )[:20]
                except Exception as list_exc:
                    nearby_pt_files = [f"<listdir failed: {list_exc}>"]
            self.checkpoint_error = f"checkpoint not found: {path}"
            print(
                f"[{self.name}] ERROR: checkpoint not found after retries: {path} "
                f"| parent_exists={parent_exists} nearby_pt={nearby_pt_files}",
                flush=True,
            )
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            if not (isinstance(checkpoint, dict) and "actor_state_dict" in checkpoint):
                self.checkpoint_error = f"unsupported checkpoint format: {path}"
                print(f"[{self.name}] ERROR: checkpoint format not recognized: {path}")
                return False

            saved_conf = checkpoint.get("config", {})
            if saved_conf:
                if saved_conf.get("n_rules") != self.config.n_rules:
                    print(
                        f"[{self.name}] WARNING: Rules count mismatch! "
                        f"Saved: {saved_conf.get('n_rules')}, Current: {self.config.n_rules}"
                    )
                if saved_conf.get("mf_type") != self.config.mf_type:
                    print(
                        f"[{self.name}] WARNING: MF type mismatch! "
                        f"Saved: {saved_conf.get('mf_type')}, Current: {self.config.mf_type}"
                    )

            self.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
            self.critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)

            self.actor_target.load_state_dict(
                checkpoint.get("actor_target_state_dict", checkpoint["actor_state_dict"]),
                strict=True,
            )
            self.critic_target.load_state_dict(
                checkpoint.get("critic_target_state_dict", checkpoint["critic_state_dict"]),
                strict=True,
            )

            actor_opt = checkpoint.get("actor_optimizer_state_dict")
            critic_opt = checkpoint.get("critic_optimizer_state_dict")
            if actor_opt:
                self.actor_optimizer.load_state_dict(actor_opt)
            if critic_opt:
                self.critic_optimizer.load_state_dict(critic_opt)

            self.total_steps = int(checkpoint.get("total_steps", 0) or 0)
            self.train_steps = int(checkpoint.get("train_steps", 0) or 0)
            self.games_played = int(checkpoint.get("games_played", 0) or 0)
            self.best_score = float(checkpoint.get("best_score", float("-inf")))
            if not used_fallback:
                self.checkpoint_error = None
            print(
                f"[{self.name}] checkpoint loaded: {path} "
                f"| total_steps={self.total_steps} games={self.games_played}",
                flush=True,
            )
            return True
        except Exception as exc:
            self.checkpoint_error = str(exc)
            print(f"[{self.name}] ERROR: failed to load checkpoint {path}: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)
            return False

    def _reset_episode_state(self) -> None:
        self.last_observation = None
        self.last_action_vector = None
        self.last_command = ActionCommand()
        self.prev_enemies_remaining = None
        self.current_episode_score = 0.0
        self.last_status = None
        self.trace_positions.clear()
        self.trace_hp.clear()
        self.trace_shots.clear()
        self.trace_allies.clear()
        self.trace_enemies.clear()
        self.trace_labels.clear()
        self.pos_history.clear()
        self.trace_actor_raw.clear()
        self.last_actor_raw = None
        self.trace_mock_actions.clear()
        self.last_mock_action = None
        self.trace_damage_taken.clear()
        self.trace_hit_target.clear()
        self.trace_friendly_hit.clear()
        self.trace_should_fire.clear()
        self.episode_reward_total = 0.0
        self.episode_reward_parts = {}
        self.episode_reward_parts_steps = 0
        self.target = None
        self.last_dangerous_terrain_pos = None

    def _record_trace(
        self,
        my_status: Dict[str, Any],
        sensor_data: Dict[str, Any],
        current_obs: Observation,
        action: ActionSpec,
        damage_taken: float,
        hit_target: bool,
        friendly_hit: bool,
    ) -> None:
        pos = my_status.get("position")
        if not pos or "x" not in pos or "y" not in pos:
            raise ValueError(f"Plotting {pos=}")
        x = float(pos.get("x") or 0.0)
        y = float(pos.get("y") or 0.0)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Plotting non-finite position x={x} y={y}")
        hp = float(current_obs.hp_ratio)
        if not math.isfinite(hp):
            raise ValueError(f"Plotting non-finite hp={hp}")
        self.trace_positions.append((x, y))
        self.trace_hp.append(hp)
        self.trace_labels.append(
            f"actor: v={action.move_speed:.2f} h={action.heading_rotation_angle:.2f} "
            f"mock: b={action.barrel_rotation_angle:.2f} f={int(action.should_fire)}"
        )
        self.trace_damage_taken.append(float(max(0.0, damage_taken)))
        self.trace_hit_target.append(1.0 if hit_target else 0.0)
        self.trace_friendly_hit.append(1.0 if friendly_hit else 0.0)
        self.trace_should_fire.append(1.0 if action.should_fire else 0.0)

        seen_tanks = sensor_data.get("seen_tanks", [])
        my_team = my_status.get("_team")
        for tank in seen_tanks:
            tpos = tank.get("position")
            if not tpos or "x" not in tpos or "y" not in tpos:
                raise ValueError(f'Plotting {tpos=}')
            tx = float(tpos.get("x") or 0.0)
            ty = float(tpos.get("y") or 0.0)
            if not (math.isfinite(tx) and math.isfinite(ty)):
                raise ValueError(f"Plotting non-finite tank position x={tx} y={ty} tpos={tpos}")
            if tank.get("team") == my_team:
                self.trace_allies.append((tx, ty))
            else:
                self.trace_enemies.append((tx, ty))

        if action.should_fire:
            heading = float(my_status.get("heading", 0.0) or 0.0)
            barrel = float(my_status.get("barrel_angle", 0.0) or 0.0)
            self.trace_shots.append((x, y, heading + barrel))

    def _save_episode_plot(self) -> None:
        if len(self.trace_positions) == 0:
            raise ValueError("Plotting: no trace positions")

        from matplotlib.collections import LineCollection

        hps = self.trace_hp
        fig, ax = plt.subplots(figsize=(8, 8))
        segments = []
        colors = []
        for i in range(len(self.trace_positions) - 1):
            segments.append([self.trace_positions[i], self.trace_positions[i + 1]])
            colors.append((hps[i] + hps[i + 1]) * 0.5)
        lc = LineCollection(segments, cmap="RdYlGn", linewidths=2.0)
        lc.set_array(np.array(colors, dtype=np.float32))
        ax.add_collection(lc)

        xs = [p[0] for p in self.trace_positions]
        ys = [p[1] for p in self.trace_positions]
        sc_self = ax.scatter(xs, ys, c=hps, cmap="RdYlGn", s=10, alpha=0.7, label="self")

        if self.trace_allies:
            ax.scatter(
                [p[0] for p in self.trace_allies],
                [p[1] for p in self.trace_allies],
                c="blue",
                s=12,
                alpha=0.6,
                label="allies",
            )
            ax_x, ax_y = self.trace_allies[0]
            ax.text(ax_x, ax_y, "A", color="blue", fontsize=7, ha="left", va="bottom")
        if self.trace_enemies:
            ax.scatter(
                [p[0] for p in self.trace_enemies],
                [p[1] for p in self.trace_enemies],
                c="black",
                s=12,
                alpha=0.6,
                label="enemies",
            )
            ex, ey = self.trace_enemies[0]
            ax.text(ex, ey, "E", color="black", fontsize=7, ha="left", va="bottom")
        if self.trace_shots:
            for sx, sy, ang in self.trace_shots:
                dx = math.cos(math.radians(ang)) * 30.0
                dy = math.sin(math.radians(ang)) * 30.0
                ax.arrow(
                    sx,
                    sy,
                    dx,
                    dy,
                    color="#8a7500",
                    width=0.12,
                    head_width=0.6,
                    head_length=1.0,
                    length_includes_head=True,
                )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0.0, MAP_WIDTH)
        ax.set_ylim(0.0, MAP_HEIGHT)
        ax.autoscale(False)
        ax.set_title(f"{self.name} trajectory (game {self.games_played}) | mock barrel+shoot")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right", frameon=True, fontsize=8)
        cbar = fig.colorbar(sc_self, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("hp (normalized)")
        ax.grid(True, alpha=0.3)

        out_dir = os.path.join(current_dir, "training_reports")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"game_{self.games_played}_agent_{self.name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def get_action(
        self,
        current_tick: int,
        my_tank_status: Dict[str, Any],
        sensor_data: Dict[str, Any],
        enemies_remaining: int,
        damage_taken: float = 0.0,
        hit_target: bool = False,
        friendly_hit: bool = False,
    ) -> ActionCommand:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Concurrent get_action call detected")
        try:
            pos = my_tank_status.get("position")
            current_pos = (float(pos["x"]), float(pos["y"]))
            my_team = my_tank_status.get("_team")
            if self.target is None and my_team in (1, 2):
                # self.checkpoints = [(40, 40), (150, 40)] if my_team == 1 else [(150, 40), (40, 40)] # dobra mamy checkpointy - jak ich nie bedzie to sobie znajdzie
                self.checkpoints = [(MAP_WIDTH - pos["x"], pos["y"])]
                self.target = self.checkpoints[0]
            
            seen_tanks = sensor_data.get("seen_tanks", [])
            nearest_enemy = self.encoder._nearest_enemy(
                {"y": current_pos[0], "x": current_pos[1]},
                my_team,
                seen_tanks,
            )
            if nearest_enemy is not None:
                epos = nearest_enemy.get("position")
                if not epos or "x" not in epos or "y" not in epos:
                    raise ValueError(f"Missing enemy position in nearest_enemy: {nearest_enemy}")
                self.target = (float(epos["x"]), float(epos["y"]))
            current_obs, is_enemy_at_ammo_range = self.encoder.encode(
                my_tank_status,
                sensor_data,
                enemies_remaining,
                enemy_target_pos=self.target,
            )

            if current_obs.enemy_visible and nearest_enemy is not None:
                self.checkpoints.insert(0, self.target)
                self.target = (float(epos.get("x", self.target[0])), float(epos.get("y", self.target[1])))
            # elif current_obs.powerup_visible:
            #     powerups = [
            #         (float(p["position"]["x"]), float(p["position"]["y"]))
            #         for p in sensor_data.get("seen_powerups", [])
            #         if p.get("position") and "x" in p["position"] and "y" in p["position"]
            #     ]
            #     keep_current = any(math.hypot(self.target[0] - x, self.target[1] - y) <= 5.0 for x, y in powerups)
            #     if (powerups and not keep_current) or current_obs.danger_ahead:
            #         self.target = min(powerups, key=lambda q: math.hypot(current_pos[0] - q[0], current_pos[1] - q[1]))
            elif np.hypot(current_pos[0] - self.target[0], current_pos[1] - self.target[1]) < 30 and not current_obs.enemy_visible: # dojechal do checkpointa
                if len(self.checkpoints) > 0:
                    self.target = self.checkpoints.pop()
                else:
                    self.target = (np.random.randint(int(MAP_WIDTH)), np.random.randint(int(MAP_HEIGHT)))

            damage_taken_value = self._safe_nonnegative_float(damage_taken, default=0.0)
            hit_target_value = self._safe_bool(hit_target, default=False)
            friendly_hit_value = self._safe_bool(friendly_hit, default=False)
            x_norm = current_pos[0] / MAP_WIDTH
            y_norm = current_pos[1] / MAP_HEIGHT

            recent = self.pos_history[-200:] or [current_pos]
            prev = self.pos_history[-400:-200] or recent
            rcx = sum(p[0] for p in recent) / float(len(recent))
            rcy = sum(p[1] for p in recent) / float(len(recent))
            pcx = sum(p[0] for p in prev) / float(len(prev))
            pcy = sum(p[1] for p in prev) / float(len(prev))

            dx_recent = (current_pos[0] - rcx) / MAP_WIDTH
            dy_recent = (current_pos[1] - rcy) / MAP_HEIGHT
            dx_prev = (current_pos[0] - pcx) / MAP_WIDTH
            dy_prev = (current_pos[1] - pcy) / MAP_HEIGHT
            dx_recent = (dx_recent + 1.0) * 0.5
            dy_recent = (dy_recent + 1.0) * 0.5
            dx_prev = (dx_prev + 1.0) * 0.5
            dy_prev = (dy_prev + 1.0) * 0.5
            current_obs.vector = np.concatenate(
                [current_obs.vector, np.array([x_norm, y_norm, dx_recent, dy_recent, dx_prev, dy_prev], dtype=np.float32)]
            )
            self.pos_history.append(current_pos)

            if (
                self.config.frame_skip > 1
                and self.last_action_vector is not None
                and current_tick % self.config.frame_skip != 0
            ):
                self._record_trace(
                    my_tank_status,
                    sensor_data,
                    current_obs,
                    ActionSpec("ddpg", self.last_command.move_speed, self.last_command.heading_rotation_angle, self.last_command.barrel_rotation_angle, self.last_command.should_fire),
                    damage_taken=damage_taken_value,
                    hit_target=hit_target_value,
                    friendly_hit=friendly_hit_value,
                )
                return self.last_command

            if current_tick > 0:
                if self.last_observation is None or self.last_action_vector is None:
                    if current_tick <= 1:
                        pass
                    elif current_tick > 10:
                        print(
                            f"[{self.name}] WARNING: missing previous state/action at tick={current_tick}"
                        )
                    else:
                        raise ValueError("Missing previous state/action at tick > 0")
                else:
                    reward = self._compute_step_reward(
                        prev_obs=self.last_observation,
                        current_obs=current_obs,
                        action=self.last_command,
                        enemies_remaining=enemies_remaining,
                        current_tick=current_tick,
                        current_pos=current_pos,
                        is_enemy_at_ammo_range=is_enemy_at_ammo_range,
                        # dangerous_terrain_angle_error=dangerous_terrain_angle_error,
                    )
                    self.current_episode_score += reward
                    self.replay.add(
                        state=self.last_observation.vector,
                        action=self.last_action_vector,
                        reward=reward,
                        next_state=current_obs.vector,
                        done=0.0,
                    )
                    self._maybe_train()

            action_vec = self._select_action(current_obs.vector, training=self.training_enabled)
            command = self._to_command(action_vec, my_tank_status, current_obs, sensor_data, current_tick, is_enemy_at_ammo_range)

            if command.should_fire:
                self.last_fire_tick = current_tick

            self.last_observation = current_obs
            self.last_action_vector = action_vec
            self.last_command = command
            self.prev_enemies_remaining = enemies_remaining
            self.total_steps += 1
            self.last_status = my_tank_status
            self._record_trace(
                my_tank_status,
                sensor_data,
                current_obs,
                ActionSpec("ddpg", action_vec[0], action_vec[1], command.barrel_rotation_angle, command.should_fire),
                damage_taken=damage_taken_value,
                hit_target=hit_target_value,
                friendly_hit=friendly_hit_value,
            )
            if self.last_actor_raw is not None:
                self.trace_actor_raw.append(self.last_actor_raw.copy())
            if self.last_mock_action is not None:
                self.trace_mock_actions.append(self.last_mock_action.copy())

            return command
        finally:
            self.lock.release()

    def destroy(self, payload: Optional[Dict[str, Any]] = None) -> None:
        with self.lock:
            print(f"[{self.name}] destroyed")
            if self.last_observation is not None:
                tags = []
                if self.last_observation.danger_ahead:
                    tags.append("rough_terrain")
                if self.last_observation.enemy_visible:
                    tags.append("shot_likely")
                if self.last_observation.ally_fire_risk:
                    tags.append("ally_fire_risk")
                if self.last_observation.obstacle_ahead:
                    tags.append("obstacle_ahead")
                if not tags:
                    tags.append("unknown")
                print(f"[{self.name}] destroy_context: {','.join(tags)}, {self.last_observation}\n\n")
            status = self.last_status or {}
            print(f"[{self.name}] destroy_state: hp={status.get('hp')} shield={status.get('shield')}")
            if payload:
                print(f"[{self.name}] destroy_reason: {payload.get('cause')} damage_events={payload.get('damage_events')}")
            if self.last_observation is not None and self.last_action_vector is not None:
                self.current_episode_score -= 8.0
                self.replay.add(
                    state=self.last_observation.vector,
                    action=self.last_action_vector,
                    reward=-8.0,
                    next_state=self.last_observation.vector,
                    done=1.0,
                )
                for _ in range(3):
                    self._maybe_train()

            self.was_destroyed = True
            # Do not reset trace here; /agent/end is called after destroy
            # and should finalize the episode plot.

    def end(self, damage_dealt: float, tanks_killed: int) -> None:
        with self.lock:
            final_reward = tanks_killed * 4.0 + (damage_dealt / 75.0)
            if not self.was_destroyed:
                final_reward += 1.5

            self.current_episode_score += final_reward
            self.episode_reward_total += final_reward
            if self.last_observation is not None and self.last_action_vector is not None:
                self.replay.add(
                    state=self.last_observation.vector,
                    action=self.last_action_vector,
                    reward=final_reward,
                    next_state=self.last_observation.vector,
                    done=1.0,
                )
                for _ in range(5):
                    self._maybe_train()

            self.last_episode_score = self.current_episode_score
            self.games_played += 1

            print(
                f"[{self.name}] end | games={self.games_played} "
                f"damage={damage_dealt:.1f} kills={tanks_killed} "
                f"replay={len(self.replay)} train_steps={self.train_steps}"
            )

            if self.experiment:
                step = self.train_steps
                self.experiment.log_metric("total_episode_reward", self.current_episode_score, step=step)
                self.experiment.log_metric("damage_dealt", damage_dealt, step=step)
                self.experiment.log_metric("tanks_killed", tanks_killed, step=step)
                self.experiment.log_metric("was_destroyed", int(self.was_destroyed), step=step)
                self.experiment.log_metric("replay_size", len(self.replay), step=step)

            if self.training_enabled and self.games_played % max(1, self.config.save_every_games) == 0:
                episode_score = final_reward + (4.0 * tanks_killed) + (damage_dealt / 40.0)
                if episode_score > self.best_score:
                    self.best_score = episode_score
                    self.save_checkpoint(self.best_model_path, label="best")

                # Save latest after potential best-score update, so
                # best_score persists across process restarts.
                self.save_checkpoint(self.config.model_path, label="latest")

            self._save_episode_plot()
            self._save_actor_raw_plot()
            self._save_combat_feedback_plot()
            steps = max(1, self.episode_reward_parts_steps)
            episode_parts = {k: v / steps for k, v in self.episode_reward_parts.items()}
            episode_parts["total_reward_avg"] = self.episode_reward_total / steps
            all_keys = set(self.reward_parts_history) | set(episode_parts)
            for key in all_keys:
                self.reward_parts_history.setdefault(key, [])
                self.reward_parts_history[key].append(episode_parts.get(key, 0.0))
            self._save_reward_plot()
            
            self.was_destroyed = False
            self._reset_episode_state()

    def _save_reward_plot(self) -> None:
        if not self.reward_parts_history:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = list(plt.cm.tab20.colors)
        for idx, (label, ys) in enumerate(self.reward_parts_history.items()):
            xs = list(range(1, len(ys) + 1))
            ax.plot(
                xs,
                ys,
                label=label,
                marker="o",
                markersize=4,
                linewidth=1.4,
                color=colors[idx % len(colors)],
            )
        ax.set_xlabel("Game")
        ax.set_ylabel("Reward")
        ax.set_title(f"{self.name} reward history")
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=3,
            fontsize=7,
            frameon=False,
            handlelength=1.5,
            columnspacing=0.8,
            borderaxespad=0.2,
        )
        fig.subplots_adjust(bottom=0.25)
        out_dir = os.path.join(current_dir, "training_reports")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"rewards_{self.name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        last_total = None
        if "total_reward_avg" in self.reward_parts_history and self.reward_parts_history["total_reward_avg"]:
            last_total = self.reward_parts_history["total_reward_avg"][-1]
        suffix = f" last_total={last_total:.3f}" if last_total is not None else ""
        print(f"[{self.name}] reward plot saved: {out_path}{suffix}")

    def _save_actor_raw_plot(self) -> None:
        if not self.trace_actor_raw and not self.trace_mock_actions:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        window = 5
        if self.trace_actor_raw:
            data = np.stack(self.trace_actor_raw, axis=0)
            labels = [
                "actor_move_raw",
                "actor_heading_raw",
            ]
            for idx in range(data.shape[1]):
                label = labels[idx] if idx < len(labels) else f"raw_{idx}"
                series = data[:, idx]
                if len(series) >= window:
                    kernel = np.ones(window, dtype=np.float32) / float(window)
                    series = np.convolve(series, kernel, mode="same")
                    label = f"{label}"
                ax.plot(series, label=label)
        if self.trace_mock_actions:
            mock_data = np.stack(self.trace_mock_actions, axis=0)
            mock_labels = [
                "mock barrel norm",
                "mock shoot val",
            ]
            for idx in range(mock_data.shape[1]):
                label = mock_labels[idx] if idx < len(mock_labels) else f"mock_{idx}"
                series = mock_data[:, idx]
                if len(series) >= window:
                    kernel = np.ones(window, dtype=np.float32) / float(window)
                    series = np.convolve(series, kernel, mode="same")
                ax.plot(series, label=f'{label}')
        ax.set_title(f"{self.name} actor raw outputs (game {self.games_played})")
        ax.set_xlabel("step")
        ax.set_ylabel("raw value")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        out_dir = os.path.join(current_dir, "training_reports")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"actor_raw_{self.games_played}_agent_{self.name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[{self.name}] actor raw plot saved: {out_path}")

    def _save_combat_feedback_plot(self) -> None:
        if (
            not self.trace_damage_taken
            and not self.trace_hit_target
            and not self.trace_friendly_hit
            and not self.trace_should_fire
        ):
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        if self.trace_damage_taken:
            damage = np.array(self.trace_damage_taken, dtype=np.float32)
            ax1.plot(damage, label="damage_taken_step", linewidth=1.2)
            ax1.plot(np.cumsum(damage), label="damage_taken_cum", linewidth=1.0)
        ax1.set_ylabel("Damage")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=8)

        should_fire = np.array(self.trace_should_fire, dtype=np.int32) if self.trace_should_fire else np.zeros((0,), dtype=np.int32)
        hit = np.array(self.trace_hit_target, dtype=np.int32) if self.trace_hit_target else np.zeros((0,), dtype=np.int32)
        friendly = np.array(self.trace_friendly_hit, dtype=np.int32) if self.trace_friendly_hit else np.zeros((0,), dtype=np.int32)
        n = min(len(should_fire), len(hit), len(friendly))
        if n == 0:
            cm = np.zeros((2, 3), dtype=np.int32)
        else:
            sf = should_fire[:n]
            ht = hit[:n]
            fh = friendly[:n]
            # Rows: shot command (no/yes), Cols: outcome (miss/no-hit, enemy hit, friendly hit).
            cm = np.array(
                [
                    [
                        int(np.sum((sf == 0) & (ht == 0) & (fh == 0))),
                        int(np.sum((sf == 0) & (ht == 1) & (fh == 0))),
                        int(np.sum((sf == 0) & (fh == 1))),
                    ],
                    [
                        int(np.sum((sf == 1) & (ht == 0) & (fh == 0))),
                        int(np.sum((sf == 1) & (ht == 1) & (fh == 0))),
                        int(np.sum((sf == 1) & (fh == 1))),
                    ],
                ],
                dtype=np.int32,
            )

        im = ax2.imshow(cm, cmap="Blues")
        ax2.set_xticks([0, 1, 2])
        ax2.set_xticklabels(["Miss/NoHit", "Enemy Hit", "Friendly Hit"])
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["No Fire", "Fire"])
        ax2.set_xlabel("Actual Outcome")
        ax2.set_ylabel("Shot Command")
        for i in range(2):
            for j in range(3):
                ax2.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        fired_enemy_hits = int(cm[1, 1])
        fired_friendly_hits = int(cm[1, 2])
        fired_misses = int(cm[1, 0])
        ax2.set_title(
            "Shooting Confusion Matrix | "
            f"enemy_hit={fired_enemy_hits}, friendly_hit={fired_friendly_hits}, miss={fired_misses}"
        )
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

        fig.suptitle(f"{self.name} combat feedback (game {self.games_played})")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        out_dir = os.path.join(current_dir, "training_reports")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"combat_feedback_{self.games_played}_agent_{self.name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[{self.name}] combat feedback plot saved: {out_path}")

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "training_enabled": self.training_enabled,
            "map_name": self.map_name,
            "steps": self.total_steps,
            "train_steps": self.train_steps,
            "games_played": self.games_played,
            "replay_size": len(self.replay),
            "last_loss": self.last_loss,
            "model_path": self.config.model_path,
            "best_model_path": self.best_model_path,
            "best_score": self.best_score,
            "last_episode_score": self.last_episode_score,
        }


app = FastAPI(
    title="Fuzzy DQN Agent",
    description="ANFIS-based DQN agent for tank battles",
    version="2.0.0",
)

agent: Optional[FuzzyDQNAgent] = None


def _get_agent() -> FuzzyDQNAgent:
    global agent
    if agent is None:
        agent = FuzzyDQNAgent(
            name="FuzzyDQN",
            config=AgentConfig(model_path=""),
            training_enabled=False,
            load_checkpoint=False,
        )
    return agent


@app.get("/")
async def root() -> Dict[str, Any]:
    print('Root communicating')
    return _get_agent().status()


@app.post("/agent/action", response_model=ActionCommand)
async def get_action(payload: Dict[str, Any] = Body(...)) -> ActionCommand:
    damage_taken = FuzzyDQNAgent._safe_nonnegative_float(payload.get("damage_taken", 0.0), default=0.0)
    hit_target = FuzzyDQNAgent._safe_bool(payload.get("hit_target", False), default=False)
    friendly_hit = FuzzyDQNAgent._safe_bool(payload.get("friendly_hit", False), default=False)
    result = _get_agent().get_action(
        current_tick=int(payload.get("current_tick", 0) or 0),
        my_tank_status=payload.get("my_tank_status", {}),
        sensor_data=payload.get("sensor_data", {}),
        enemies_remaining=int(payload.get("enemies_remaining", 0) or 0),
        damage_taken=damage_taken,
        hit_target=hit_target,
        friendly_hit=friendly_hit,
    )
    # print(f"[ACTION tick={payload.get('current_tick')}]\
    #         move={result.move_speed:.3f}\
    #         heading={result.heading_rotation_angle:.3f}\
    #         fire={result.should_fire}", flush=True
    #     )
    return result


@app.post("/agent/context")
async def update_context(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    print(f'Context: {payload=}')
    map_name = str(payload.get("map_name", "") or "")
    agent_obj = _get_agent()
    with agent_obj.lock:
        agent_obj.map_name = map_name
    return {"ok": True, "map_name": map_name}


@app.post("/agent/destroy", status_code=204, response_model=None)
async def destroy(payload: Dict[str, Any] = Body(None)) -> None:
    # print(f'Destroy: {destroy=}')
    _get_agent().destroy(payload)


@app.post("/agent/end", status_code=204, response_model=None)
async def end(payload: Dict[str, Any] = Body(...)) -> None:
    # print(f'End: {payload=}')
    _get_agent().end(
        damage_dealt=float(payload.get("damage_dealt", 0.0) or 0.0),
        tanks_killed=int(payload.get("tanks_killed", 0) or 0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fuzzy DQN tank agent")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--train", action="store_true", help="Enable online learning")

    parser.add_argument("--model-path", type=str, default=os.path.join(current_dir, 'dependencyAgent10/fuzzy_dqn_model_best.pt'))
    parser.add_argument("--best-model-path", type=str, default=os.path.join(current_dir, 'dependencyAgent10/fuzzy_dqn_model_best.pt'))
    parser.add_argument("--rules", type=int, default=32)
    parser.add_argument("--mf-type", choices=["gaussian", "bell", "triangular"], default="gaussian")
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-every-games", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-every", type=int, default=2)
    parser.add_argument("--target-sync-every", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--action-noise-start", type=float, default=0.3)
    parser.add_argument("--action-noise-end", type=float, default=0.05)
    parser.add_argument("--action-noise-decay-steps", type=int, default=50_000)
    parser.add_argument("--no-load", action="store_true", help="Do not load checkpoints on startup.")
    parser.add_argument("--map-name", type=str, default="", help="Current map identifier (for context/debugging).")

    return parser.parse_args()


if __name__ == "__main__":
    import traceback as _traceback
    print("[AGENT STARTUP] Agent10.py __main__ reached", flush=True)
    try:
        args = parse_args()
        print(f"[AGENT STARTUP] args parsed: port={args.port} name={args.name}", flush=True)

        config = AgentConfig(
            n_rules=max(4, int(args.rules)),
            mf_type=args.mf_type,
            frame_skip=max(1, int(args.frame_skip)),
            model_path=args.model_path,
            best_model_path=args.best_model_path,
            seed=int(args.seed),
            save_every_games=max(1, int(args.save_every_games)),
            warmup_steps=max(0, int(args.warmup_steps)),
            batch_size=max(16, int(args.batch_size)),
            train_every=max(1, int(args.train_every)),
            target_sync_every=max(1, int(args.target_sync_every)),
            gamma=float(args.gamma),
            actor_lr=float(args.actor_lr),
            critic_lr=float(args.critic_lr),
            tau=float(args.tau),
            action_noise_start=float(args.action_noise_start),
            action_noise_end=float(args.action_noise_end),
            action_noise_decay_steps=max(1, int(args.action_noise_decay_steps)),
            map_name=str(args.map_name or ""),
        )

        agent_name = args.name or f"FuzzyDQN_{args.port}"
        print(f"[AGENT STARTUP] Creating FuzzyDQNAgent: {agent_name}", flush=True)

        # Replace default global agent with runtime configuration.
        agent = FuzzyDQNAgent(
            name=agent_name,
            config=config,
            training_enabled=bool(args.train),
            load_checkpoint=not bool(args.no_load),
        )

        print(
            f"[AGENT STARTUP] Starting {agent_name} on {args.host}:{args.port} "
            f"| train={args.train} | model={config.model_path}",
            flush=True,
        )
        uvicorn.run(app, host=args.host, port=args.port, access_log=False)
    except Exception as _e:
        print(f"[AGENT STARTUP ERROR] {_e}", flush=True)
        _traceback.print_exc()
        raise
