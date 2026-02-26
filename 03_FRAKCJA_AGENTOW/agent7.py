from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict

from fastapi import Body, FastAPI
from pydantic import BaseModel
import uvicorn

from utils.astar_navigator import AStar_Navigator
from utils.controller_types import AgentAction
from utils.payload_adapter import build_controller_inputs, to_engine_action
from utils.runtime_map import RuntimeMapMemory
from utils.tsk_motion_controller import TSK_MotionController
from utils.tsk_weapon_controller import TSK_WeaponController


class ActionCommand(BaseModel):
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: str | None = None
    should_fire: bool = False


class TankApiAgent:
    """
    Port/API agent wrapper.
    Internal architecture:
    - AStar_Navigator (path planning)
    - TSK_MotionController (hull steering + speed)
    - Turret NN runtime for barrel/ammo/fire (checkpoint in utils)
      with TSK fallback when runtime cannot be loaded.
    """

    def __init__(self, name: str = "AStarTSK_Agent") -> None:
        self.name = str(name)
        self.base_dir = Path(__file__).resolve().parent
        self.weights_path = self.base_dir / "utils" / "tsk_weights.json"
        self.weights = self._load_weights(self.weights_path)

        map_cfg = dict(self.weights.get("map", {}))
        self.map_memory = RuntimeMapMemory(
            map_w=float(map_cfg.get("width", 200.0)),
            map_h=float(map_cfg.get("height", 200.0)),
            obstacle_ttl=int(map_cfg.get("obstacle_memory_ttl", 1200)),
            terrain_ttl=int(map_cfg.get("terrain_memory_ttl", 900)),
            obstacle_default_size=float(map_cfg.get("obstacle_default_size", 9.0)),
            obstacle_size_by_type=dict(map_cfg.get("obstacle_size_by_type", {})),
            terrain_patch_size=float(map_cfg.get("terrain_patch_size", 10.0)),
        )

        self.navigator = AStar_Navigator(
            map_info=self.map_memory,
            cell_size=float(map_cfg.get("cell_size", 10.0)),
            enemy_memory_ttl=30,
            powerup_memory_ttl=1000,
        )
        self.motion_controller = TSK_MotionController(weights=self.weights)
        self.weapon_controller = TSK_WeaponController(weights=self.weights)

        self.weapon_backend = "TSK"
        self.turret_runtime = None
        self.turret_checkpoint_path: str | None = None
        self._turret_init_error: str | None = None
        self._try_init_turret_runtime()

        self._last_map_version = -1
        self._prev_engine_cmd = {
            "move_speed": 0.0,
            "heading_delta": 0.0,
            "barrel_delta": 0.0,
        }
        self.is_destroyed = False

    @staticmethod
    def _load_weights(path: Path) -> Dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
                if isinstance(payload, dict):
                    return payload
        except Exception:
            pass
        return {}

    def _try_init_turret_runtime(self) -> None:
        turret_cfg = dict(self.weights.get("turret_nn", {}))
        if not bool(turret_cfg.get("enabled", True)):
            self.weapon_backend = "TSK"
            return

        checkpoint_name = str(turret_cfg.get("checkpoint", "turret_aim_only.pt"))
        checkpoint_path = self.base_dir / "utils" / checkpoint_name
        self.turret_checkpoint_path = str(checkpoint_path)

        try:
            rt_mod = importlib.import_module("utils.turret_nn_runtime")
            TurretRuntimeAgent = getattr(rt_mod, "TurretRuntimeAgent")
            TurretRuntimeConfig = getattr(rt_mod, "TurretRuntimeConfig")

            cfg = TurretRuntimeConfig(
                tank_type=str(turret_cfg.get("tank_type", "ANY")),
                device=str(turret_cfg.get("device", "auto")),
                seed=int(turret_cfg.get("seed", 42)),
                epsilon=float(turret_cfg.get("epsilon", 0.0)),
                scan_action=float(turret_cfg.get("scan_action", 0.35)),
                scan_flip_ticks=int(turret_cfg.get("scan_flip_ticks", 120)),
                close_distance=float(turret_cfg.get("close_distance", 22.0)),
                close_track_error_deg=float(turret_cfg.get("close_track_error_deg", 2.0)),
                close_track_action=float(turret_cfg.get("close_track_action", 0.35)),
                fire_error_deg=float(turret_cfg.get("fire_error_deg", 4.8)),
                fire_conf_threshold=float(turret_cfg.get("fire_conf_threshold", 0.48)),
                enable_fire=bool(turret_cfg.get("enable_fire", True)),
            )
            self.turret_runtime = TurretRuntimeAgent(checkpoint=str(checkpoint_path), config=cfg)
            self.weapon_backend = "TURRET_NN"
            self._turret_init_error = None
        except Exception as exc:
            self.turret_runtime = None
            self.weapon_backend = "TSK_FALLBACK"
            self._turret_init_error = str(exc)

    def _refresh_planner_if_needed(self) -> None:
        if self._last_map_version != self.map_memory.version:
            self.navigator.refresh_map(self.map_memory)
            self._last_map_version = self.map_memory.version

    def get_action(
        self,
        current_tick: int,
        my_tank_status: Dict[str, Any],
        sensor_data: Dict[str, Any],
        enemies_remaining: int,
    ) -> Dict[str, Any]:
        del current_tick, enemies_remaining

        self.is_destroyed = False
        x = float((my_tank_status or {}).get("position", {}).get("x", 0.0))
        y = float((my_tank_status or {}).get("position", {}).get("y", 0.0))

        self.map_memory.update(sensor_data or {}, my_position=(x, y))
        self._refresh_planner_if_needed()

        state, sensor_snapshot, tank_type = build_controller_inputs(
            {
                "my_tank_status": my_tank_status,
                "sensor_data": sensor_data,
            },
            self.weights,
        )

        nav = self.navigator.navigate(state=state, sensor_snapshot=sensor_snapshot)
        motion = self.motion_controller.decide(
            state=state,
            sensor_snapshot=sensor_snapshot,
            nav=nav,
        )
        tsk_weapon = self.weapon_controller.decide(
            state=state,
            sensor_snapshot=sensor_snapshot,
            nav=nav,
        )

        barrel_cmd = float(tsk_weapon.barrel_rotation_angle)
        ammo_to_load = str(tsk_weapon.ammo_to_load)
        should_fire = bool(tsk_weapon.should_fire)

        if self.turret_runtime is not None:
            try:
                nn_out = self.turret_runtime.act_from_state(state=state, sensor_snapshot=sensor_snapshot)
                barrel_cmd = float(nn_out.get("barrel_rotation_angle", barrel_cmd))
                ammo_to_load = str(nn_out.get("ammo_to_load", ammo_to_load))
                should_fire = bool(nn_out.get("should_fire", should_fire))
            except Exception as exc:
                self._turret_init_error = str(exc)

        normalized_action = AgentAction(
            move_speed=float(motion.move_speed),
            heading_rotation_angle=float(motion.heading_rotation_angle),
            barrel_rotation_angle=float(barrel_cmd),
            ammo_to_load=str(ammo_to_load),
            should_fire=bool(should_fire),
        )

        engine_action, self._prev_engine_cmd = to_engine_action(
            action=normalized_action,
            state=state,
            tank_type=tank_type,
            weights=self.weights,
            prev=self._prev_engine_cmd,
        )

        return engine_action

    def destroy(self) -> None:
        self.is_destroyed = True
        self._prev_engine_cmd = {
            "move_speed": 0.0,
            "heading_delta": 0.0,
            "barrel_delta": 0.0,
        }
        if self.turret_runtime is not None:
            try:
                self.turret_runtime.reset_runtime()
            except Exception:
                pass

    def end(self, final_score: Dict[str, Any] | None = None) -> None:
        del final_score


app = FastAPI(
    title="AStar + TSK Tank Agent",
    description="HTTP agent server compatible with Engine /agent/* endpoints",
    version="1.0.0",
)

agent = TankApiAgent()


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "message": f"{agent.name} is running",
        "destroyed": bool(agent.is_destroyed),
        "weights": str(agent.weights_path),
        "weapon_backend": str(agent.weapon_backend),
        "turret_checkpoint": str(agent.turret_checkpoint_path),
        "turret_error": agent._turret_init_error,
    }


@app.post("/agent/action", response_model=ActionCommand)
async def action_endpoint(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return agent.get_action(
        current_tick=int(payload.get("current_tick", 0)),
        my_tank_status=dict(payload.get("my_tank_status", {}) or {}),
        sensor_data=dict(payload.get("sensor_data", {}) or {}),
        enemies_remaining=int(payload.get("enemies_remaining", 0)),
    )


@app.post("/agent/destroy", status_code=204)
async def destroy_endpoint() -> None:
    agent.destroy()


@app.post("/agent/end", status_code=204)
async def end_endpoint(payload: Dict[str, Any] = Body(...)) -> None:
    agent.end(final_score=payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AStar+TSK agent server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8001, help="Port number")
    parser.add_argument("--name", type=str, default="AStarTSK_Agent", help="Agent name")
    args = parser.parse_args()

    agent.name = str(args.name)
    uvicorn.run(app, host=args.host, port=int(args.port))
