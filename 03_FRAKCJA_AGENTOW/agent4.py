from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Body, FastAPI
from pydantic import BaseModel
import uvicorn

from agent_core.checkpoints import STATIC_CORRIDOR_CHECKPOINTS, lane_offset_checkpoint
from agent_core.driver import MotionDriver
from agent_core.fuzzy_turret import FuzzyTurretController
from agent_core.geometry import (
    euclidean_distance,
    heading_to_angle_deg,
    normalize_angle_diff,
    to_xy,
)
from agent_core.planner import AStarPlanner
from agent_core.world_model import WorldModel


class ActionCommand(BaseModel):
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: Optional[str] = None
    should_fire: bool = False


class TankAgent:
    def __init__(self, name: str = "TankAgent", enable_autonomous: bool = True):
        self.name = name
        self.is_destroyed = False
        self.enable_autonomous = enable_autonomous

        self.checkpoints: Optional[List[Tuple[float, float]]] = None
        self.checkpoint_idx: int = 0
        self.tank_id: str = "default"
        self.team: Optional[int] = None
        self.arrival_radius: float = 3.0
        self.turret: Optional[FuzzyTurretController] = None

        self.mode: str = "checkpoint"
        self.world_model: Optional[WorldModel] = None
        self.planner: Optional[AStarPlanner] = None
        self.driver: Optional[MotionDriver] = None
        self.path: List[Tuple[int, int]] = []
        self.replan_cooldown: int = 0

        status = "with autonomous mode" if enable_autonomous else "checkpoint-only"
        print(f"[{self.name}] online ({status})")

    def _init_checkpoints(
        self, my_tank_status: Dict[str, Any], x: float, y: float
    ) -> None:
        self.team = my_tank_status.get("_team")
        if self.team is None:
            self.team = 2 if x > 100.0 else 1

        if self.team == 2:
            self.checkpoints = list(reversed(STATIC_CORRIDOR_CHECKPOINTS))
        else:
            self.checkpoints = list(STATIC_CORRIDOR_CHECKPOINTS)

        self.tank_id = str(my_tank_status.get("_id", "default"))

        closest_idx = 0
        closest_dist = float("inf")
        for idx, cp in enumerate(self.checkpoints):
            tx, ty = lane_offset_checkpoint(self.tank_id, cp)
            dist = euclidean_distance(x, y, tx, ty)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = idx

        self.checkpoint_idx = closest_idx
        print(
            f"[{self.name}] team={self.team} start_cp={closest_idx + 1}/{len(self.checkpoints)} dist={closest_dist:.1f}"
        )

        if self.enable_autonomous:
            self.world_model = WorldModel(map_size=200)
            self.planner = AStarPlanner(self.world_model)
            self.driver = MotionDriver(self.world_model)
            print(f"[{self.name}] autonomous pathfinding initialized")

    def _current_target(self) -> Tuple[float, float]:
        assert self.checkpoints is not None
        cp = self.checkpoints[self.checkpoint_idx]
        return lane_offset_checkpoint(self.tank_id, cp)

    def _check_autonomous_threshold(self, y: float) -> bool:
        if not self.enable_autonomous or self.mode == "autonomous":
            return False

        if self.team == 1 and y <= 80.0:
            return True
        elif self.team == 2 and y >= 120.0:
            return True
        return False

    def _update_world_model(
        self, x: float, y: float, sensor_data: Dict[str, Any]
    ) -> None:
        if not self.world_model:
            return

        self.world_model.decay()

        for obstacle in sensor_data.get("seen_obstacles", []):
            ox = float(obstacle.get("position", {}).get("x", 0))
            oy = float(obstacle.get("position", {}).get("y", 0))
            cell = self.world_model.to_cell(ox, oy)
            self.world_model.get_state(cell).blocked += 1.5

        for terrain in sensor_data.get("seen_terrains", []):
            dmg = terrain.get("dmg", 0)
            if dmg > 0:
                tx = float(terrain.get("position", {}).get("x", 0))
                ty = float(terrain.get("position", {}).get("y", 0))
                cell = self.world_model.to_cell(tx, ty)
                danger_score = 3.0 if dmg >= 2 else 1.5
                self.world_model.get_state(cell).danger += danger_score

        for tank in sensor_data.get("seen_tanks", []):
            tank_team = tank.get("team")
            tank_x = float(tank.get("position", {}).get("x", 0))
            tank_y = float(tank.get("position", {}).get("y", 0))
            cell = self.world_model.to_cell(tank_x, tank_y)

            if tank_team == self.team:
                self.world_model.mark_ally_occupancy(cell, ttl=10)
            else:
                self.world_model.mark_enemy_occupancy(cell, ttl=10)

        for powerup in sensor_data.get("seen_powerups", []):
            px = float(powerup.get("position", {}).get("x", 0))
            py = float(powerup.get("position", {}).get("y", 0))
            cell = self.world_model.to_cell(px, py)
            self.world_model.powerup_cells.add(cell)

    def _select_autonomous_goal(
        self, x: float, y: float, sensor_data: Dict[str, Any]
    ) -> Optional[Tuple[float, float]]:
        enemies = [
            t for t in sensor_data.get("seen_tanks", []) if t.get("team") != self.team
        ]

        if enemies:
            closest_enemy = None
            closest_dist = float("inf")
            for enemy in enemies:
                ex = float(enemy.get("position", {}).get("x", 0))
                ey = float(enemy.get("position", {}).get("y", 0))
                dist = euclidean_distance(x, y, ex, ey)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_enemy = (ex, ey)
            return closest_enemy

        powerups = sensor_data.get("seen_powerups", [])
        if powerups:
            closest_powerup = None
            closest_dist = float("inf")
            for powerup in powerups:
                px = float(powerup.get("position", {}).get("x", 0))
                py = float(powerup.get("position", {}).get("y", 0))
                dist = euclidean_distance(x, y, px, py)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_powerup = (px, py)
            return closest_powerup

        if self.team == 1:
            return (150.0, y)
        else:
            return (50.0, y)

    def _compute_path(self, x: float, y: float, goal: Tuple[float, float]) -> bool:
        if not self.world_model or not self.planner:
            return False

        my_cell = self.world_model.to_cell(x, y)
        goal_cell = self.world_model.to_cell(goal[0], goal[1])

        try:
            self.path = self.planner.build_path(my_cell, goal_cell, radius=18)
            if not self.path:
                self.world_model.mark_dead_end(my_cell, ttl=30)
                return False

            return True
        except Exception as e:
            print(f"[{self.name}] pathfinding error: {e}")
            return False

    def get_action(
        self,
        current_tick: int,
        my_tank_status: Dict[str, Any],
        sensor_data: Dict[str, Any],
        enemies_remaining: int,
    ) -> ActionCommand:
        x, y = to_xy(my_tank_status.get("position", {}))
        heading = float(my_tank_status.get("heading", 0.0) or 0.0)
        top_speed = float(my_tank_status.get("_top_speed", 3.0) or 3.0)
        max_heading = float(my_tank_status.get("_heading_spin_rate", 30.0) or 30.0)
        barrel_angle = float(my_tank_status.get("barrel_angle", 0.0) or 0.0)
        max_barrel = float(my_tank_status.get("_barrel_spin_rate", 30.0) or 30.0)
        vision_range = float(my_tank_status.get("_vision_range", 70.0) or 70.0)

        if self.checkpoints is None:
            self._init_checkpoints(my_tank_status, x, y)

        if self.turret is None:
            self.turret = FuzzyTurretController(
                max_barrel_spin_rate=max_barrel,
                vision_range=vision_range,
            )

        assert self.checkpoints is not None

        if self._check_autonomous_threshold(y):
            self.mode = "autonomous"
            print(f"[{self.name}] switching to AUTONOMOUS mode at y={y:.1f}")

        if self.mode == "autonomous":
            self._update_world_model(x, y, sensor_data)

        seen = sensor_data.get("seen_tanks", [])
        enemies = [t for t in seen if t.get("team") != self.team]

        if self.mode == "checkpoint":
            while self.checkpoint_idx < len(self.checkpoints) - 1:
                tx, ty = self._current_target()
                if euclidean_distance(x, y, tx, ty) < self.arrival_radius:
                    self.checkpoint_idx += 1
                else:
                    break

            tx, ty = self._current_target()
            desired = heading_to_angle_deg(x, y, tx, ty)
            diff = normalize_angle_diff(desired, heading)
            turn = max(-max_heading, min(diff, max_heading))

            if abs(diff) > 45.0:
                speed = top_speed * 0.3
            elif abs(diff) > 20.0:
                speed = top_speed * 0.6
            else:
                speed = top_speed
        else:
            self.replan_cooldown -= 1

            if self.replan_cooldown <= 0 or not self.path:
                goal = self._select_autonomous_goal(x, y, sensor_data)
                if goal:
                    if self._compute_path(x, y, goal):
                        if self.driver:
                            self.driver.path = self.path
                        self.replan_cooldown = 30
                else:
                    self.path = []

            if self.path and self.driver:
                turn, speed = self.driver.drive_path(x, y, heading, top_speed)
            else:
                turn = 0.0
                speed = top_speed * 0.5

            if self.path:
                next_cell = (
                    self.path[0] if self.path else self.world_model.to_cell(x, y)
                )
                tx, ty = self.world_model.from_cell(next_cell[0], next_cell[1])
            else:
                tx, ty = x, y

        ammo_stocks: Dict[str, int] = {}
        raw_ammo = my_tank_status.get("ammo", {})
        if isinstance(raw_ammo, dict):
            for key, val in raw_ammo.items():
                if isinstance(val, dict):
                    ammo_stocks[key] = int(val.get("count", 0) or 0)
                else:
                    ammo_stocks[key] = int(val or 0)
        current_ammo = str(my_tank_status.get("ammo_loaded", "") or "").upper() or None

        seen_obstacles = sensor_data.get("seen_obstacles", [])
        barrel_rotation, should_fire, ammo_to_load = self.turret.update(
            my_x=x,
            my_y=y,
            my_heading=heading,
            current_barrel_angle=barrel_angle,
            seen_tanks=enemies,
            max_barrel_rotation=max_barrel,
            ammo_stocks=ammo_stocks,
            current_ammo=current_ammo,
            seen_obstacles=seen_obstacles,
        )

        if current_tick % 60 == 0:
            dist = euclidean_distance(x, y, tx, ty)
            if self.mode == "checkpoint":
                print(
                    f"[{self.name}] tick={current_tick} mode=CP cp={self.checkpoint_idx + 1}/{len(self.checkpoints)} "
                    f"dist={dist:.1f} speed={speed:.2f} turn={turn:.1f} enemies={len(enemies)} ammo={current_ammo}"
                )
            else:
                path_len = len(self.path)
                print(
                    f"[{self.name}] tick={current_tick} mode=AUTO path_len={path_len} "
                    f"dist={dist:.1f} speed={speed:.2f} turn={turn:.1f} enemies={len(enemies)} ammo={current_ammo}"
                )

        speed = top_speed
        return ActionCommand(
            heading_rotation_angle=turn,
            move_speed=speed,
            barrel_rotation_angle=barrel_rotation,
            should_fire=should_fire,
            ammo_to_load=ammo_to_load,
        )

    def destroy(self):
        self.is_destroyed = True
        print(f"[{self.name}] destroyed")

    def end(self, damage_dealt: float, tanks_killed: int):
        print(f"[{self.name}] end damage={damage_dealt} kills={tanks_killed}")


app = FastAPI(title="Simple Driver Agent", version="1.0.0")
agent = TankAgent()


@app.get("/")
async def root():
    return {
        "message": f"Agent {agent.name} is running",
        "destroyed": agent.is_destroyed,
    }


@app.post("/agent/action", response_model=ActionCommand)
async def get_action(payload: Dict[str, Any] = Body(...)):
    return agent.get_action(
        current_tick=payload.get("current_tick", 0),
        my_tank_status=payload.get("my_tank_status", {}),
        sensor_data=payload.get("sensor_data", {}),
        enemies_remaining=payload.get("enemies_remaining", 0),
    )


@app.post("/agent/destroy", status_code=204)
async def destroy():
    agent.destroy()


@app.post("/agent/end", status_code=204)
async def end(payload: Dict[str, Any] = Body(...)):
    agent.end(
        damage_dealt=payload.get("damage_dealt", 0.0),
        tanks_killed=payload.get("tanks_killed", 0),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simple checkpoint driver agent")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Enable autonomous A* pathfinding after crossing threshold",
    )
    args = parser.parse_args()

    agent = TankAgent(enable_autonomous=args.autonomous)

    if args.name:
        agent.name = args.name
    else:
        agent.name = f"SimpleDriver_{args.port}"

    print(f"Starting {agent.name} on {args.host}:{args.port}")
    uvicorn.run(
        app, host=args.host, port=args.port, log_level="warning", access_log=False
    )
