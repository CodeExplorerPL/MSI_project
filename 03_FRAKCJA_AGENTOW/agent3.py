import copy
import json
import pickle
import argparse
import sys
import os
import numpy as np
from sklearn.cluster import KMeans
from agent3_agent_memory import Memory
from agent3_agent_state import prepare_agent_state_from_data, calculate_distance, calculate_angle_to_target, \
    normalize_angle
from agent3_field_state import check_possible_fields, prepare_field_state_from_data
from agent3_stage4 import create_fuzzy_model, get_alpha
from agent3_enemies_state import prepare_visible_tanks_states, decide_if_should_shot
from agent3_stage3 import get_path, make_search
from agent3_stage6 import create_fuzzy_model_stage6, get_preferable_ammo

current_dir = os.path.dirname(os.path.abspath(__file__))
controller_dir = os.path.join(os.path.dirname(current_dir), 'FRAKCJA_SILNIKA', 'controller')
sys.path.insert(0, controller_dir)

parent_dir = os.path.join(os.path.dirname(current_dir), 'FRAKCJA_SILNIKA')
sys.path.insert(0, parent_dir)

from typing import Dict, Any
from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn
from pathlib import Path

MOVEMENT_CHECK_TICKS = 250
BARREL_CHECK_TICKS = 30
ENEMY_SCAN_TICKS = 100
BASE_DIR = Path(__file__).resolve().parent


class ActionCommand(BaseModel):
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: str | None = None
    should_fire: bool = False


class Agent_v3:
    def __init__(self, name='Group3_Agent', state_preparing=True):
        self.name = name
        self.memory = Memory()
        self.is_destroyed = False
        self.state_preparing = state_preparing
        self.stage2_data: list[dict] = []
        self.reward_data: list[dict] = []
        print(f"Agent initialized")

        with open(BASE_DIR / "group3_states_km.pkl", 'rb') as f:
            self.km: KMeans = pickle.load(f)

        with open(BASE_DIR / "group3_states_to_field_features_map.json", 'r') as f:
            self.cluster_to_field_map = json.load(f)

        self.a_star_risk_factor_fuzzy_model = create_fuzzy_model()

        self.enemies_features_weights = np.array([0.52, 0.78, 0.25, 0.08, 0.67])
        self.enemies_threshold = 0.13
        self.max_dist ={
        'LIGHT': 42.8,
        'HEAVY': 29.1,
        'LONG_DISTANCE': 78.6
    }

        self.target_tank_tick = -1
        self.target_tank_data = None

        self.destination_route = None

        self.ammo_type_fuzzy_model = create_fuzzy_model_stage6()
        self.barrel_rotation_speed = 5

    def add_to_memory(self, payload):
        self.memory.refresh_memory(payload)

    @staticmethod
    def get_heading_direction(position, destination):
        return calculate_angle_to_target(position, destination)

    def get_action(
            self,
            payload: Dict[str, Any],
            current_tick: int,
            my_tank_status: Dict[str, Any],
            sensor_data: Dict[str, Any],
            enemies_remaining: int
    ) -> ActionCommand:
        if current_tick == 1:
            self.memory.start_ammo = sum([ammo['count'] for ammo in my_tank_status['ammo'].values()])
            self.memory.start_enemies = enemies_remaining
        current_x = my_tank_status['position']['x']
        current_y = my_tank_status['position']['y']

        if self.destination_route is not None and calculate_distance((current_x, current_y), self.destination_route[0]) < 2:
            self.destination_route.pop(0)
            if len(self.destination_route) == 0:
                self.destination_route = None

        if current_tick == 1 or current_tick % MOVEMENT_CHECK_TICKS == 0:
            state = prepare_agent_state_from_data(payload, self.memory)
            state_np = np.array(state).reshape(1, -1)
            cluster_no = str(self.km.predict(state_np)[0])
            field_weights = np.array(self.cluster_to_field_map[cluster_no])

            possible_fields = check_possible_fields(payload, self.memory)
            possible_fields_values = {}
            for field_position, field_state in possible_fields.items():
                possible_fields_values[field_position] = field_state @ field_weights

            risk_factor_alpha = get_alpha(self.a_star_risk_factor_fuzzy_model, state, payload, self.memory)

            if self.destination_route is not None:
                best_field_value = max(list(possible_fields_values.values()))
                current_destination_value = prepare_field_state_from_data(payload, self.destination_route[-1], self.memory)
                if current_destination_value is None:
                    self.destination_route = None
                else:
                    current_destination_value = np.array(current_destination_value) @ field_weights
                    if current_destination_value > best_field_value or abs(current_destination_value - best_field_value) < 0.1:
                        route, cost = make_search((current_x, current_y), self.destination_route[-1], risk_factor_alpha,
                                    self.memory, current_tick, my_tank_status['_team'], my_tank_status['_id'])
                        if cost is None or cost > 1000:
                            self.destination_route = None
                        else:
                            self.destination_route = route
                    else:
                        self.destination_route = None

            if self.destination_route is None:
                best_position = max(possible_fields_values, key=possible_fields_values.get)
                if calculate_distance((current_x, current_y), best_position) > 5:
                    self.destination_route = get_path((current_x, current_y), possible_fields_values, risk_factor_alpha, self.memory,
                                                     current_tick, my_tank_status['_team'], my_tank_status['_id'])

        if current_tick == 1 or current_tick % BARREL_CHECK_TICKS == 0:
            tanks_features = prepare_visible_tanks_states(payload, (current_x, current_y))
            best_tank_id = None
            if len(tanks_features) > 0:
                max_score = 0
                for tank_id, tank_features in tanks_features.items():
                    tank_score = tank_features @ self.enemies_features_weights
                    if tank_score > max_score and tank_score > self.enemies_threshold:
                        max_score = tank_score
                        best_tank_id = tank_id

            if best_tank_id is not None and (self.target_tank_data is None or self.target_tank_data['id'] != best_tank_id):
                for tank in sensor_data['seen_tanks']:
                    if tank['id'] == best_tank_id:
                        self.target_tank_data = copy.deepcopy(tank)
                        break
                self.target_tank_tick = current_tick

        if self.target_tank_data is not None:
            for tank in sensor_data['seen_tanks']:
                if tank['id'] == self.target_tank_data['id']:
                    self.target_tank_data = copy.deepcopy(tank)
                    self.target_tank_tick = current_tick
                    break
            if current_tick - self.target_tank_tick > ENEMY_SCAN_TICKS:
                self.target_tank_tick = None
                self.target_tank_data = None

        if self.destination_route is not None:
            heading_direction = (self.get_heading_direction(my_tank_status['position'],
                                           {'x': self.destination_route[0][0], 'y': self.destination_route[0][1]}))
            movement_speed = 100
        else:
            heading_direction = my_tank_status['heading']
            movement_speed = 0

        if self.target_tank_data is not None:
            ammo_to_load = get_preferable_ammo(self.ammo_type_fuzzy_model, (current_x, current_y),
                                               self.target_tank_data, my_tank_status['ammo'])
            barrel_rotation = self.get_heading_direction({'x': current_x, 'y': current_y},
                                                         self.target_tank_data['position'])

            goal_barrel_rotation = normalize_angle(barrel_rotation - my_tank_status['heading'])
            barrel_rotation_angle = goal_barrel_rotation - my_tank_status['barrel_angle']
            should_fire = decide_if_should_shot(payload, self.target_tank_data['position'], barrel_rotation_angle, self.max_dist)
            if should_fire:
                barrel_rotation_angle = 0
                movement_speed = 0
                heading_direction = my_tank_status['heading']
        else:
            ammo_to_load = None
            should_fire = False
            barrel_rotation_angle = self.barrel_rotation_speed

        return ActionCommand(
            barrel_rotation_angle=barrel_rotation_angle,
            heading_rotation_angle=heading_direction - my_tank_status['heading'],
            move_speed=movement_speed,
            ammo_to_load=ammo_to_load,
            should_fire=should_fire
        )

    def destroy(self):
        self.is_destroyed = True
        print(f"[{self.name}] Tank destroyed!")

    def end(self, damage_dealt: float, tanks_killed: int):
        print(f"[{self.name}] Game ended!")
        print(f"[{self.name}] Damage dealt: {damage_dealt}")
        print(f"[{self.name}] Tanks killed: {tanks_killed}")

# ============================================================================
# FASTAPI SERVER
# ============================================================================

app = FastAPI()
agent = Agent_v3()


@app.get("/")
async def root():
    return {"message": f"Agent {agent.name} is running", "destroyed": agent.is_destroyed}


@app.post("/agent/action", response_model=ActionCommand)
async def get_action(payload: Dict[str, Any] = Body(...)):
    """Main endpoint called each tick by the engine."""
    agent.add_to_memory(payload)
    action = agent.get_action(
        payload=payload,
        current_tick=payload.get('current_tick', 0),
        my_tank_status=payload.get('my_tank_status', {}),
        sensor_data=payload.get('sensor_data', {}),
        enemies_remaining=payload.get('enemies_remaining', 0)
    )
    return action


@app.post("/agent/destroy", status_code=204)
async def destroy():
    """Called when the tank is destroyed."""
    agent.destroy()


@app.post("/agent/end", status_code=204)
async def end(payload: Dict[str, Any] = Body(...)):
    """Called when the game ends."""
    agent.end(
        damage_dealt=payload.get('damage_dealt', 0.0),
        tanks_killed=payload.get('tanks_killed', 0)
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run random test agent")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8001, help="Port number")
    parser.add_argument("--name", type=str, default=None, help="Agent name")
    args = parser.parse_args()

    if args.name:
        agent.name = args.name
    else:
        agent.name = f"RandomBot_{args.port}"

    print(f"Starting {agent.name} on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
