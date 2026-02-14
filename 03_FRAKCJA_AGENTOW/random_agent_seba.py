"""
Random Walking and Shooting Agent for Testing
Agent losowo chodzący i strzelający do testów

This agent implements IAgentController and performs random actions:
- Random barrel and heading rotation
- Random movement speed
- Random shooting

Usage:
    python random_agent.py --port 8001
    
To run multiple agents:
    python random_agent.py --port 8001  # Tank 1
    python random_agent.py --port 8002  # Tank 2
    ...
"""


PATH_CHANGE_TIME = 300

import numpy as np
import random
import argparse
import sys
import os
import math
import json

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
controller_dir = os.path.join(os.path.dirname(current_dir), '02_FRAKCJA_SILNIKA', 'controller')
sys.path.insert(0, controller_dir)

parent_dir = os.path.join(os.path.dirname(current_dir), '02_FRAKCJA_SILNIKA')
sys.path.insert(0, parent_dir)

from typing import Dict, Any
from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn


TILE_SIZE = 10.0
SUBDIV = 5
CELL_SIZE = TILE_SIZE / SUBDIV   # 5.0
PATH_CHANGE_TIME = 100  

EVAL_EVERY = 50
FORCE_REPLAN_EVERY = 1000

IMPROVEMENT_MARGIN = 0.10  
MIN_ABS_IMPROVEMENT = 2.0  



from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

@dataclass
class GridNode:
    cell: Tuple[int, int]                     # (ix, iy) w sub-grid
    world: Tuple[float, float]                # (x,y) środek sub-komórki w świecie
    dmg: int
    speed: float
    blocked: bool
    dist_to_me: float
    neighbors: List[Tuple[int, int]] = field(default_factory=list)  # 4-kierunkowo


def make_grid_helpers(tile_size, subdiv):
    if subdiv < 1:
        raise ValueError("subdiv must be >= 1")

    cell_size = tile_size / float(subdiv)
    tile_half = tile_size / 2.0

    def cell_from_xy(x, y):
        return (int(math.floor(x / cell_size)), int(math.floor(y / cell_size)))

    def cell_center(cell):
        return ((cell[0] + 0.5) * cell_size, (cell[1] + 0.5) * cell_size)

    def stamp_tile_center_to_subcells(cx, cy):
  
        cells = []
        start_x = cx - tile_half
        start_y = cy - tile_half

        for i in range(subdiv):
            sub_x = start_x + (i + 0.5) * cell_size
            for j in range(subdiv):
                sub_y = start_y + (j + 0.5) * cell_size
                cells.append(cell_from_xy(sub_x, sub_y))
        return cells

    return cell_size, cell_from_xy, stamp_tile_center_to_subcells, cell_center

# ============================================================================
# ACTION COMMAND MODEL
# ============================================================================

class ActionCommand(BaseModel):
    """Output action from agent to engine."""
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: str = None
    should_fire: bool = False


import logging
LOG_PATH = os.path.join(os.path.dirname(__file__), "agent_debug.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,  # crucial: overrides uvicorn's logging config
)

logging.info(f"Logging to: {LOG_PATH}")

class RandomAgent:
    def __init__(self, name: str = "TestBot", modifier = None):
        logging.debug(f"[DEBUG] Initializing {name}...")
        # w __init__:
        self.mode = "search"
        self.last_eval_tick = -10_000
        self.last_forced_replan_tick = -10_000
        self.current_goal_cell = None
        self.current_path_cost = None
        self.debug_goal_cell = None
        self.debug_path = None
        self.debug_path_index = 0
        self.memory = 0
        self.path_index = 0
        self.name = name
        self.last_angle = 0
        self.path_to_follow = None
        self.reached_points = None
        self.path_stuck_ticks =  0
        self.is_destroyed = False
        logging.info(f"[{self.name}] Agent initialized")
        
        self.modifier = modifier
        logging.info(f"[{self.modifier}] otrzymałem argument")

        # State for movement
        self.move_timer = 0
        self.current_move_speed = 0.0

        # State for hull rotation
        self.heading_timer = 0
        self.current_heading_rotation = 0.0        

        # State for barrel scanning
        self.barrel_scan_direction = 1.0  # 1.0 for right, -1.0 for left
        self.barrel_rotation_speed = 15.0

        # State for aiming before shooting
        self.aim_timer = 0  # Ticks to wait before firing

        # Initialize dictionaries
        self.static_info = {}
        self.dynamic_info = {}
        self.meta_info = {
            "is_aimed_at": False,
            "closest_enemy_angle": 0.0,
            "closest_enemy_dist": 707,  # 500 * sqrt(2)
            "target_id": None
        }
        self.memory = {}
        self.current_tick = 0
        self.movement_list = []
        logging.info(f"[{self.name}] Agent initialized finish")
        
        
        #### Stan czołgu - podjęcie akcji 
        self.current_state = "search"
        
        
        self.scan_offset = 0.0
        self.scan_direction = 1.0
        self.scan_max_offset = 90.0
        self.full_scan_interval = 180
        self.full_scan_active = False
        self.full_scan_remaining = 0.0
        self.last_full_scan_tick = -10_000
        
         # --- GRID CONFIG (uniwersalne) ---
        self.TILE_SIZE = 10.0
        self.SUBDIV = 5  # <-- ustawiasz jak chcesz (1,2,5,...)
        (self.CELL_SIZE,
         self._cell_from_xy,
         self._stamp_tile_center_to_subcells,
         self._cell_center) = make_grid_helpers(self.TILE_SIZE, self.SUBDIV)
    
    def _get(self, d, key, default=None):
        return d.get(key, default) if isinstance(d, dict) else getattr(d, key, default)

    def _terrain_tiles_from_seen(self):
        tiles = []
        seen_terrains = self.dynamic_info.get("visible_terrains", [])
        for t in seen_terrains:
            pos = t.get("position", {})
            cx = float(pos.get("x", 0.0))
            cy = float(pos.get("y", 0.0))
            tile_x = int(cx // self.TILE_SIZE)
            tile_y = int(cy // self.TILE_SIZE)
            tiles.append((tile_x, tile_y))
        # unikalne
        tiles = list(dict.fromkeys(tiles))
        return tiles

    def save_state_to_file(self):
        logging.debug(f"save_state_to_file")
        """Save the current state of the agent to a JSON file."""

        state = {
            "current_tick": self.current_tick,
            "static_info": self.static_info,
            "dynamic_info": self.dynamic_info,
            "meta_info": self.meta_info,
            "memory": self.memory,

            "debug": {
            "goal_cell": self.debug_goal_cell,
            "path": self.debug_path,
            "path_index": self.debug_path_index,
            "seen_terrain_tiles": self._terrain_tiles_from_seen(),
            "my_pos": self.dynamic_info.get("position", None),
            }
        }
        
        with open('agent_state.json', 'w') as f:
            json.dump(state, f, indent=4)
        # logging.debug("[DEBUG] Agent state saved to agent_state.json")
    
    def _path_cost(self, path, node_by_cell):
        if not path or len(path) < 2:
            return float("inf")
        cost = 0.0
        for cell in path[1:]:
            n = node_by_cell.get(cell)
            if n is None:
                return float("inf")  
            if n.blocked:
                return float("inf")
            cost += float(n.dmg)
        return cost
    
    
    def _plan_best_path(self, nodes, targets, max_targets=5):
        node_by_cell = {n.cell: n for n in nodes}

        best_path = None
        best_goal = None
        best_cost = float("inf")

        for goal in targets[:max_targets]:
            path = self._a_star(nodes, goal)
            if not path:
                continue
            c = self._path_cost(path, node_by_cell)
            if c < best_cost:
                best_cost = c
                best_path = path
                best_goal = goal

        return best_path, best_goal, best_cost
        
        
    def  next_or_first(self, arr, value):
        try:
            index = arr.index(value)
        except ValueError:
            return 1  # or raise an error if value is not found

        if index + 1 < len(arr):
            return arr[index + 1]
        else:
            return arr[0]

    def get_action(self, current_tick: int, my_tank_status: Dict[str, Any], sensor_data: Dict[str, Any], enemies_remaining: int) -> ActionCommand:
        # logging.debug(f"[DEBUG] get_action called at tick {current_tick}")

        if self.movement_list != None:
            current_angle = self.next_or_first(movement_list, self.last_angle)
            print(current_angle)
            self.last_angle = current_angle
        else:
            print("No movement list provided")
            
            

        self.current_tick = current_tick
        self.enemies_remaining = enemies_remaining
        # logging.debug(f"[DEBUG] enemies_remaining: {self.enemies_remaining}")

        # Update internal state
        self._update_internal_state(my_tank_status, sensor_data)

        # Process action
        action = self._process_action()

        # Save state to file
        self.save_state_to_file()

        # logging.debug(f"[DEBUG] Action generated: {action}")
        return action
    

    def _update_internal_state(self, status, sensors):
        # logging.debug("[DEBUG] Updating internal state with status and sensors...")

        # Safely extract static info
        self.static_info = {
            "id": self._get(status, '_id', 'MISSING'),
            "team": self._get(status, '_team', 'MISSING'),
            "tank_type": self._get(status, '_tank_type', 'MISSING'),
            "vision_angle": self._get(status, '_vision_angle', 'MISSING'),
            "vision_range": self._get(status, '_vision_range', 'MISSING'),
            "top_speed": self._get(status, '_top_speed', 'MISSING'),
            "barrel_spin_rate": self._get(status, '_barrel_spin_rate', 30),                  #### NIE OBECNE W PAYLOAD 
            "heading_spin_rate": self._get(status, '_heading_spin_rate', 2),                 #### NIE OBECNE W PAYLOAD 
            "max_hp": self._get(status, '_max_hp', 'MISSING'),
            "max_shield": self._get(status, '_max_shield', 'MISSING'),
        }
        # logging.debug(f"[DEBUG] static_info updated: {self.static_info}")

        # Safely extract dynamic info
        self.dynamic_info = {
            "hp": self._get(status, 'hp', 'MISSING'),
            "shield": self._get(status, 'shield', 'MISSING'),
            "position": self._get(status, 'position', 'MISSING'),
            "move_speed": self._get(status, 'move_speed', 'MISSING'),
            "barrel_angle": self._get(status, 'barrel_angle', 'MISSING'),
            "heading": self._get(status, 'heading', 'MISSING'),
            "ammo": self._get(status, 'ammo', 'MISSING'),
            "ammo_loaded": self._get(status, 'ammo_loaded', 'MISSING'),
            "is_overcharged": self._get(status, 'is_overcharged', 'MISSING'),
            "size": self._get(status, 'size', 'MISSING'),
            "reload_timer": self._get(status, '_reload_timer', 'MISSING'),
            "enemies_remaining": self.enemies_remaining,
            "visible_tanks": self._get(sensors, 'seen_tanks', []),
            "visible_enemies": [tank for tank in self._get(sensors, 'seen_tanks', []) if tank["team"] != self.static_info["team"]],
            "visible_obstacles": self._get(sensors, 'seen_obstacles', []),
            "visible_terrains": self._get(sensors, 'seen_terrains', []),
            "visible_powerups": self._get(sensors, "seen_powerups", []),
            
        }
        # logging.debug(f"[DEBUG] dynamic_info updated: {self.dynamic_info}")

        # Process meta info and memory
        self._process_meta_info()
        self._process_memory()
        

    def _process_memory(self):
        # logging.debug("[DEBUG] Processing memory for visible enemies...")
        visible_enemies = self.dynamic_info.get('visible_enemies', [])

        for enemy in visible_enemies:
            # logging.debug(f"[DEBUG] Updating memory for enemy {self._get(enemy, 'id', 'MISSING')}")
            self.memory[self._get(enemy, 'id', 'MISSING')] = {
                "id": self._get(enemy, 'id', 'MISSING'),
                "last_seen_pos": self._get(enemy, 'position', 'MISSING'),
                "last_seen_tick": self.current_tick,
                "tank_type": self._get(enemy, 'tank_type', 'MISSING'),
                "team": self._get(enemy, 'team', 'MISSING'),
            }
        # logging.debug(f"[DEBUG] memory updated: {self.memory}")
        

    def _process_meta_info(self):
        # logging.debug("[DEBUG] Processing meta info...")
        my_pos = self.dynamic_info.get('position', None)
        my_angle = self.dynamic_info.get('barrel_angle', 0.0)
        visible_enemies = self.dynamic_info.get('visible_enemies', [])

        if not my_pos or not visible_enemies:
            # logging.debug("[DEBUG] No position or visible enemies!")
            return

        closest_angle = 10.0
        closest_dist = 9999.0
        target_id = None
        is_targeted = False

        if visible_enemies:
            try:
                closest = sorted(visible_enemies, key=lambda t: self._get(t, 'distance', 9999))[0]
                target_id = self._get(closest, 'id', None)
                closest_dist = self._get(closest, 'distance', 9999)

                dx = self._get(self._get(closest, 'position', None), 'x', 0) - self._get(my_pos, 'x', 0)
                dy = self._get(self._get(closest, 'position', None), 'y', 0) - self._get(my_pos, 'y', 0)
                angle_rad = math.atan2(dy, dx)
                desired_angle = math.degrees(angle_rad)
                diff = (desired_angle - my_angle + 180) % 360 - 180
                closest_angle = diff
            except Exception as e:
                logging.error(f"[ERROR] Failed to calculate closest enemy angle: {e}")

        for enemy in visible_enemies:
            try:
                dx = self._get(my_pos, 'x', 0) - self._get(self._get(enemy, 'position', None), 'x', 0)
                dy = self._get(my_pos, 'y', 0) - self._get(self._get(enemy, 'position', None), 'y', 0)
                angle_to_me = math.degrees(math.atan2(dy, dx))
                aim_diff = (angle_to_me - self._get(enemy, 'barrel_angle', 0) + 180) % 360 - 180
                if abs(aim_diff) < 15:
                    is_targeted = True
                    break
            except Exception as e:
                logging.error(f"[ERROR] Failed to check if enemy is aiming: {e}")

        self.meta_info = {
            "closest_enemy_angle": closest_angle,
            "closest_enemy_dist": closest_dist,
            "target_id": target_id,
            "is_aimed_at": is_targeted,
        }

    def _angle_diff(self, target_deg: float, current_deg: float) -> float:
        return (target_deg - current_deg + 180) % 360 - 180

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))
    

    def _scan_strategy(self) -> float:
            """
            heading: aboslute to map
            barrel_angle: relative to tank
            barrel_abs = (heading + barrel_angle) % 360

            Modes (movement_list[0]):
            1: fast scan (continuous spin)
            2: slow scan (continuous spin, half speed)
            3: random look
            4: look in heading direction always (barrel_abs == heading)
            5: look around heading with sweep -90..+90 (memory-based)
            6: as 5, plus occasional full 360 scan (memory-based)
            """
            if not self.movement_list:
                return 0.0

            mode = self.movement_list[0]

            barrel_spin = float(self.static_info.get("barrel_spin_rate", 0.0))
            if barrel_spin <= 0.0:
                return 0.0

            heading_abs = float(self.dynamic_info.get("heading", 0.0))
            barrel_rel = float(self.dynamic_info.get("barrel_angle", 0.0))
            barrel_abs = (heading_abs + barrel_rel) % 360.0

            def clamp_step(step: float) -> float:
                return self._clamp(step, -barrel_spin, barrel_spin)

            # 1) Fast scan: constant rotation
            if mode == 1:
                return clamp_step(barrel_spin * self.scan_direction)

            # 2) Slow scan: constant rotation, slower
            if mode == 2:
                return clamp_step((barrel_spin * 0.5) * self.scan_direction)

            # 3) Random look
            if mode == 3:
                return random.uniform(-barrel_spin, barrel_spin)

            # 4) Lock barrel to heading direction
            if mode == 4:
                desired_abs = heading_abs
                err = self._angle_diff(desired_abs, barrel_abs)
                # Reset scan memory to avoid drift when switching modes
                self.scan_offset = 0.0
                self.scan_direction = 1.0
                self.full_scan_active = False
                return clamp_step(err)

            # Modes 5/6: partial sweep around heading
            scan_center_abs = heading_abs
            self.scan_max_offset = 90.0

            # 6) Occasionally do a full 360 scan (relative rotation)
            if mode == 6:
                if (not self.full_scan_active) and (self.current_tick - self.last_full_scan_tick >= self.full_scan_interval):
                    self.full_scan_active = True
                    self.full_scan_remaining = 360.0
                    self.last_full_scan_tick = self.current_tick

                if self.full_scan_active:
                    step = barrel_spin * self.scan_direction
                    self.full_scan_remaining -= abs(step)
                    if self.full_scan_remaining <= 0.0:
                        self.full_scan_active = False
                        self.scan_offset = 0.0
                    return clamp_step(step)

            # 5) (and 6 when not in full scan): oscillate offset and aim to (center + offset)
            if mode in (5, 6):
                # advance offset in world terms
                step = barrel_spin * self.scan_direction
                self.scan_offset += step

                if self.scan_offset > self.scan_max_offset:
                    self.scan_offset = self.scan_max_offset
                    self.scan_direction = -1.0
                elif self.scan_offset < -self.scan_max_offset:
                    self.scan_offset = -self.scan_max_offset
                    self.scan_direction = 1.0

                desired_abs = (scan_center_abs + self.scan_offset) % 360.0
                err = self._angle_diff(desired_abs, barrel_abs)
                return clamp_step(err)

            return 0.0
    
    def _divide_seen_area(self, visible_obstacles, visible_terrains):
        """
        Zwraca listę węzłów (GridNode). Każdy węzeł ma:
        - dmg, speed, blocked
        - neighbors 4-kierunkowo (tylko jeśli istnieją)
        - dist_to_me policzone od razu (na podstawie aktualnej pozycji czołgu)
        """

        # aktualna pozycja czołgu (world coords)
        my_pos = self.dynamic_info.get("position", {"x": 0.0, "y": 0.0})
        me_x = float(my_pos.get("x", 0.0))
        me_y = float(my_pos.get("y", 0.0))

        # 1) cell -> terrain info
        terrain_info = {}
        for t in visible_terrains:
            pos = t.get("position", {})
            cx = float(pos.get("x", 0.0))
            cy = float(pos.get("y", 0.0))

            dmg = int(t.get("dmg", 0))
            speed = float(t.get("speed_modifier", 1.0))

            for cell in self._stamp_tile_center_to_subcells(cx, cy):
                terrain_info[cell] = {"dmg": dmg, "speed": speed}

        blocked_cells = set()
        inflate_cells = 0  
        
        for ob in visible_obstacles:
            pos = ob.get("position", {})
            cx = float(pos.get("x", 0.0))
            cy = float(pos.get("y", 0.0))

            for cell in self._stamp_tile_center_to_subcells(cx, cy):
                for dx in range(-inflate_cells, inflate_cells + 1):
                    for dy in range(-inflate_cells, inflate_cells + 1):
                        blocked_cells.add((cell[0] + dx, cell[1] + dy))

        nodes_by_cell = {}

        for cell, info in terrain_info.items():
            wx, wy = self._cell_center(cell)
            dist = math.hypot(wx - me_x, wy - me_y)

            nodes_by_cell[cell] = GridNode(
                cell=cell,
                world=(wx, wy),
                dmg=int(info["dmg"]),
                speed=float(info["speed"]),
                blocked=(cell in blocked_cells),
                dist_to_me=dist,
                neighbors=[]
            )

        DIRS4 = [(1,0), (-1,0), (0,1), (0,-1)]
        for cell, node in nodes_by_cell.items():
            x, y = cell
            for dx, dy in DIRS4:
                nb = (x + dx, y + dy)
                if nb in nodes_by_cell:
                    node.neighbors.append(nb)

        return list(nodes_by_cell.values())
    
    def _presence(self, obj):
        if obj is None:
            return "not present"
        if isinstance(obj, (list, dict, set, tuple)) and len(obj) == 0:
            return "not present"
        return "present"
        
    def _select_target_point(self, divided_area, top_k = 20):

        # 1) odrzuć zablokowane
        candidates = [n for n in divided_area if not n.blocked]
        if not candidates:
            return []

        # 2) top_k najdalszych
        candidates.sort(key=lambda n: n.dist_to_me, reverse=True)
        top = candidates[:min(top_k, len(candidates))]

        # 3) najbezpieczniejsze w top_k: dmg rosnąco, dist malejąco
        top.sort(key=lambda n: (n.dmg, -n.dist_to_me))

        return [n.cell for n in top]

    
    def _a_star(self, nodes, goal_cell):
        if nodes is None:
            raise RuntimeError("_a_star: nodes is None")
        if len(nodes) == 0:
            raise RuntimeError("_a_star: nodes is empty (no grid nodes built)")

        node_by_cell = {n.cell: n for n in nodes}

        
        my_pos = self.dynamic_info.get("position", {"x": 0.0, "y": 0.0})
        sx = float(my_pos.get("x", 0.0))
        sy = float(my_pos.get("y", 0.0))

        start_cell = None
        best_d2 = 1e18
        for cell, node in node_by_cell.items():
            wx, wy = node.world
            d2 = (wx - sx) * (wx - sx) + (wy - sy) * (wy - sy)
            if d2 < best_d2:
                best_d2 = d2
                start_cell = cell

        def Heuristic_Cost(cell):
            return abs(cell[0] - goal_cell[0]) + abs(cell[1] - goal_cell[1])

        list_of_possible_paths = [{
            "Path": [start_cell],
            "Path_Cost": 0,
            "Heuristic": Heuristic_Cost(start_cell),
        }]

        best_paths = {start_cell: 0}

        while list_of_possible_paths:

            list_of_possible_paths.sort(
                key=lambda p: p["Path_Cost"] + p["Heuristic"]
            )

            current_best_path = list_of_possible_paths.pop(0)

            current_node = current_best_path["Path"][-1]
            current_path_length = current_best_path["Path_Cost"]

            if current_node == goal_cell:
                return current_best_path["Path"]

            for neighbour in node_by_cell[current_node].neighbors:

                next_path_length = current_path_length + 1

                if neighbour in best_paths and next_path_length >= best_paths[neighbour]:
                    continue

                best_paths[neighbour] = next_path_length

                new_path = current_best_path["Path"] + [neighbour]

                list_of_possible_paths.append({
                    "Path": new_path,
                    "Path_Cost": next_path_length,
                    "Heuristic": Heuristic_Cost(neighbour),
                })
                
        return None
    
    
    def _Find_Target_and_Find_Path(self):
        visible_obstacles = self.dynamic_info.get("visible_obstacles")
        visible_terrains = self.dynamic_info.get("visible_terrains")

        if not visible_obstacles or not visible_terrains:
            return None, None, None, None  # path, goal, cost, node_by_cell

        nodes = self._divide_seen_area(visible_obstacles, visible_terrains)
        if not nodes:
            return None, None, None, None

        targets = self._select_target_point(nodes)  # lista komórek
        if not targets:
            return None, None, None, None

        path, goal, cost = self._plan_best_path(nodes, targets, max_targets=5)
        node_by_cell = {n.cell: n for n in nodes}
        return path, goal, cost, node_by_cell
        
    
##############################################################################################   
    def _FollowPath(self):
        if not self.path_to_follow:
            raise RuntimeError("_FollowPath: path_to_follow is empty/None")

        my_position = self.dynamic_info.get("position")
        my_x = float(my_position.get("x", 0.0))
        my_y = float(my_position.get("y", 0.0))

        if self.path_index < 0:
            self.path_index = 0
        if self.path_index >= len(self.path_to_follow) - 1:
            return 0.0, 0.0  
        next_cell = self.path_to_follow[self.path_index + 1]
        next_x, next_y = self._cell_center(next_cell)
        
        dist = np.sqrt((next_x - my_x)**2 + (next_y - my_y)**2)

        # warunki "reached"
        reach_threshold = self.TILE_SIZE / 10.0  
        reached_by_distance = dist <= reach_threshold
        reached_by_timeout = self.path_stuck_ticks >= 100

        if reached_by_distance or reached_by_timeout:
            if reached_by_timeout:
                print(f"[STUCK] skipping point {next_cell} after 50 ticks")
            self.path_index += 1
            self.path_stuck_ticks = 0

            if self.path_index >= len(self.path_to_follow) - 1:
                return 0.0, 0.0

            next_cell = self.path_to_follow[self.path_index + 1]
            next_x, next_y = self._cell_center(next_cell)
            dist = math.hypot(next_x - my_x, next_y - my_y)

        self.path_stuck_ticks += 1

        # kąt docelowy
        dx = next_x - my_x
        dy = next_y - my_y
        desired_heading = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0

        current_heading = float(self.dynamic_info.get("heading", 0.0))
        err = self._angle_diff(desired_heading, current_heading)

        heading_spin = float(self.static_info.get("heading_spin_rate", 0.0))
        heading_rotation_angle = self._clamp(err, -heading_spin, heading_spin)

        top_speed = float(self.static_info.get("top_speed", 0.0))

        # ruch zależny od tego jak bardzo jesteśmy odchyleni od kierunku
        
        move_speed = top_speed

        return heading_rotation_angle, move_speed
            
##############################################################################################  
    
    def Follow_Path_With_Modifiers(self):
        EVAL_EVERY = 50
        FORCE_REPLAN_EVERY = 1000

        IMPROVEMENT_MARGIN = 0.10     
        MIN_ABS_IMPROVEMENT = 2.0      

        def angle_diff(target_deg: float, current_deg: float) -> float:
            return (target_deg - current_deg + 180) % 360 - 180

        def clamp(x: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, x))

        def should_force_replan() -> bool:
            return (self.current_tick - self.last_forced_replan_tick) >= FORCE_REPLAN_EVERY

        def should_eval() -> bool:
            return (self.current_tick - self.last_eval_tick) >= EVAL_EVERY

        def path_cost(path, node_by_cell) -> float:
            if not path or len(path) < 2:
                return float("inf")
            c = 0.0
            for cell in path[1:]:
                n = node_by_cell.get(cell)
                if n is None or n.blocked:
                    return float("inf")
                c += float(n.dmg)
            return c

        def better_enough(new_cost: float, old_cost: float) -> bool:
            if old_cost is None or old_cost == float("inf"):
                return True
            if new_cost is None or new_cost == float("inf"):
                return False
            if new_cost <= old_cost * (1.0 - IMPROVEMENT_MARGIN):
                return True
            if (old_cost - new_cost) >= MIN_ABS_IMPROVEMENT:
                return True
            return False


        force = should_force_replan()
        eval_now = should_eval() or force

        if eval_now:
            self.last_eval_tick = self.current_tick

            new_path, new_goal, new_cost, node_by_cell = self._Find_Target_and_Find_Path()

            if new_path and new_goal is not None:
                do_change = False

                if force:
                    do_change = True
                    self.last_forced_replan_tick = self.current_tick
                else:
                    do_change = better_enough(new_cost, self.current_path_cost)

                if do_change:
                    self.path_to_follow = new_path
                    self.current_goal_cell = new_goal
                    self.current_path_cost = new_cost

                    # reset progresu na nowej ścieżce
                    self.path_index = 0
                    self.path_stuck_ticks = 0

                    print(
                        f"[REPLAN] tick={self.current_tick} "
                        f"goal={self.current_goal_cell} cost={self.current_path_cost:.2f} "
                        f"force={force}"
                    )
        hull_rot = 0.0
        move_speed = 0.0
        if self.path_to_follow is not None:
            hull_rot, move_speed = self._FollowPath()
        
        self.debug_goal_cell = self.current_goal_cell
        self.debug_path = self.path_to_follow
        self.debug_path_index = self.path_index
        
        return hull_rot, move_speed
    
    
    
    
##############################################################
    def _closest_visible_enemy(self):
        visible_enemies = self.dynamic_info.get("visible_enemies", [])
        if not visible_enemies:
            return None
        return min(visible_enemies, key=lambda enemy: self._get(enemy, "distance", 999999.0))


    def _aim_barrel_at_enemy(self, enemy, aim_eps_deg=2.0):

        if enemy is None:
            return 0.0

        my_position = self.dynamic_info.get("position") or {}
        enemy_position = self._get(enemy, "position", {}) or {}

        my_x = float(self._get(my_position, "x", 0.0))
        my_y = float(self._get(my_position, "y", 0.0))
        enemy_x = float(self._get(enemy_position, "x", 0.0))
        enemy_y = float(self._get(enemy_position, "y", 0.0))

        desired_abs_angle = (math.degrees(math.atan2(enemy_y - my_y, enemy_x - my_x)) + 360.0) % 360.0

        heading_abs = float(self.dynamic_info.get("heading", 0.0))
        barrel_rel = float(self.dynamic_info.get("barrel_angle", 0.0))
        barrel_abs = (heading_abs + barrel_rel) % 360.0

        error_angle = self._angle_diff(desired_abs_angle, barrel_abs)

        if abs(error_angle) <= aim_eps_deg:
            return 0.0

        barrel_spin_rate = float(self.static_info.get("barrel_spin_rate", 0.0))
        return self._clamp(error_angle, -barrel_spin_rate, barrel_spin_rate)


    def _can_fire_at_enemy(self, enemy, aim_tolerance_deg=5.0):
        if enemy is None:
            return False

        reload_timer = self.dynamic_info.get("reload_timer", None)
        if reload_timer is None:
            reload_timer = self.dynamic_info.get("current_reload_progress", 0)

        try:
            if int(reload_timer) > 0:
                return False
        except:
            pass

        my_position = self.dynamic_info.get("position") or {}
        enemy_position = self._get(enemy, "position", {}) or {}

        my_x = float(self._get(my_position, "x", 0.0))
        my_y = float(self._get(my_position, "y", 0.0))
        enemy_x = float(self._get(enemy_position, "x", 0.0))
        enemy_y = float(self._get(enemy_position, "y", 0.0))

        desired_abs_angle = (math.degrees(math.atan2(enemy_y - my_y, enemy_x - my_x)) + 360.0) % 360.0

        heading_abs = float(self.dynamic_info.get("heading", 0.0))
        barrel_rel = float(self.dynamic_info.get("barrel_angle", 0.0))
        barrel_abs = (heading_abs + barrel_rel) % 360.0

        error_angle = self._angle_diff(desired_abs_angle, barrel_abs)
        if abs(error_angle) > aim_tolerance_deg:
            return False

        ammo_loaded = self.dynamic_info.get("ammo_loaded", None)
        if ammo_loaded is None:
            return False

        return True
##############################################################
    
    def _process_action(self) -> ActionCommand:
        
        
        MODE = self.mode

        if self.dynamic_info["visible_enemies"]:
            # TODO dodać licznik żeby tryb się nie zmienia przy immym kącie obserwacji  
            # TODO dodać wymuszenie patrzenia w kierunku wroga  
            print("visible enemies",self.dynamic_info["visible_enemies"])
            MODE = "attack"
        elif self.dynamic_info["visible_powerups"]:
            # TODO dodać licznik żeby tryb się nie zmienia przy immym kącie obserwacji  
            MODE = "power_up"
        else:
            MODE = "search"
            
            
        print("MODE")
        print(MODE)
        print(MODE)
        print(MODE)
        print(MODE)
        print("MODE")
        

        if MODE == "search":
            barrel_rot = self._scan_strategy()
            hull_rot, move_speed =  self.Follow_Path_With_Modifiers()
            should_fire = False
            ammo_to_load = random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"])
        
        if MODE == "attack":
            hull_rot, move_speed = self.Follow_Path_With_Modifiers()

            enemy = self._closest_visible_enemy()
            barrel_rot = self._aim_barrel_at_enemy(enemy)

            ammo_loaded = self.dynamic_info.get("ammo_loaded", None)

            if ammo_loaded is None:
                ammo_to_load = "LIGHT"
                ammo = self.dynamic_info.get("ammo", {}) or {}
                for ammo_type, ammo_slot in ammo.items():
                    name = getattr(ammo_type, "name", str(ammo_type))
                    count = getattr(ammo_slot, "count", 0)
                    if count > 0:
                        ammo_to_load = name
                        break

                should_fire = False 
            else:
                ammo_to_load = getattr(ammo_loaded, "name", str(ammo_loaded))  
                should_fire = self._can_fire_at_enemy(enemy)
                                
                    
        if MODE == "power_up":
            ### Celem jest power up
            barrel_rot = self._scan_strategy()
            hull_rot, move_speed =  self.Follow_Path_With_Modifiers()
            should_fire = False
            ammo_to_load = random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"])

        return ActionCommand(
            barrel_rotation_angle=barrel_rot,
            heading_rotation_angle=hull_rot,
            move_speed=move_speed,
            should_fire=should_fire,
            ammo_to_load=ammo_to_load
        )

    

    def destroy(self):
        self.is_destroyed = True
        logging.info(f"[{self.name}] Tank destroyed!")

    def end(self, damage_dealt: float, tanks_killed: int):
        logging.info(f"[{self.name}] Game ended!")
        logging.info(f"[{self.name}] Damage dealt: {damage_dealt}")
        logging.info(f"[{self.name}] Tanks killed: {tanks_killed}")

# ============================================================================
# FASTAPI SERVER
# ============================================================================

app = FastAPI(
    title="Random Test Agent",
    description="Random walking and shooting agent for testing",
    version="1.0.0"
)

# Global agent instance
agent = RandomAgent()
from fastapi import HTTPException

@app.get("/")
async def root():
    return {"message": f"Agent {agent.name} is running", "destroyed": agent.is_destroyed}


@app.post("/agent/action", response_model=ActionCommand)
async def get_action(payload: Dict[str, Any] = Body(...)):
    # print("\n=== RAW PAYLOAD RECEIVED ===")
    # print(payload)
    try:
        # logging.debug("=== RAW PAYLOAD RECEIVED ===")
        # logging.debug(json.dumps(payload, indent=2, default=str))

        action = agent.get_action(
            current_tick=payload.get('current_tick', 0),
            my_tank_status=payload.get('my_tank_status', {}),
            sensor_data=payload.get('sensor_data', {}),
            enemies_remaining=payload.get('enemies_remaining', 0)
        )
        return action

    except Exception as e:
        logging.exception("Exception in /agent/action")
        raise HTTPException(status_code=500, detail=str(payload))
    
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
    parser.add_argument("--modifier", type=str, default=None, help="Modifier")    #### ACHTUNG - sprawdzić czy modifier działa 
    args = parser.parse_args()
    
    if args.name:
        agent.name = args.name
    else:
        agent.name = f"RandomBot_{args.port}"
        
    if args.modifier != None:
        try:
            agent.modifier = args.modifier
            modifier = agent.modifier.split("_")
            movement_list = [int(element) for element in modifier]
            agent.movement_list = movement_list
            print("agent.movement_list: ", agent.movement_list)
        except:
            print("No modifier except")
    else:
        print("No mdofier")

    print(f"Starting {agent.name} on {args.host}:{args.port}, with modifier {args.modifier}")
    
    uvicorn.run(app, host=args.host, port=args.port)
