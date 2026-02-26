import random
import argparse
import sys
import os
import math
import itertools
from Lib.scan_strategy import scan_strategy
from Lib.agent_memory_update import update_internal_state

from Lib.Attack import mode_attack
from Lib.Escape import mode_escape
from Lib.Fuzzy_Controller import FuzzyCombatDecider
from Lib.a_star import a_star
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MAX_PATH_STUCK_TICKS = 300
NO_MOVE_LIMIT_TICKS = 200

ESCAPE_COMMIT_TICKS = 300      
ESCAPE_BACK_TICKS   = 150       
ESCAPE_REPLAN_EVERY = 120     

current_dir = os.path.dirname(os.path.abspath(__file__))
controller_dir = os.path.join(os.path.dirname(current_dir), '02_FRAKCJA_SILNIKA', 'controller')
sys.path.insert(0, controller_dir)

parent_dir = os.path.join(os.path.dirname(current_dir), '02_FRAKCJA_SILNIKA')
sys.path.insert(0, parent_dir)

from typing import Dict, Any
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
import uvicorn
from dataclasses import dataclass, field
import logging

TILE_SIZE = 10.0
SUBDIV = 1  # Must be 1 to enforce CELL_SIZE = 10.0
EVAL_EVERY = 1000

FORCE_REPLAN_EVERY = 1000
IMPROVEMENT_MARGIN = 0.3
MIN_ABS_IMPROVEMENT = 20.0

LOG_PATH = os.path.join(os.path.dirname(__file__), "agent_debug.log")

def make_grid_helpers(tile_size, subdiv):
    cell_size = tile_size / float(subdiv)
    def cell_from_xy(x, y): return (int(math.floor(x / cell_size)), int(math.floor(y / cell_size)))
    def cell_center(cell): return ((cell[0] + 0.5) * cell_size, (cell[1] + 0.5) * cell_size)
    return cell_size, cell_from_xy, cell_center

class ActionCommand(BaseModel):
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: str = None
    should_fire: bool = False

@dataclass
class GridNode:
    cell: tuple                     
    world: tuple                
    dmg: int
    speed: float
    blocked: bool
    dist_to_me: float
    neighbors: list = field(default_factory=list)  
    is_risk: bool = False
    is_water: bool = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)], force=True)

class RandomAgent:
    def __init__(self, name: str = "TestBot", modifier=None):
        self.name = name
        self.modifier = modifier
        self.is_destroyed = False
        self.current_tick = 0
        
        self.state_dir = os.path.join(os.path.dirname(__file__), "agent_states")
        os.makedirs(self.state_dir, exist_ok=True)
        self.state_path = os.path.join(self.state_dir, f"agent_state_{self.name}.json")
        
        self.plot_dir = os.path.join(os.path.dirname(__file__), "plots")
        os.makedirs(self.plot_dir, exist_ok=True)

        self.tank_type = None 
        self.tank_type_used = True
        self.friend_target_cell = None
        self.friend_target_last_seen_tick = -10_000
        self.friend_forget_after = 120

        self.mode = "search"              

        self.attack_commit_ticks = 100
        self.attack_until_tick = -10_000

        self.enemy_target_id = None
        self.enemy_target_cell = None
        self.enemy_target_last_seen_tick = -10_000
        self.enemy_forget_after = 80
        
        self.escape_commit_ticks = ESCAPE_COMMIT_TICKS
        self.escape_until_tick = -10_000
        self.escape_target_cell = None
        self.escape_target_last_pick_tick = -10_000

        self.powerup_target_cell = None
        self.powerup_target_last_seen_tick = -10_000
        self.powerup_target_acquired_tick = -10_000
        self.powerup_forget_after = 200
        self.powerup_commit_ticks = 200
        self.powerup_switch_ratio = 0.80

        self.last_world_pos = None
        self.last_commanded_speed = 0.0
        self.no_move_ticks = 0
        self.force_change_goal = False

        self.MIN_MOVE_EPS = 0.05
        self.NO_MOVE_LIMIT_TICKS = 100 

        self.last_eval_tick = -10_000
        self.last_forced_replan_tick = -10_000

        self.current_goal_cell = None
        self.current_path_cost = None

        self.path_to_follow = None
        self.path_index = 0
        self.path_stuck_ticks = 0
        
        self.debug_goal_cell = None
        self.debug_path = None
        self.debug_path_index = 0

        self.movement_list = []
        self.last_angle = 0
        self.unstuck_should_fire = False

        self.static_info = {}
        self.dynamic_info = {}
        self.virtual_map = {} 
        # Matrix must strictly be 20x20 (400 elements)
        self.virtual_map = {(x, y): {"type": 0, "tick": 0} for x in range(20) for y in range(20)}
        self.memory = {} # Fixed initialization for debug plotting

        self.TILE_SIZE = TILE_SIZE
        self.SUBDIV = SUBDIV
        self.CELL_SIZE, self._cell_from_xy, self._cell_center = make_grid_helpers(self.TILE_SIZE, self.SUBDIV)
        
        self.combat_decider = FuzzyCombatDecider(cooldown_ticks=30)
    
    def _get(self, d, key, default=None): return d.get(key, default) if isinstance(d, dict) else getattr(d, key, default)

    def _dist2_to_cell_center(self, cell):
        """Helper to calculate squared distance to a cell's center, preventing the AttributeError."""
        cx, cy = self._cell_center(cell)
        mx = float(self.dynamic_info.get("position", {}).get("x", 0.0))
        my = float(self.dynamic_info.get("position", {}).get("y", 0.0))
        return (cx - mx)**2 + (cy - my)**2

    def _save_map_plot(self):
        # Image matrix strictly matches the 20x20 map
        img = np.full((20, 20, 3), 128, dtype=np.uint8) 
        colors = {
            0: [100, 100, 100], 1: [144, 238, 144], 2: [0, 0, 255], 
            3: [0, 0, 0], 4: [34, 139, 34], 5: [255, 0, 0], 6: [139, 69, 19],
        }

        for (cx, cy), data in self.virtual_map.items():
            if 0 <= cx < 20 and 0 <= cy < 20:
                img[cy, cx] = colors.get(data["type"], [100, 100, 100])

        fig, ax = plt.subplots(figsize=(10, 10))
        # Extent remains 200 to scale the 20x20 matrix up to the world coordinates overlay
        ax.imshow(img, origin='lower', extent=[0, 200, 0, 200])

        if getattr(self, "path_to_follow", None):
            px = [self._cell_center(c)[0] for c in self.path_to_follow]
            py = [self._cell_center(c)[1] for c in self.path_to_follow]
            ax.plot(px, py, color='magenta', linewidth=1.5, marker='o', markersize=2, label='Path')

        my_pos = self.dynamic_info.get("position", {})
        mx, my = float(my_pos.get("x", 0.0)), float(my_pos.get("y", 0.0))
        tank_rect = plt.Rectangle((mx - 2.5, my - 2.5), 5, 5, color='yellow', zorder=10, label='Tank')
        ax.add_patch(tank_rect)

        ax.set_xticks(np.arange(0, 201, 10))
        ax.set_yticks(np.arange(0, 201, 10))
        ax.set_xticks(np.arange(0, 201, 1), minor=True)
        ax.set_yticks(np.arange(0, 201, 1), minor=True)
        
        ax.grid(which='major', color='white', linestyle='-', linewidth=0.8, alpha=0.7)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.2, alpha=0.3)

        plt.title(f"Virtual Map - {self.name} (Tick {self.current_tick})")
        plt.legend(loc='upper right')
        
        plot_path = os.path.join(self.plot_dir, f"map_{self.name}_{self.current_tick}.png")
        
        # Save explicitly to file, bypassing display streams
        plt.savefig(plot_path, bbox_inches='tight')
        
        # Aggressive memory cleanup
        plt.close(fig)
        plt.close('all')

    def _commit_mode(self, mode: str, enemy) -> None:
        mode = (mode or "").lower()
        if mode == "attack":
            self._enter_attack(enemy)
            self.mode = "attack"
        elif mode == "escape":
            self._enter_escape(enemy)
            self.mode = "escape"

    def _maybe_reconsider_combat_mode(self, enemy) -> bool:
        if enemy is None: 
            return False
            
        # Get the fuzzy controller's current optimal decision based on HP and Ammo
        desired_mode = self.combat_decider.decide(self)
        
        # Determine the agent's currently committed mode
        current_mode = "attack" if self._should_stay_in_attack() else "escape" if self._should_stay_in_escape() else None
        
        # If the fuzzy controller dictates a change, break current mode commitments
        if current_mode and desired_mode != current_mode:
            self.attack_until_tick = -10_000
            self.escape_until_tick = -10_000
            self.escape_target_locked = False
            return True
            
        return False
    

    def Follow_Path_With_Modifiers(self, override_goal_cell=None):
        force = False
        
        # 1. Lookahead Obstacle Detection
        if self.path_to_follow and self.path_index < len(self.path_to_follow):
            for i in range(self.path_index, min(len(self.path_to_follow), self.path_index + 15)):
                ctype = self.virtual_map.get(self.path_to_follow[i], {}).get("type", 0)
                # Replan if path intersects newly seen Walls (3) or Danger/Potholes (5)
                if ctype in [3, 5]: 
                    force = True
                    break
                    
        # 2. Standard completion/failure triggers
        if self.path_stuck_ticks > MAX_PATH_STUCK_TICKS or self.path_to_follow is None or self.path_index >= len(self.path_to_follow) - 1:
            force = True
        if override_goal_cell is not None and override_goal_cell != self.current_goal_cell:
            force = True
        if getattr(self, "force_change_goal", False):
            force = True

        # 3. Execution & Failsafe
        if force:
            new_path, new_goal, new_cost, _ = self._Find_Target_and_Find_Path(override_goal_cell)
            if new_path and new_goal is not None:
                self.path_to_follow, self.current_goal_cell, self.current_path_cost = new_path, new_goal, new_cost
                self.path_index, self.path_stuck_ticks, self.force_change_goal = 0, 0, False
            else:
                # CRITICAL: If no path found, kill the current path to prevent infinite loops
                self.path_to_follow = None
                self.current_goal_cell = None
                self.force_change_goal = True 

        hull_rot, move_speed = self._FollowPath() if self.path_to_follow is not None else (0.0, 0.0)
        self.debug_goal_cell, self.debug_path, self.debug_path_index = self.current_goal_cell, self.path_to_follow, self.path_index
        return hull_rot, move_speed    
    
    
    def _choose_combat_mode(self, enemy) -> str: return self.combat_decider.decide(self)
    
    def _enter_attack(self, enemy): self.attack_until_tick = max(self.attack_until_tick, self.current_tick + self.attack_commit_ticks)

    def _enter_escape(self, enemy):
        was_active = (self.current_tick <= self.escape_until_tick)
        self.escape_until_tick = max(self.escape_until_tick, self.current_tick + self.escape_commit_ticks)
        if not was_active or self.escape_target_cell is None: self._start_escape(enemy) 

    def _select_escape_goal_cell(self, enemy_cell, max_candidates=15):
        if enemy_cell is None: return None
        ex, ey = enemy_cell
        
        # Directly filter the virtual map for safe sub-tiles
        safe_cells = []
        for (cx, cy), data in self.virtual_map.items():
            if data["type"] in [1, 6]: 
                safe_cells.append((cx, cy))
                
        if not safe_cells: return None

        safe_cells.sort(key=lambda c: (c[0] - ex) ** 2 + (c[1] - ey) ** 2, reverse=True)
        start_cell = self._cell_from_xy(float(self.dynamic_info.get("position", {}).get("x", 0.0)), float(self.dynamic_info.get("position", {}).get("y", 0.0)))

        for goal in safe_cells[:max_candidates]:
            path = a_star(self, None, start_cell, goal)
            if path and len(path) >= 2: return goal
        return None
    
    def _start_escape(self, enemy):
        self.escape_until_tick = max(self.escape_until_tick, self.current_tick + self.escape_commit_ticks)
        enemy_cell = self._cell_from_xy(float(self._get(self._get(enemy, "position", {}), "x", 0.0)), float(self._get(self._get(enemy, "position", {}), "y", 0.0)))
        
        goal = self._select_escape_goal_cell(enemy_cell)
        if goal is not None:
            self.escape_target_cell = goal
            self.escape_target_last_pick_tick = self.current_tick
        return enemy_cell

    def _reload_ready(self) -> bool:
        rt = self.dynamic_info.get("reload_timer", self.dynamic_info.get("current_reload_progress", 0))
        try: return int(rt) <= 0
        except: return True

    def _can_fire_at_enemy_with_range(self, enemy, ammo_name: str, aim_tolerance_deg=5.0) -> bool:
        if enemy is None or not self._reload_ready(): return False
        if float(self._get(enemy, "distance", 1e9)) > self._ammo_range_world(ammo_name): return False

        my_pos, enemy_pos = self.dynamic_info.get("position", {}), self._get(enemy, "position", {})
        mx, my = float(self._get(my_pos, "x", 0.0)), float(self._get(my_pos, "y", 0.0))
        ex, ey = float(self._get(enemy_pos, "x", 0.0)), float(self._get(enemy_pos, "y", 0.0))

        desired_abs = (math.degrees(math.atan2(ey - my, ex - mx)) + 360.0) % 360.0
        barrel_abs = (float(self.dynamic_info.get("heading", 0.0)) + float(self.dynamic_info.get("barrel_angle", 0.0))) % 360.0
        return abs(self._angle_diff(desired_abs, barrel_abs)) <= aim_tolerance_deg
    
    def _closest_visible_powerup(self):
        powerups = self.dynamic_info.get("visible_powerups", [])
        if not powerups: return None
        my_pos = self.dynamic_info.get("position", {})
        mx, my = float(my_pos.get("x", 0.0)), float(my_pos.get("y", 0.0))
        def dist2(p): return (float(p.get("position", {}).get("x", 0.0)) - mx) ** 2 + (float(p.get("position", {}).get("y", 0.0)) - my) ** 2
        pu = min(powerups, key=dist2)
        return pu, (float(pu.get("position", {}).get("x", 0.0)), float(pu.get("position", {}).get("y", 0.0))), dist2(pu)

    def _is_in_vision(self, x: float, y: float) -> bool:
        pos = self.dynamic_info.get("position", {})
        mx, my = float(pos.get("x", 0.0)), float(pos.get("y", 0.0))
        vr, va = float(self.static_info.get("vision_range", 0.0)), float(self.static_info.get("vision_angle", 0.0))
        if vr > 0.0 and math.hypot(x - mx, y - my) > vr: return False
        return abs(self._angle_diff((math.degrees(math.atan2(y - my, x - mx)) + 360.0) % 360.0, float(self.dynamic_info.get("heading", 0.0)))) <= (va * 0.5)
    
    def _is_target_powerup_visible(self, target_cell) -> bool:
        if target_cell is None: return False
        for p in (self.dynamic_info.get("visible_powerups", []) or []):
            if self._cell_from_xy(float(p.get("position", {}).get("x", 0.0)), float(p.get("position", {}).get("y", 0.0))) == target_cell: return True
        return False
    
    def _select_powerup_goal_cell(self):
        if self._is_target_powerup_visible(self.powerup_target_cell): self.powerup_target_last_seen_tick = self.current_tick
        self._maybe_forget_powerup_target()
        closest = self._closest_visible_powerup()
        if closest is None: return self.powerup_target_cell

        _, (px, py), d2_new = closest
        new_cell = self._cell_from_xy(px, py)

        if self.powerup_target_cell is None or new_cell == self.powerup_target_cell:
            self.powerup_target_cell = new_cell
            self.powerup_target_last_seen_tick = self.powerup_target_acquired_tick = self.current_tick
            return self.powerup_target_cell

        cx, cy = self._cell_center(self.powerup_target_cell)
        d2_cur = (cx - float(self.dynamic_info.get("position", {}).get("x", 0.0))) ** 2 + (cy - float(self.dynamic_info.get("position", {}).get("y", 0.0))) ** 2

        if (self.current_tick - self.powerup_target_acquired_tick) >= self.powerup_commit_ticks and (d2_new <= d2_cur * (self.powerup_switch_ratio ** 2)):
            self.powerup_target_cell = new_cell
            self.powerup_target_last_seen_tick = self.powerup_target_acquired_tick = self.current_tick

        return self.powerup_target_cell
    
    def _path_cost(self, path, node_by_cell):
        if not path or len(path) < 2: return float("inf")
        cost = 0.0
        for cell in path[1:]:
            n = node_by_cell.get(cell)
            if n is None or n.blocked: return float("inf")  
            cost += float(n.dmg)
        return cost
    
    def _plan_best_path(self, nodes, targets, max_targets=3):
        node_by_cell = {n.cell: n for n in nodes}
        start_cell = self._cell_from_xy(float(self.dynamic_info.get("position", {}).get("x", 0.0)), float(self.dynamic_info.get("position", {}).get("y", 0.0)))
        best_path, best_goal, best_cost = None, None, float("inf")

        for goal in targets[:max_targets]:
            path = a_star(self, nodes, start_cell, goal, max_iterations=2000)
            if not path: continue
            
            c = self._path_cost(path, node_by_cell)
            if c < best_cost:
                best_cost, best_path, best_goal = c, path, goal

        return best_path, best_goal, best_cost

    def get_action(self, current_tick: int, my_tank_status: Dict[str, Any], sensor_data: Dict[str, Any], enemies_remaining: int) -> ActionCommand:
        self.current_tick = current_tick
        self.enemies_remaining = enemies_remaining
        update_internal_state(self, my_tank_status, sensor_data)
        
        #if self.current_tick > 0 and self.current_tick % 250 == 0: 
            #self._save_map_plot()

        pos = self.dynamic_info.get("position") or {}
        px, py = float(pos.get("x", 0.0)), float(pos.get("y", 0.0))
        heading_rad = math.radians(float(self.dynamic_info.get("heading", 0.0)))
        top_speed = float(self.static_info.get("top_speed", 1.0))
        heading_spin = float(self.static_info.get("heading_spin_rate", 2.0))
        
        # STUCK / WALL BOUNCE DETECTION
        if self.last_world_pos:
            lx, ly = self.last_world_pos
            dist_moved = math.hypot(px - lx, py - ly)
            
            # 1. Track immobility mathematically
            # CRITICAL FIX: Only increment stuck counter if we ACTUALLY commanded movement.
            # If the tank is pivoting in place (speed == 0), it is not stuck!
            if abs(self.last_commanded_speed) > 0.01:
                if dist_moved < self.MIN_MOVE_EPS: 
                    self.no_move_ticks += 1
                else: 
                    self.no_move_ticks = 0
            else:
                # Tank is legally rotating. Reset to avoid accumulating false positives over many turns.
                self.no_move_ticks = 0

            # Physics engine clip prevention
            if dist_moved > (top_speed * 1.2) and self.last_commanded_speed > 0:
                front_cell = self._cell_from_xy(lx + math.cos(heading_rad) * self.CELL_SIZE * 1.5, ly + math.sin(heading_rad) * self.CELL_SIZE * 1.5)
                if not hasattr(self, "virtual_map"): self.virtual_map = {}
                if self.virtual_map.get(front_cell, {}).get("type", 0) not in [1, 2, 4, 5]:
                    self.virtual_map[front_cell] = {"type": 3, "tick": self.current_tick}
                self.force_change_goal, self.path_to_follow, self.current_goal_cell = True, None, None

            # 2. Trigger Deterministic Anti-Stuck Reverse Protocol
            if self.no_move_ticks > 150:
                self.unstuck_until_tick = self.current_tick + 30  # Force reverse for 30 ticks
                self.unstuck_turn_dir = random.choice([-1.0, 1.0]) # Lock a turn direction for the arc
                self.no_move_ticks = 0
                self.force_change_goal = True # Ensure a new path is calculated when reverse ends

        # ---------------------------------------------------------
        # DE-INDENTED SECTION: Must execute every tick
        # ---------------------------------------------------------
        # 3. Process standard strategic action
        action = self._process_action()

        # 4. Hazard-Aware Kinematic Override if Anti-Stuck is active
        if self.current_tick <= getattr(self, "unstuck_until_tick", -10_000):
            # Calculate the coordinate 30 units directly behind the tank
            back_x = max(0.0, min(199.0, px - math.cos(heading_rad) * self.CELL_SIZE * 3.0))
            back_y = max(0.0, min(199.0, py - math.sin(heading_rad) * self.CELL_SIZE * 3.0))
            back_cell = self._cell_from_xy(back_x, back_y)
            back_type = self.virtual_map.get(back_cell, {}).get("type", 0)
            
            # Prevent reversing into walls or the newly expanded potholes
            if back_type in [3, 5]: 
                action.move_speed = 0.0 # Halt reverse to prevent damage/clipping
                action.heading_rotation_angle = heading_spin # Spin in place to find a new angle
            else:
                action.move_speed = -top_speed  # Safe to reverse
                action.heading_rotation_angle = heading_spin * getattr(self, "unstuck_turn_dir", 1.0)
                
            action.should_fire = False

        self.last_world_pos, self.last_commanded_speed = (px, py), action.move_speed
        
        #if self.current_tick < 20 or self.current_tick % 60 == 0: 
            #save_state_to_file(self)
            
        return action
     
    def _process_action(self) -> ActionCommand:
        # 1. Update perception and target memory
        enemy_now = self._update_attack_memory()
        self._maybe_forget_enemy_target()
        
        # 2. Check current mode commitments
        in_attack = self._should_stay_in_attack()
        in_escape = self._should_stay_in_escape()

        # 3. Fuzzy Logic Decision Tree
        if enemy_now is not None:
            # Re-evaluate if we should switch between Attack and Escape
            if in_attack or in_escape:
                if self._maybe_reconsider_combat_mode(enemy_now):
                    in_attack = self._should_stay_in_attack() 
                    in_escape = self._should_stay_in_escape() 
            
            # If not committed to a mode, pick one based on HP/Ammo
            if not in_attack and not in_escape:
                decision = self._choose_combat_mode(enemy_now)
                # CRITICAL: Force 'power_up' status to 'search' to maintain map pressure
                if decision == "power_up":
                    decision = "search"
                self._commit_mode(decision, enemy_now)

        # 4. Final Mode Assignment
        if self._should_stay_in_escape():
            self.mode = "escape"
        elif self._should_stay_in_attack():
            self.mode = "attack"
        else:
            self.mode = "search"

        # 5. Execute Action based on Final Mode
        if self.mode == "search":
            self._update_macro_goal()
            hull_rot, move_speed = self.Follow_Path_With_Modifiers()
            return ActionCommand(
                barrel_rotation_angle=scan_strategy(self), 
                heading_rotation_angle=hull_rot, 
                move_speed=move_speed, 
                should_fire=False, 
                ammo_to_load=random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"])
            )

        if self.mode == "attack": 
            return mode_attack(self, enemy_now)
            
        if self.mode == "escape": 
            return mode_escape(self, enemy_now)
            
        # Failsafe: stop and scan
        return ActionCommand(barrel_rotation_angle=scan_strategy(self))    
    
    
    def _angle_diff(self, target_deg: float, current_deg: float) -> float: return (target_deg - current_deg + 180) % 360 - 180
    def _clamp(self, x: float, lo: float, hi: float) -> float: return max(lo, min(hi, x))
    
        
    def _select_target_point(self, forbid_cells=None):
        my_x, my_y = float(self.dynamic_info.get("position", {}).get("x", 0.0)), float(self.dynamic_info.get("position", {}).get("y", 0.0))
        
        candidates = []
        attempts = 0
        
        mac_x, mac_y = getattr(self, "macro_target_world", (100.0, 100.0))
        mac_cell_x, mac_cell_y = self._cell_from_xy(mac_x, mac_y)
        
        while len(candidates) < 15 and attempts < 150:
            attempts += 1
            
            if random.random() < 0.8:
                cx = int(random.gauss(mac_cell_x, 3)) 
                cy = int(random.gauss(mac_cell_y, 3))
            else:
                cx, cy = random.randint(0, 19), random.randint(0, 19)
                
            cx = max(0, min(19, cx))
            cy = max(0, min(19, cy))
            
            if forbid_cells and (cx, cy) in forbid_cells: continue
            
            if self.virtual_map.get((cx, cy), {}).get("type") in [0, 1, 4, 6]:
                candidates.append((cx, cy))
        
        if not candidates: return []
        
        def sort_key(cell):
            wx, wy = self._cell_center(cell)
            dist_to_me = math.hypot(wx - my_x, wy - my_y)
            dist_to_macro = math.hypot(wx - mac_x, wy - mac_y)
            
            ctype = self.virtual_map.get(cell, {}).get("type", 0)
            is_known = ctype != 0
            
            if ctype == 4:
                has_fog = any(self.virtual_map.get((cell[0]+dx, cell[1]+dy), {}).get("type", 0) == 0 
                              for dx, dy in [(0,1),(1,0),(0,-1),(-1,0), (1,1), (-1,-1), (1,-1), (-1,1)])
                exploration_penalty = -500.0 if has_fog else 1000.0 
            else:
                # OVERRIDE: Increase the drive to explore unknown cells significantly 
                # to overpower any pathfinding resistance from a_star wall gradients
                exploration_penalty = 2000.0 if is_known else -1000.0
            
            return (dist_to_macro * 3.0) + dist_to_me + exploration_penalty
            
        candidates.sort(key=sort_key)
        return candidates



    def Follow_Path_With_Modifiers(self, override_goal_cell=None):
        force = False
        
        # 1. Lookahead Obstacle Detection
        if self.path_to_follow and self.path_index < len(self.path_to_follow):
            for i in range(self.path_index, min(len(self.path_to_follow), self.path_index + 15)):
                ctype = self.virtual_map.get(self.path_to_follow[i], {}).get("type", 0)
                # Omit Type 5 (Danger) from this list. A* evaluates potholes mathematically. 
                # Aborting paths here due to potholes causes infinite calculation loops.
                if ctype == 3: 
                    force = True
                    break
                    
        # 2. Standard completion/failure triggers
        if self.path_stuck_ticks > MAX_PATH_STUCK_TICKS or self.path_to_follow is None or self.path_index >= len(self.path_to_follow) - 1:
            force = True
        if override_goal_cell is not None and override_goal_cell != self.current_goal_cell:
            force = True
        if getattr(self, "force_change_goal", False):
            force = True

        # 3. Execution & Failsafe
        if force:
            new_path, new_goal, new_cost, _ = self._Find_Target_and_Find_Path(override_goal_cell)
            if new_path and new_goal is not None:
                self.path_to_follow, self.current_goal_cell, self.current_path_cost = new_path, new_goal, new_cost
                self.path_index, self.path_stuck_ticks, self.force_change_goal = 0, 0, False
            else:
                self.path_to_follow = None
                self.current_goal_cell = None
                self.force_change_goal = True 

        hull_rot, move_speed = self._FollowPath() if self.path_to_follow is not None else (0.0, 0.0)
        self.debug_goal_cell, self.debug_path, self.debug_path_index = self.current_goal_cell, self.path_to_follow, self.path_index
        return hull_rot, move_speed


    def _Find_Target_and_Find_Path(self, override_goal_cell=None, forbid_cells=None):
        targets = [override_goal_cell] if override_goal_cell is not None else self._select_target_point(forbid_cells)
        if not targets: return None, None, None, None
        
        start_cell = self._cell_from_xy(float(self.dynamic_info.get("position", {}).get("x", 0.0)), float(self.dynamic_info.get("position", {}).get("y", 0.0)))
        
        best_path, best_goal, best_cost = None, None, float("inf")
        # Evaluate up to 10 candidates to bypass wall-clipping path failures
        for goal in targets[:10]: 
            path = a_star(self, None, start_cell, goal, max_iterations=6000)
            if path:
                cost = len(path)
                if cost < best_cost:
                    best_cost, best_path, best_goal = cost, path, goal
                    
        return best_path, best_goal, best_cost, None



    def _update_macro_goal(self):
        mx, my = float(self._get(self.dynamic_info.get("position", {}), "x", 0.0)), float(self._get(self.dynamic_info.get("position", {}), "y", 0.0))
        
        # 1. Trigger conditions for replanning the macro sector
        needs_new_goal = False
        if not hasattr(self, "macro_target_world") or self.macro_target_world[0] is None:
            needs_new_goal = True
        elif (self.current_tick - getattr(self, "macro_target_tick", 0)) > 400: # Standard refresh rate
            needs_new_goal = True
        elif math.hypot(self.macro_target_world[0] - mx, self.macro_target_world[1] - my) < 20.0:
            needs_new_goal = True
        elif self.path_to_follow is None and self.no_move_ticks > 15:
            needs_new_goal = True
            
        if not needs_new_goal:
            return

        # 2. Quadrant Analysis (4 quadrants of 10x10 cells for a 20x20 map)
        quadrant_explored = {0: 0, 1: 0, 2: 0, 3: 0}
        
        for (cx, cy), data in self.virtual_map.items():
            if data.get("type", 0) != 0:
                qx = 1 if cx >= 10 else 0
                qy = 2 if cy >= 10 else 0
                quadrant_explored[qx + qy] += 1
                
        target_quadrant = None
        best_ratio = 1.0 
        
        # 3. Evaluate Thresholds (100 cells per quadrant)
        for q, explored_count in quadrant_explored.items():
            ratio = explored_count / 100.0 
            if ratio < 0.70 and ratio < best_ratio:
                best_ratio = ratio
                target_quadrant = q
                
        # 4. Assign Macro Target World Coordinates
        if target_quadrant is None:
            # Map is largely explored, patrol center
            self.macro_target_world = (100.0, 100.0)
        else:
            qx_offset = (target_quadrant % 2) * 10
            qy_offset = (target_quadrant // 2) * 10
            
            found = False
            for _ in range(30):
                rx = random.randint(qx_offset, qx_offset + 9)
                ry = random.randint(qy_offset, qy_offset + 9)
                # Specifically target fog-of-war
                if self.virtual_map.get((rx, ry), {}).get("type", 0) == 0:
                    self.macro_target_world = self._cell_center((rx, ry))
                    found = True
                    break
                    
            if not found:
                self.macro_target_world = self._cell_center((qx_offset + 5, qy_offset + 5))

        self.macro_target_tick = self.current_tick

  
    def _FollowPath(self):
        if not self.path_to_follow or self.path_index >= len(self.path_to_follow):
            self.path_to_follow, self.current_goal_cell, self.current_path_cost = None, None, None
            return 0.0, 0.0
        
        my_pos = self.dynamic_info.get("position", {})
        my_x, my_y = float(my_pos.get("x", 0.0)), float(my_pos.get("y", 0.0))
        
        # 1. Target Cell Acquisition & Distance Check
        target_cell = self.path_to_follow[self.path_index]
        tx, ty = self._cell_center(target_cell)
        
        dist_sq = (tx - my_x)**2 + (ty - my_y)**2
        
        # INCREASED ACCEPTANCE RADIUS: 16.0 units (4.0^2 = 16.0).
        # Prevents "waypoint orbiting" where the tank overshoots a tiny radius,
        # stops, spins 180 degrees, and gets locked in an infinite correction loop.
        if dist_sq <= 16.0:
            self.path_index += 1
            self.path_stuck_ticks = 0
            
            if self.path_index >= len(self.path_to_follow):
                self.path_to_follow = None
                return 0.0, 0.0
                
            # Update coordinate to the newly acquired next waypoint
            target_cell = self.path_to_follow[self.path_index]
            tx, ty = self._cell_center(target_cell)

        # 2. Path Stuck Failsafe
        self.path_stuck_ticks += 1
        if self.path_stuck_ticks >= MAX_PATH_STUCK_TICKS:
            self.path_to_follow = None
            return 0.0, 0.0
            
        # 3. Kinematic Vector Calculation
        desired_angle = math.degrees(math.atan2(ty - my_y, tx - my_x))
        current_heading = float(self.dynamic_info.get("heading", 0.0))
        err = self._angle_diff(desired_angle, current_heading)
        
        heading_spin = float(self.static_info.get("heading_spin_rate", 2.0))
        top_speed = float(self.static_info.get("top_speed", 1.0))
        
        # 4. STRICT ROBOTIC DEADBAND
        if abs(err) > 3.0:
            # ROTATION STATE: Halt translation entirely to pivot in place.
            move_speed = 0.0
            hull_rot = self._clamp(err, -heading_spin, heading_spin)
        else:
            # TRANSLATION STATE: Push forward cleanly.
            move_speed = top_speed
            # ZERO-ROTATION LOCK: Prevent micro-vibrations across the 180-degree boundary
            hull_rot = 0.0 
            
        return hull_rot, move_speed
            
    
    

    def _closest_visible_enemy(self):
        visible_enemies = self.dynamic_info.get("visible_enemies", [])
        return min(visible_enemies, key=lambda enemy: self._get(enemy, "distance", 999999.0)) if visible_enemies else None
    
    def _update_attack_memory(self):
        now = self.current_tick
        enemy = self._closest_visible_enemy()
        if enemy is None: return None
        self.enemy_target_id = self._get(enemy, "id", None)
        self.enemy_target_cell = self._cell_from_xy(float(self._get(self._get(enemy, "position", {}), "x", 0.0)), float(self._get(self._get(enemy, "position", {}), "y", 0.0)))
        self.enemy_target_last_seen_tick = now
        return enemy
        
    def _should_stay_in_escape(self): return self.current_tick <= self.escape_until_tick
    def _should_stay_in_attack(self): return self.current_tick <= self.attack_until_tick

    def _maybe_forget_enemy_target(self):
        if self.enemy_target_id and (self.current_tick - self.enemy_target_last_seen_tick) >= self.enemy_forget_after:
            self.enemy_target_id, self.enemy_target_cell = None, None
            
    def _maybe_forget_powerup_target(self):
        if self.powerup_target_cell is None: return
        if self._is_in_vision(*self._cell_center(self.powerup_target_cell)) and (self.current_tick - self.powerup_target_last_seen_tick) >= self.powerup_forget_after:
            self.powerup_target_cell = None
                
    def _ammo_range_world(self, ammo_name): return {"HEAVY": 25.0, "LIGHT": 50.0, "LONG_DISTANCE": 100.0}.get(ammo_name.upper(), 0.0)

    def _ammo_inventory(self) -> dict:
        inv = {"HEAVY": 0, "LIGHT": 0, "LONG_DISTANCE": 0}
        ammo_raw = self.dynamic_info.get("ammo")
        if isinstance(ammo_raw, dict):
            for ammo_name, slot in ammo_raw.items(): inv[str(ammo_name).upper()] = int(slot.get("count", 0) if isinstance(slot, dict) else getattr(slot, "count", 0) or 0)
        return inv

    def _loaded_ammo_name(self) -> str | None:
        ammo_loaded = self.dynamic_info.get("ammo_loaded", None)
        return getattr(ammo_loaded, "name", str(ammo_loaded)).upper() if ammo_loaded else None

    def _aim_barrel_at_enemy(self, enemy, aim_eps_deg=2.0):
        if enemy is None: return 0.0
        desired_abs_angle = (math.degrees(math.atan2(float(self._get(self._get(enemy, "position", {}), "y", 0.0)) - float(self._get(self.dynamic_info.get("position"), "y", 0.0)), float(self._get(self._get(enemy, "position", {}), "x", 0.0)) - float(self._get(self.dynamic_info.get("position"), "x", 0.0)))) + 360.0) % 360.0
        error_angle = self._angle_diff(desired_abs_angle, (float(self.dynamic_info.get("heading", 0.0)) + float(self.dynamic_info.get("barrel_angle", 0.0))) % 360.0)
        return 0.0 if abs(error_angle) <= aim_eps_deg else self._clamp(error_angle, -float(self.static_info.get("barrel_spin_rate", 0.0)), float(self.static_info.get("barrel_spin_rate", 0.0)))


    def destroy(self):
        self.is_destroyed = True
        logging.info(f"[{self.name}] Tank destroyed!")

    def end(self, damage_dealt: float, tanks_killed: int):
        logging.info(f"[{self.name}] Game ended! Damage: {damage_dealt}, Kills: {tanks_killed}")

app = FastAPI(title="Random Test Agent", description="Random walking and shooting agent for testing", version="1.0.0")
agent = RandomAgent()

@app.get("/")
async def root(): return {"message": f"Agent {agent.name} is running", "destroyed": agent.is_destroyed}

@app.post("/agent/action", response_model=ActionCommand)
async def get_action(payload: Dict[str, Any] = Body(...)):
    try: return agent.get_action(current_tick=payload.get('current_tick', 0), my_tank_status=payload.get('my_tank_status', {}), sensor_data=payload.get('sensor_data', {}), enemies_remaining=payload.get('enemies_remaining', 0))
    except Exception as e:
        logging.exception("Exception in /agent/action")
        raise HTTPException(status_code=500, detail=str(payload))
    
@app.post("/agent/destroy", status_code=204)
async def destroy(): agent.destroy()

@app.post("/agent/end", status_code=204)
async def end(payload: Dict[str, Any] = Body(...)): agent.end(damage_dealt=payload.get('damage_dealt', 0.0), tanks_killed=payload.get('tanks_killed', 0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run random test agent")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8001, help="Port number")
    parser.add_argument("--name", type=str, default=None, help="Agent name")
    parser.add_argument("--modifier", type=str, default=None, help="Modifier")    
    args = parser.parse_args()
    
    agent.name = args.name if args.name else f"RandomBot_{args.port}"
    agent.state_dir = os.path.join(os.path.dirname(__file__), "agent_states")
    os.makedirs(agent.state_dir, exist_ok=True)
    agent.state_path = os.path.join(agent.state_dir, f"agent_state_{agent.name}.json")
        
    if args.modifier != None:
        try: agent.movement_list = [int(element) for element in agent.modifier.split("_")]
        except: pass

    uvicorn.run(app, host=args.host, port=args.port)