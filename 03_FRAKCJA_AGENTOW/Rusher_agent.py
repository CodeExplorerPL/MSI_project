"""
Smart Rusher Agent (v7 - True Random Patrol)
- Improved Randomness: Uses system time/PID seeding to prevent identical paths.
- Minimum Patrol Distance: Rejects patrol points that are too close (forces cross-map travel).
- Timeout: Increased to 500 ticks.
- Universal Ammo & Fallback Logic retained.
"""

import random
import argparse
import sys
import os
import math
import time
from typing import Dict, Any, Optional

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
controller_dir = os.path.join(os.path.dirname(current_dir), '02_FRAKCJA_SILNIKA', 'controller')
if os.path.exists(controller_dir):
    sys.path.insert(0, controller_dir)
    sys.path.insert(0, os.path.join(os.path.dirname(current_dir), '02_FRAKCJA_SILNIKA'))

from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn

# ============================================================================
# CONFIGURATION
# ============================================================================

AMMO_STATS = {
    "HEAVY": {"Range": 25, "Value": -40},
    "LIGHT": {"Range": 50, "Value": -20},
    "LONG_DISTANCE": {"Range": 100, "Value": -25}
}

class ActionCommand(BaseModel):
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: Optional[str] = None
    should_fire: bool = False

# ============================================================================
# AGENT LOGIC
# ============================================================================

class RusherAgent:
    def __init__(self, name="Rusher_RNG"):
        self.name = name
        self.is_destroyed = False
        
        # --- SEEDING RANDOMNESS ---
        # Combine current time and memory address/name to ensure unique seeds per agent
        random.seed(time.time() + hash(self.name))
        
        # --- STATE ---
        self.patrol_destination = None
        self.last_known_enemy_pos = None
        self.patrol_timer = 0
        
        # --- SETTINGS ---
        self.PATROL_TIMEOUT = 500      # Increased from 300
        self.MIN_PATROL_DIST = 400.0   # Don't pick points closer than this
        self.MAP_SIZE = 1200.0         
        
        self.HULL_TURN_SPEED = 3.0     
        self.TURRET_TURN_SPEED = 10.0
        
        print(f"[{self.name}] Online. Patrol Timeout: 500. RNG Seeding Applied.")

    def _normalize_angle(self, angle):
        return (angle + 180) % 360 - 180

    def _get_ammo_counts(self, tank_status: Dict[str, Any]) -> Dict[str, int]:
        counts = {"LIGHT": 0, "HEAVY": 0, "LONG_DISTANCE": 0}
        inventory = tank_status.get('ammo', {})
        for k, v in inventory.items():
            key_str = str(k).upper()
            val = v.get('count', 0) if isinstance(v, dict) else getattr(v, 'count', 0)
            if "LONG" in key_str or "DIST" in key_str: counts["LONG_DISTANCE"] = val
            elif "HEAVY" in key_str: counts["HEAVY"] = val
            elif "LIGHT" in key_str: counts["LIGHT"] = val
        return counts

    def _select_best_valid_ammo(self, dist: float, ammo: Dict[str, int]) -> str:
        candidates = []
        if dist <= AMMO_STATS["HEAVY"]["Range"] and ammo["HEAVY"] > 0: candidates.append("HEAVY")
        if dist <= AMMO_STATS["LIGHT"]["Range"] and ammo["LIGHT"] > 0: candidates.append("LIGHT")
        if dist <= AMMO_STATS["LONG_DISTANCE"]["Range"] and ammo["LONG_DISTANCE"] > 0: candidates.append("LONG_DISTANCE")
        
        if "HEAVY" in candidates: return "HEAVY"
        if "LIGHT" in candidates: return "LIGHT"
        if "LONG_DISTANCE" in candidates: return "LONG_DISTANCE"
        
        # Fallback
        if ammo["LONG_DISTANCE"] > 0: return "LONG_DISTANCE"
        if ammo["LIGHT"] > 0: return "LIGHT"
        return "HEAVY"

    def _generate_new_patrol_point(self, my_x, my_y):
        """Generates a point that is explicitly FAR away from current position."""
        for _ in range(10): # Try 10 times to find a far point
            rx = random.uniform(50, self.MAP_SIZE)
            ry = random.uniform(50, self.MAP_SIZE)
            
            dist = math.hypot(rx - my_x, ry - my_y)
            if dist > self.MIN_PATROL_DIST:
                return (rx, ry)
        
        # If we fail (corner case), just return the last random point
        return (rx, ry)

    def get_action(self, current_tick: int, my_tank_status: Dict[str, Any], sensor_data: Dict[str, Any], enemies_remaining: int) -> ActionCommand:
        
        # 1. PARSE
        my_pos = my_tank_status.get('position', {'x': 0, 'y': 0})
        mx, my = float(my_pos['x']), float(my_pos['y'])
        my_heading = float(my_tank_status.get('heading', 0))
        my_barrel_rel = float(my_tank_status.get('barrel_angle', 0))
        my_team = my_tank_status.get('_team')
        top_speed = float(my_tank_status.get('_top_speed', 10.0))
        reload_timer = float(my_tank_status.get('_reload_timer', 0))

        # 2. TARGETING
        all_tanks = sensor_data.get('seen_tanks', [])
        enemies = [t for t in all_tanks if t.get('team') != my_team]
        
        target_enemy = None
        if enemies:
            target_enemy = min(enemies, key=lambda t: t.get('distance', 9999))
            self.last_known_enemy_pos = (float(target_enemy['position']['x']), float(target_enemy['position']['y']))
            self.patrol_destination = None

        # 3. MOVEMENT
        move_speed = 0.0
        hull_rot = 0.0
        target_coords = None
        is_combat = False

        if target_enemy:
            target_coords = self.last_known_enemy_pos
            is_combat = True
        elif self.last_known_enemy_pos:
            target_coords = self.last_known_enemy_pos
            if math.hypot(target_coords[0] - mx, target_coords[1] - my) < 30.0:
                self.last_known_enemy_pos = None
                target_coords = None
        
        if not target_coords:
            # Check Patrol State
            if not self.patrol_destination or self.patrol_timer <= 0:
                # Generate new point with distance check
                self.patrol_destination = self._generate_new_patrol_point(mx, my)
                self.patrol_timer = self.PATROL_TIMEOUT
                # Debug print to verify diversity
                # print(f"[{self.name}] New Patrol Target: {int(self.patrol_destination[0])}, {int(self.patrol_destination[1])}")

            target_coords = self.patrol_destination
            self.patrol_timer -= 1
            
            # Check Arrival
            if math.hypot(target_coords[0] - mx, target_coords[1] - my) < 40.0:
                self.patrol_destination = None # Trigger new point next tick

        if target_coords:
            tx, ty = target_coords
            desired_heading = math.degrees(math.atan2(ty - my, tx - mx))
            heading_err = self._normalize_angle(desired_heading - my_heading)
            hull_rot = max(-self.HULL_TURN_SPEED, min(self.HULL_TURN_SPEED, heading_err))
            
            if abs(heading_err) < (90.0 if is_combat else 45.0):
                move_speed = top_speed

        # 4. WEAPONS
        barrel_rot = 0.0
        should_fire = False
        ammo_cmd = None

        if target_enemy:
            ex, ey = float(target_enemy['position']['x']), float(target_enemy['position']['y'])
            dist = float(target_enemy.get('distance', 9999))
            
            ammo_counts = self._get_ammo_counts(my_tank_status)
            selected_ammo = self._select_best_valid_ammo(dist, ammo_counts)
            ammo_cmd = selected_ammo
            max_range = AMMO_STATS.get(selected_ammo, {}).get("Range", 50)

            angle_to_enemy = math.degrees(math.atan2(ey - my, ex - mx))
            abs_barrel = (my_heading + my_barrel_rel) % 360
            aim_err = self._normalize_angle(angle_to_enemy - abs_barrel)
            
            needed_correction = self._normalize_angle(aim_err - hull_rot)
            barrel_rot = max(-self.TURRET_TURN_SPEED, min(self.TURRET_TURN_SPEED, needed_correction))
            
            current_tolerance = 30.0 if dist < 30.0 else 10.0
            
            if abs(aim_err) < current_tolerance and reload_timer <= 0:
                if dist <= max_range:
                    should_fire = True
                
        else:
            barrel_rot = self.TURRET_TURN_SPEED

        return ActionCommand(
            barrel_rotation_angle=barrel_rot,
            heading_rotation_angle=hull_rot,
            move_speed=move_speed,
            ammo_to_load=ammo_cmd,
            should_fire=should_fire
        )

    def destroy(self):
        self.is_destroyed = True
        print(f"[{self.name}] DESTROYED!")

    def end(self, d, k):
        print(f"[{self.name}] END. Dmg: {d}, Kills: {k}")

# ============================================================================
# SERVER
# ============================================================================

app = FastAPI(title="Rusher Agent")
agent = RusherAgent()

@app.get("/")
async def root():
    return {"message": f"Agent {agent.name} Running", "destroyed": agent.is_destroyed}

@app.post("/agent/action", response_model=ActionCommand)
async def action(p: Dict[str, Any] = Body(...)):
    return agent.get_action(
        p.get('current_tick', 0),
        p.get('my_tank_status', {}),
        p.get('sensor_data', {}),
        p.get('enemies_remaining', 0)
    )

@app.post("/agent/destroy")
async def destroy(): agent.destroy()

@app.post("/agent/end")
async def end(p: Dict[str, Any] = Body(...)): agent.end(p.get('damage_dealt', 0), p.get('tanks_killed', 0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    
    agent.name = args.name or f"Rusher_{args.port}"
    print(f"Starting {agent.name} on {args.port}...")
    uvicorn.run(app, host=args.host, port=args.port)