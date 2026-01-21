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

import random
import argparse
import sys
import os

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


# ============================================================================
# RANDOM AGENT LOGIC
# ============================================================================

class RandomAgent:
    """
    Agent with more structured, stateful behavior for testing purposes.
    Drives in one direction for a while, then changes.
    Scans with its turret.
    """
    
    def __init__(self, name: str = "TestBot"):
        self.name = name
        self.is_destroyed = False
        print(f"[{self.name}] Agent initialized")

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
    
    def get_action(
        self, 
        current_tick: int, 
        my_tank_status: Dict[str, Any], 
        sensor_data: Dict[str, Any], 
        enemies_remaining: int
    ) -> ActionCommand:
        """Generate a stateful, predictable action for testing."""
        should_fire = False
        heading_rotation = 0.0
        barrel_rotation = 0.0
        
        if self.aim_timer > 0:
            # --- AIMING PHASE ---
            self.aim_timer -= 1
            
            # Stop all rotation while aiming
            heading_rotation = 0.0
            barrel_rotation = 0.0
            
            # Fire on the last tick of aiming
            if self.aim_timer == 0:
                should_fire = True
        else:
            # --- NORMAL OPERATION PHASE ---

            # --- Hull Rotation Logic ---
            self.heading_timer -= 1
            if self.heading_timer <= 0:
                self.current_heading_rotation = random.choice([-15.0, 0, 15.0])
                self.heading_timer = random.randint(30, 90)
            heading_rotation = self.current_heading_rotation

            # --- Barrel Scanning Logic ---
            barrel_angle = my_tank_status.get("barrel_angle", 0.0)
            if barrel_angle > 45.0:
                self.barrel_scan_direction = -1.0  # Scan left
            elif barrel_angle < -45.0:
                self.barrel_scan_direction = 1.0  # Scan right
            barrel_rotation = self.barrel_rotation_speed * self.barrel_scan_direction

            # --- Shooting Decision ---
            # Decide if we should start aiming
            wants_to_shoot = random.random() < 0.3
            if wants_to_shoot:
                self.aim_timer = 10  # Start aiming for 10 ticks

        # --- Movement Logic (independent of aiming) ---
        self.move_timer -= 1
        if self.move_timer <= 0:
            self.current_move_speed = random.choice([30.0, 30.0, 0.0, -10.0])
            self.move_timer = random.randint(1, 10)
        
        return ActionCommand(
            barrel_rotation_angle=barrel_rotation,
            heading_rotation_angle=heading_rotation,
            move_speed=self.current_move_speed,
            should_fire=should_fire
        )
    
    def destroy(self):
        """Called when tank is destroyed."""
        self.is_destroyed = True
        print(f"[{self.name}] Tank destroyed!")
    
    def end(self, damage_dealt: float, tanks_killed: int):
        """Called when game ends."""
        print(f"[{self.name}] Game ended!")
        print(f"[{self.name}] Damage dealt: {damage_dealt}")
        print(f"[{self.name}] Tanks killed: {tanks_killed}")


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


@app.get("/")
async def root():
    return {"message": f"Agent {agent.name} is running", "destroyed": agent.is_destroyed}


@app.post("/agent/action", response_model=ActionCommand)
async def get_action(payload: Dict[str, Any] = Body(...)):
    """Main endpoint called each tick by the engine."""
    action = agent.get_action(
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
