import argparse
import sys
import os
import logging

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
controller_dir = os.path.join(os.path.dirname(current_dir), 'FRAKCJA_SILNIKA', 'controller')
sys.path.insert(0, controller_dir)

parent_dir = os.path.join(os.path.dirname(current_dir), 'FRAKCJA_SILNIKA')
sys.path.insert(0, parent_dir)

from typing import Dict, Any
from fastapi import FastAPI, Body

import uvicorn

from src.agent import Agent007, ActionCommand
from src.genetic import ANFIS_Specimen

# ============================================================================
# FASTAPI SERVER
# ============================================================================


def configure_logging(level: str = "INFO", log_file: str = None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


logger = logging.getLogger("agent007.server")

app = FastAPI(
    title="Agent 007",
    description="Her Majesty's Suspicious Agent",
    version="1.0.0"
)

# Global agent instance
agent = Agent007(name="Agent 007")


@app.get("/")
async def root():
    logger.debug("Health check called")
    return {"message": f"{agent.name} is running", "destroyed": agent.is_destroyed}


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
    logger.info("Received /agent/destroy")
    agent.destroy()


@app.post("/agent/end", status_code=204)
async def end(payload: Dict[str, Any] = Body(...)):
    """Called when the game ends."""
    logger.info(
        "Received /agent/end damage_dealt=%s tanks_killed=%s",
        payload.get('damage_dealt', 0.0),
        payload.get('tanks_killed', 0),
    )
    agent.end(
        damage_dealt=payload.get('damage_dealt', 0.0),
        tanks_killed=payload.get('tanks_killed', 0)
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Agent 007")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8001, help="Port number")
    parser.add_argument("--name", type=str, default=None, help="Agent name")
    parser.add_argument("--genes", type=str, default=None, help="Path to genes JSON file")
    parser.add_argument("--train", action="store_true", help="Enable Training")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to write logs to file",
    )
    parser.add_argument(
        "--log-actions",
        action="store_true",
        help="Log action decision for every tick",
    )
    args = parser.parse_args()

    configure_logging(args.log_level, args.log_file)
    
    if args.name:
        agent.name = args.name
    else:
        agent.name = f"Agent007_{args.port}"
    
    default_genes_path = os.path.join(current_dir, "genes_007.json")
    genes_path = args.genes
    if genes_path is None and os.path.isfile(default_genes_path):
        genes_path = default_genes_path

    if genes_path:
        specimen = ANFIS_Specimen.load_from_file(genes_path)
        agent.load_specimen(specimen)

    if args.train:
        agent.set_training_mode(True)
    
    agent.log_actions = args.log_actions

    logger.info(
        "Starting %s on %s:%s train=%s log_file=%s log_actions=%s",
        agent.name,
        args.host,
        args.port,
        args.train,
        args.log_file,
        args.log_actions,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
