"""
Script to start multiple random agents for testing.
Uruchamia wiele agentów losowych do testów.

Usage:
    python run_agents.py         # Start 10 agents (5 per team)
    python run_agents.py --count 4  # Start 4 agents
"""

import subprocess
import sys
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description="Start multiple test agents")
    parser.add_argument("--count", type=int, default=10, help="Number of agents to start")
    parser.add_argument("--base-port", type=int, default=8001, help="Base port number")
    args = parser.parse_args()
    
    processes = []
    
    print(f"Starting {args.count} random agents...")
    
    for i in range(args.count):
        port = args.base_port + i
        name = f"Bot_{i+1}"
        
        # Start agent in new process
        cmd = [sys.executable, "random_agent.py", "--port", str(port), "--name", name]
        
        # Use CREATE_NEW_CONSOLE on Windows to open separate windows
        if sys.platform == "win32":
            proc = subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            proc = subprocess.Popen(cmd)
        
        processes.append(proc)
        print(f"  Started {name} on port {port}")
        time.sleep(0.2)  # Small delay between starts
    
    print(f"\nAll {args.count} agents started!")
    print("Ports:", ", ".join(str(args.base_port + i) for i in range(args.count)))
    print("\nNow run the game engine:")
    print("  python run_game.py --headless --log-level INFO")
    print("\nPress Ctrl+C to stop all agents...")
    
    try:
        # Wait for all processes
        for proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("\nStopping all agents...")
        for proc in processes:
            proc.terminate()


if __name__ == "__main__":
    main()
