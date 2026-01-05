"""
Main Game Runner - Entry point for the Tank Battle Game Engine
Główny plik do uruchomienia silnika
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add the backend path to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from backend.engine.game_loop import run_game
from backend.utils.config import GameConfig, game_config
from backend.utils.logger import get_logger, set_log_level


def load_agent_modules(agent_paths: List[str]) -> List[str]:
    """
    Load agent modules from file paths.

    Args:
        agent_paths: List of paths to agent Python files

    Returns:
        List of loaded agent modules
    """
    agents = []

    for agent_path in agent_paths:
        try:
            # TODO: Implement dynamic agent loading
            # This will be implemented when the agent API is finalized
            print(f"Loading agent from: {agent_path}")
            agents.append(agent_path)  # Placeholder
        except Exception as e:
            print(f"Failed to load agent from {agent_path}: {e}")

    return agents


def main():
    """Main entry point for the game."""
    parser = argparse.ArgumentParser(
        description="Tank Battle Game Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_game.py --headless
  python run_game.py --map-seed "test123" --log-level DEBUG
  python run_game.py --agent1 path/to/agent1.py --agent2 path/to/agent2.py
  python run_game.py --quick-test
        """,
    )

    # Game configuration options
    parser.add_argument(
        "--map-seed", type=str, help="Seed for map generation (default: random)"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run game without GUI (faster execution)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    # Agent options
    parser.add_argument("--agent1", type=str, help="Path to team 1 agent module")

    parser.add_argument("--agent2", type=str, help="Path to team 2 agent module")

    parser.add_argument("--agents", nargs="+", help="List of agent module paths")

    # Quick test options
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a quick test game with dummy agents",
    )

    parser.add_argument(
        "--config", type=str, help="Path to custom game configuration file (JSON)"
    )

    # Performance options
    parser.add_argument(
        "--max-ticks", type=int, help="Maximum number of ticks to run (for testing)"
    )

    parser.add_argument(
        "--performance-test", action="store_true", help="Run performance benchmarking"
    )

    args = parser.parse_args()

    # Setup logging
    set_log_level(args.log_level)
    logger = get_logger()

    logger.info("Starting Tank Battle Game Engine")
    logger.info(f"Arguments: {vars(args)}")

    try:
        # Load configuration
        config = game_config

        if args.config:
            # TODO: Load custom configuration from JSON file when needed
            logger.info(f"Loading custom config from: {args.config}")
            # For now, use default config

        # Modify config based on arguments
        if args.max_ticks:
            config.game_rules.sudden_death_tick = args.max_ticks
            logger.info(f"Set max ticks to: {args.max_ticks}")

        # Load agents
        agent_modules = []

        if args.quick_test:
            logger.info("Running quick test with dummy agents")
            # Use dummy agents for testing
            agent_modules = ["dummy_agent_1", "dummy_agent_2"]

        elif args.agents:
            agent_modules = load_agent_modules(args.agents)

        elif args.agent1 and args.agent2:
            agent_modules = load_agent_modules([args.agent1, args.agent2])

        elif args.agent1 or args.agent2:
            logger.warning("Only one agent specified, need at least 2 for a game")
            if args.agent1:
                agent_modules = load_agent_modules([args.agent1, "dummy_agent"])
            else:
                agent_modules = load_agent_modules(["dummy_agent", args.agent2])

        else:
            logger.info("No agents specified, using dummy agents")
            agent_modules = ["dummy_agent_1", "dummy_agent_2"]

        logger.info(f"Loaded {len(agent_modules)} agents")

        # Run the game
        if args.performance_test:
            logger.info("Running performance test...")
            results = run_performance_test(
                config, args.map_seed, agent_modules, args.headless
            )
        else:
            results = run_game(
                config=config,
                map_seed=args.map_seed,
                agent_modules=agent_modules,
                headless=args.headless,
            )

        # Display results
        if results["success"]:
            logger.info("Game completed successfully!")

            if "winner_team" in results:
                if results["winner_team"]:
                    logger.info(f"🏆 Winner: Team {results['winner_team']}")
                else:
                    logger.info("🤝 Game ended in a draw")

            logger.info(f"Total ticks: {results.get('total_ticks', 0)}")

            if args.performance_test:
                display_performance_results(results)

        else:
            logger.error("Game failed!")
            if "error" in results:
                logger.error(f"Error: {results['error']}")
            return 1

    except KeyboardInterrupt:
        logger.info("Game interrupted by user")
        return 0

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1

    logger.info("Game engine shutdown complete")
    return 0


def run_performance_test(config, map_seed, agent_modules, headless) -> dict:
    """Run performance benchmarking."""
    import time

    logger = get_logger()

    # Run multiple games to get average performance
    num_tests = 5
    results = []

    logger.info(f"Running {num_tests} performance test games...")

    for i in range(num_tests):
        logger.info(f"Performance test {i + 1}/{num_tests}")

        start_time = time.time()

        # Use the same seed for consistency
        test_seed = f"{map_seed}_test_{i}" if map_seed else f"perf_test_{i}"

        game_result = run_game(
            config=config,
            map_seed=test_seed,
            agent_modules=agent_modules,
            headless=True,  # Force headless for performance testing
        )

        end_time = time.time()

        if game_result["success"]:
            test_result = {
                "test_number": i + 1,
                "total_time": end_time - start_time,
                "total_ticks": game_result.get("total_ticks", 0),
                "ticks_per_second": game_result.get("total_ticks", 0)
                / (end_time - start_time)
                if (end_time - start_time) > 0
                else 0,
            }
            results.append(test_result)
        else:
            logger.warning(f"Performance test {i + 1} failed")

    # Calculate averages
    if results:
        avg_time = sum(r["total_time"] for r in results) / len(results)
        avg_ticks = sum(r["total_ticks"] for r in results) / len(results)
        avg_tps = sum(r["ticks_per_second"] for r in results) / len(results)

        performance_summary = {
            "success": True,
            "tests_completed": len(results),
            "average_game_time": avg_time,
            "average_ticks": avg_ticks,
            "average_ticks_per_second": avg_tps,
            "individual_results": results,
        }
    else:
        performance_summary = {
            "success": False,
            "error": "No performance tests completed successfully",
        }

    return performance_summary


def display_performance_results(results: dict):
    """Display performance test results in a nice format."""
    logger = get_logger()

    if not results.get("success", False):
        logger.error("Performance test failed")
        return

    logger.info("\n" + "=" * 50)
    logger.info("PERFORMANCE TEST RESULTS")
    logger.info("=" * 50)
    logger.info(f"Tests completed: {results['tests_completed']}")
    logger.info(f"Average game time: {results['average_game_time']:.2f} seconds")
    logger.info(f"Average ticks per game: {results['average_ticks']:.0f}")
    logger.info(f"Average ticks per second: {results['average_ticks_per_second']:.1f}")
    logger.info("=" * 50)

    for result in results["individual_results"]:
        logger.info(
            f"Test {result['test_number']}: {result['total_time']:.2f}s, "
            f"{result['total_ticks']} ticks, {result['ticks_per_second']:.1f} TPS"
        )


if __name__ == "__main__":
    sys.exit(main())
