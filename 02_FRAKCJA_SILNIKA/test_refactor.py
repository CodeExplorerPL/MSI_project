"""
Simple test script to verify refactored game core and game loop compatibility
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))


def test_imports():
    """Test if all imports work correctly."""
    print("Testing imports...")

    try:
        from backend.utils.config import GameConfig, TankType, game_config

        print("✓ Config import successful")
    except ImportError as e:
        print(f"✗ Config import failed: {e}")
        return False

    try:
        from backend.utils.logger import GameEventType, get_logger

        print("✓ Logger import successful")
    except ImportError as e:
        print(f"✗ Logger import failed: {e}")
        return False

    try:
        from backend.engine.game_core import GameCore, create_default_game

        print("✓ Game core import successful")
    except ImportError as e:
        print(f"✗ Game core import failed: {e}")
        return False

    try:
        from backend.engine.game_loop import GameLoop, run_game

        print("✓ Game loop import successful")
    except ImportError as e:
        print(f"✗ Game loop import failed: {e}")
        return False

    try:
        from backend.structures.ammo import AmmoType

        print("✓ AmmoType import successful")
    except ImportError as e:
        print(f"✗ AmmoType import failed: {e}")
        return False

    try:
        from backend.structures.position import Position

        print("✓ Position import successful")
    except ImportError as e:
        print(f"✗ Position import failed: {e}")
        return False

    return True


def test_config():
    """Test configuration functionality."""
    print("\nTesting configuration...")

    try:
        from backend.structures.ammo import AmmoType
        from backend.utils.config import game_config, get_ammo_damage, get_ammo_range

        # Test basic config access
        print(
            f"Map size: {game_config.map_config.width}x{game_config.map_config.height}"
        )
        print(f"Team size: {game_config.tank_config.team_size}")
        print(f"Team count: {game_config.tank_config.team_count}")

        # Test ammo helper functions
        heavy_damage = get_ammo_damage(AmmoType.HEAVY)
        heavy_range = get_ammo_range(AmmoType.HEAVY)
        print(f"Heavy ammo: {heavy_damage} damage, {heavy_range} range")

        # Test spawn positions
        positions = game_config.get_tank_spawn_positions()
        print(f"Generated {len(positions)} spawn positions")

        print("✓ Configuration test successful")
        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_game_core():
    """Test game core functionality."""
    print("\nTesting game core...")

    try:
        from backend.engine.game_core import create_default_game

        # Create game core
        game_core = create_default_game()
        print(f"Game core session ID: {game_core.session_id}")

        # Test initialization
        init_result = game_core.initialize_game("test_seed")
        if init_result["success"]:
            print("✓ Game initialization successful")
        else:
            print(f"✗ Game initialization failed: {init_result.get('error')}")
            return False

        # Test tick processing
        tick_info = game_core.process_tick()
        print(f"Processed tick: {tick_info['tick']}")

        # Test game state
        can_continue = game_core.can_continue_game()
        print(f"Can continue game: {can_continue}")

        print("✓ Game core test successful")
        return True

    except Exception as e:
        print(f"✗ Game core test failed: {e}")
        return False


def test_game_loop():
    """Test game loop functionality."""
    print("\nTesting game loop...")

    try:
        from backend.engine.game_loop import GameLoop

        # Create game loop
        game_loop = GameLoop(headless=True)
        print("Game loop created successfully")

        # Test initialization (without agents for now)
        init_success = game_loop.initialize_game("test_seed", None)
        if init_success:
            print("✓ Game loop initialization successful")
        else:
            print("✗ Game loop initialization failed")
            return False

        # Test cleanup
        game_loop.cleanup_game()
        print("✓ Game loop cleanup successful")

        print("✓ Game loop test successful")
        return True

    except Exception as e:
        print(f"✗ Game loop test failed: {e}")
        return False


def test_logger():
    """Test logger functionality."""
    print("\nTesting logger...")

    try:
        from backend.utils.logger import GameEventType, get_logger

        logger = get_logger()

        # Test basic logging
        logger.info("Test info message")
        logger.debug("Test debug message")
        logger.warning("Test warning message")

        # Test game events
        logger.log_game_event(GameEventType.GAME_START, "Test game start event")

        print("✓ Logger test successful")
        return True

    except Exception as e:
        print(f"✗ Logger test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("REFACTOR COMPATIBILITY TEST")
    print("=" * 50)

    tests = [test_imports, test_config, test_game_core, test_game_loop, test_logger]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")

    print("\n" + "=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Refactor is compatible.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
