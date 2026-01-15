"""
Główny skrypt do uruchamiania gry w trybie z interfejsem graficznym (UI).
"""

import sys
import os

def setup_path():
    """Dodaje główny katalog projektu do ścieżki, aby umożliwić importy."""
    try:
        # Ścieżka do bieżącego pliku
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Dodaj również katalog '02_FRAKCJA_SILNIKA' dla spójności
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
    except Exception as e:
        print(f"Błąd podczas ustawiania ścieżki: {e}")
        sys.path.append(os.path.abspath('.'))

def main():
    """Główna funkcja uruchamiająca grę w trybie graficznym."""
    print("Ustawianie ścieżki projektu...")
    setup_path()
    
    print("Importowanie silnika gry...")
    # Import musi nastąpić po ustawieniu ścieżki
    from backend.engine.game_loop import run_game
    
    print("Uruchamianie gry z interfejsem użytkownika...")
    game_results = run_game(headless=False)
    
    print("\n--- Koniec Gry ---")
    print(f"Gra zakończona. Wyniki: {game_results}")

if __name__ == "__main__":
    main()