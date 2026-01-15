"""
Skrypt do uruchamiania pełnej symulacji gry w trybie headless.

Ten skrypt automatycznie:
1. Uruchamia wymaganą liczbę serwerów agentów w osobnych procesach.
2. Uruchamia główną pętlę gry z `headless=True`.
3. Czeka na zakończenie gry.
4. Wyświetla wyniki i zamyka serwery agentów.
"""

import subprocess
import sys
import os
import time

# --- Konfiguracja Ścieżek ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    from backend.engine.game_loop import run_game, TEAM_A_NBR, TEAM_B_NBR, AGENT_BASE_PORT
    from backend.utils.logger import set_log_level

except ImportError as e:
    print(f"Błąd importu: {e}")
    print("Upewnij się, że skrypt jest uruchamiany z katalogu '02_FRAKCJA_SILNIKA' lub że struktura projektu jest poprawna.")
    sys.exit(1)

# --- Konfiguracja Uruchomienia ---
LOG_LEVEL = "INFO"  # Poziomy: DEBUG, INFO, WARNING, ERROR
MAP_SEED = "headless_test_map"

def main():
    """Główna funkcja uruchamiająca symulację."""
    print("--- Uruchamianie symulacji w trybie headless ---")
    set_log_level(LOG_LEVEL)

    agent_processes = []
    total_tanks = TEAM_A_NBR + TEAM_B_NBR
    controller_script_path = os.path.join(current_dir, 'controller', 'server.py')

    if not os.path.exists(controller_script_path):
        print(f"BŁĄD: Nie znaleziono skryptu serwera agenta w: {controller_script_path}")
        return

    try:
        print(f"Uruchamianie {total_tanks} serwerów agentów...")
        for i in range(total_tanks):
            port = AGENT_BASE_PORT + i
            command = [sys.executable, controller_script_path, "--port", str(port)]
            proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            agent_processes.append(proc)
            print(f"  -> Agent {i+1} uruchomiony na porcie {port} (PID: {proc.pid})")

        print("\nOczekiwanie 3 sekundy na start serwerów agentów...")
        time.sleep(3)

        print("\n--- Rozpoczynanie pętli gry ---")
        game_results = run_game(
            map_seed=MAP_SEED,
            headless=True
        )
        print("--- Pętla gry zakończona ---")

        print("\n--- Wyniki Gry ---")
        if game_results.get("success"):
            winner = game_results.get("winner_team")
            if winner:
                print(f"🏆 Zwycięzca: Drużyna {winner}")
            else:
                print("🤝 Remis")
            print(f"Całkowita liczba ticków: {game_results.get('total_ticks')}")

            print("\nTablica wyników:")
            scoreboards = game_results.get("scoreboards", [])
            if scoreboards:
                scoreboards.sort(key=lambda x: (x.get('team', 0), -x.get('tanks_killed', 0)))
                for score in scoreboards:
                    print(f"  - Czołg: {score.get('tank_id')}, Drużyna: {score.get('team')}, "
                          f"Zabójstwa: {score.get('tanks_killed')}, Obrażenia: {score.get('damage_dealt', 0):.0f}")
            else:
                print("  Brak danych o wynikach.")
        else:
            print(f"❌ Gra zakończyła się błędem: {game_results.get('error')}")

    finally:
        print("\n--- Zamykanie serwerów agentów ---")
        for proc in agent_processes:
            proc.terminate()
            print(f"  -> Zatrzymano proces agenta (PID: {proc.pid})")
        print("\n--- Zakończono symulację ---")

if __name__ == "__main__":
    main()
