import random
from typing import Dict, Any
from api import IAgentController, ActionCommand, Scoreboard

class ExampleAgentController(IAgentController):
    """
    PRZYKŁADOWA IMPLEMENTACJA LOGIKI AGENTA (DO TESTÓW).
    
    UWAGA: To jest tylko przykładowa implementacja służąca do testowania
    komunikacji klient-serwer oraz weryfikacji działania API.
    Nie jest to docelowa logika agenta biorącego udział w rozgrywce.
    """

    def get_action(self, current_tick: int, my_tank_status: Dict[str, Any], sensor_data: Dict[str, Any], enemies_remaining: int) -> ActionCommand:
        # Generowanie losowych akcji dla testów połączenia.
        # Ignorujemy dane wejściowe i zwracamy losowe wartości, aby sprawdzić przepływ danych.
        return ActionCommand(
            barrel_rotation_angle=random.uniform(-180.0, 180.0),
            heading_rotation_angle=random.uniform(-180.0, 180.0),
            move_speed=random.uniform(-5.0, 5.0),
            should_fire=random.choice([True, False])
        )

    def destroy(self):
        print("Logika Agenta (Example): Otrzymano informację o zniszczeniu.")

    def end(self, final_score: Scoreboard):
        print(f"Logika Agenta (Example): Otrzymano informację o zakończeniu gry. Wynik: {final_score}")

# Instancja agenta, która będzie importowana przez routes.py
agent_controller = ExampleAgentController()