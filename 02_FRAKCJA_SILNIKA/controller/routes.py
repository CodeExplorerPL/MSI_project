"""
Definicja endpointów API dla serwera agenta.

Ten moduł tworzy router FastAPI i definiuje endpointy, które silnik gry
będzie wywoływał w celu uzyskania akcji od agenta i informowania go o
stanie gry.
"""

import sys
import os
from fastapi import APIRouter, Body, HTTPException
from typing import Any, Dict

# --- Dynamiczne dodawanie ścieżki do `final_api.py` ---
# W idealnym scenariuszu, projekt miałby strukturę pakietu, ale przy
# obecnym układzie folderów, jest to konieczne do znalezienia `final_api.py`.
try:
    doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '01_DOKUMENTACJA'))
    if doc_path not in sys.path:
        sys.path.append(doc_path)
    # Importujemy wszystkie modele danych i interfejs z pliku API
    from final_api import *
except ImportError as e:
    print(f"Krytyczny błąd: Nie można zaimportować `final_api`. Upewnij się, że ścieżka jest poprawna: {e}")
    sys.exit(1)

from pydantic import TypeAdapter

# ==============================================================================
# Logika Agenta (Miejsce na "mózg" czołgu)
# ==============================================================================

# Poniżej znajduje się przykładowa, prosta implementacja agenta.
# Właściwy agent powinien zastąpić tę klasę swoją własną, bardziej zaawansowaną logiką.
class MyAgentController(IAgentController):
    """Przykładowa implementacja kontrolera agenta."""

    def get_action(self, current_tick: int, my_tank_status: TankUnion, sensor_data: TankSensorData, enemies_remaining: int) -> ActionCommand:
        print(f"Tick: {current_tick}, HP: {my_tank_status.hp}, Pozycja: {my_tank_status.position}")

        # Prosta logika: jeśli widzisz wroga, celuj i strzelaj. W przeciwnym razie jedź prosto.
        if sensor_data.seen_tanks:
            target = sensor_data.seen_tanks[0]
            # TODO: Tutaj powinna znaleźć się logika obliczania kąta do celu
            target_angle_delta = 10.0  # Przykładowy obrót
            return ActionCommand(
                barrel_rotation_angle=target_angle_delta,
                heading_rotation_angle=0.0,
                move_speed=0.0,
                should_fire=True
            )

        return ActionCommand(
            barrel_rotation_angle=5.0,  # Kręć lufą w poszukiwaniu celów
            heading_rotation_angle=0.0,
            move_speed=my_tank_status._top_speed,
            should_fire=False
        )

    def destroy(self):
        print("Powiadomienie od silnika: Mój czołg został zniszczony.")

    def end(self):
        print("Powiadomienie od silnika: Gra zakończona.")

# Instancja logiki agenta
agent_controller = MyAgentController()

# Adaptery Pydantic do parsowania złożonych typów Union z JSON
TankUnionAdapter = TypeAdapter(TankUnion)
TankSensorDataAdapter = TypeAdapter(TankSensorData)

# ==============================================================================
# Definicje Endpointów API
# ==============================================================================

router = APIRouter()

@router.post("/action", response_model=ActionCommand)
async def get_action_endpoint(payload: Dict[str, Any] = Body(...)):
    """Główny endpoint, który silnik wywołuje co turę, aby uzyskać decyzję agenta."""
    try:
        # Używamy adapterów Pydantic do automatycznego sparsowania słowników JSON
        # na obiekty dataclass zdefiniowane w `final_api.py`.
        # Pydantic inteligentnie obsługuje typy Union na podstawie pól-dyskryminatorów (np. `_tank_type`).
        my_tank_status = TankUnionAdapter.validate_python(payload['my_tank_status'])
        sensor_data = TankSensorDataAdapter.validate_python(payload['sensor_data'])

        # Wywołanie właściwej logiki agenta
        action = agent_controller.get_action(
            current_tick=payload['current_tick'],
            my_tank_status=my_tank_status,
            sensor_data=sensor_data,
            enemies_remaining=payload['enemies_remaining']
        )
        return action
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Błędna struktura danych wejściowych: {e}")

@router.post("/destroy", status_code=204)
async def destroy_endpoint():
    """Endpoint do powiadamiania agenta o zniszczeniu jego czołgu."""
    agent_controller.destroy()

@router.post("/end", status_code=204)
async def end_endpoint():
    """Endpoint do powiadamiania agenta o zakończeniu gry."""
    agent_controller.end()