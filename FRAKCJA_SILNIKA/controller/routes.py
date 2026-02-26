"""
Definicja endpointów API dla serwera agenta.
"""

import json
from fastapi import APIRouter, Body, HTTPException
from typing import Any, Dict
from pydantic import TypeAdapter

from api import (
    ActionCommand, TankUnion, TankSensorData, get_active_agent, Scoreboard
)

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
        # Wyświetlanie otrzymanych danych (Symulacja logowania po stronie serwera)
        print("\n" + "="*40)
        print(f"📥 SERVER RECEIVED (Tick: {payload.get('current_tick')})")
        print(json.dumps(payload, indent=2, default=str))
        print("="*40)

        # W tym prostym przykładzie przekazujemy dicty bezpośrednio, aby uniknąć
        # problemów z walidacją klas backendu, które nie są czystymi dataclassami.
        my_tank_status = payload['my_tank_status']
        sensor_data = payload['sensor_data']

        # Wywołanie właściwej logiki agenta
        action = get_active_agent().get_action(
            current_tick=payload['current_tick'],
            my_tank_status=my_tank_status,
            sensor_data=sensor_data,
            enemies_remaining=payload['enemies_remaining']
        )
        
        # Wyświetlanie wysyłanej odpowiedzi
        print(f"📤 SERVER SENDING ACTION:")
        print(action)
        print("="*40 + "\n")
        
        return action
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Błędna struktura danych wejściowych: {e}")

@router.post("/destroy", status_code=204)
async def destroy_endpoint():
    """Endpoint do powiadamiania agenta o zniszczeniu jego czołgu."""
    get_active_agent().destroy()

@router.post("/end", status_code=204)
async def end_endpoint(payload: Dict[str, Any] = Body(...)):
    """Endpoint do powiadamiania agenta o zakończeniu gry."""
    # Logowanie otrzymania sygnału końca gry
    print("\n" + "="*40)
    print(f"📥 SERVER RECEIVED END GAME")
    print(json.dumps(payload, indent=2, default=str))
    print("="*40 + "\n")

    final_score = Scoreboard(
        damage_dealt=payload.get('damage_dealt', 0.0),
        tanks_killed=payload.get('tanks_killed', 0)
    )
    get_active_agent().end(final_score)