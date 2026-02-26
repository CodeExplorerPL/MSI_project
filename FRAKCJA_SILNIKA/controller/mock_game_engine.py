"""
SYMULATOR SILNIKA GRY (MOCK GAME ENGINE).

Ten skrypt pełni rolę KLIENTA HTTP, który udaje Silnik Gry.
Wysyła on zapytania do Twojego Agenta (który działa jako SERWER HTTP),
aby przetestować, czy Agent poprawnie odbiera dane i zwraca decyzje.

W tej architekturze:
1. AGENT = SERWER (Czeka na zapytanie "co robisz?")
2. GRA = KLIENT (Wysyła stan gry i pyta o akcję)
"""

import requests
import json
import sys
import os
import random
from enum import Enum
from dataclasses import asdict, is_dataclass
import argparse

# Dodanie ścieżki do importów z backendu
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from ..controller.api import (
    LightTank, Position, TankSensorData, SeenTank, 
    AmmoType, AmmoSlot
)

def sanitize_for_json(obj):
    """Pomocnicza funkcja do serializacji obiektów (Enumy, Dataclassy, Obiekty backendu)."""
    if isinstance(obj, Enum):
        return obj.name  # Wysyłamy nazwę Enuma (np. "HEAVY")
    elif is_dataclass(obj):
        return sanitize_for_json(asdict(obj))
    elif isinstance(obj, dict):
        return {
            (k.name if isinstance(k, Enum) else str(k)): sanitize_for_json(v) 
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif hasattr(obj, "__dict__"):
        # Fallback dla obiektów backendu, które nie są dataclassami (np. LightTank dziedziczący po Tank)
        # Pobieramy pola zdefiniowane w dataclassie rodzica (Tank)
        data = {}
        # Sprawdzamy pola z klasy bazowej Tank (która jest dataclass)
        from backend.tank.base_tank import Tank
        for field_name in Tank.__dataclass_fields__:
            if hasattr(obj, field_name):
                data[field_name] = getattr(obj, field_name)
        # Dodajemy specyficzne pola
        if hasattr(obj, "_tank_type"):
            data["_tank_type"] = obj._tank_type
        return sanitize_for_json(data)
    else:
        return obj

def create_random_position():
    return Position(random.uniform(0, 500), random.uniform(0, 500))

def create_dummy_payload():
    """Tworzy przykładowy payload zgodny ze strukturą API."""
    
    # 1. Status mojego czołgu
    my_pos = create_random_position()
    # Używamy klasy z backendu (zaimportowanej przez api.py)
    my_tank = LightTank(_id="tank_player_1", team=1, start_pos=my_pos)
    
    # Symulacja obrażeń/zużycia
    my_tank.hp = 75
    my_tank.ammo[AmmoType.HEAVY].count = 0 # Zużyta amunicja

    # 2. Dane sensoryczne
    seen_tanks = []
    # Losowo dodaj wroga
    if random.random() > 0.2:
        seen_tanks.append(SeenTank(
            id="enemy_tank_99",
            position=create_random_position(),
            is_damaged=True,
            heading=random.uniform(0, 360),
            barrel_angle=random.uniform(0, 360),
            distance=random.uniform(10, 50),
            tank_type="HeavyTank",
            team=2
        ))
        
    sensor_data = TankSensorData(
        seen_tanks=seen_tanks,
        seen_powerups=[],
        seen_obstacles=[],
        seen_terrains=[]
    )
    
    # Składanie payloadu
    payload = {
        "current_tick": random.randint(100, 500),
        "my_tank_status": my_tank,
        "sensor_data": sensor_data,
        "enemies_remaining": 4
    }
    
    return sanitize_for_json(payload)

def create_end_payload():
    """Tworzy przykładowy payload dla endpointu /end."""
    return {
        "damage_dealt": round(random.uniform(0, 5000), 2),
        "tanks_killed": random.randint(0, 5)
    }

def run_mock_engine(agent_urls=None):
    if agent_urls is None:
        agent_urls = [
            "http://localhost:8000/agent/action",
        ]
    
    print(f"Generowanie danych symulacyjnych dla {len(agent_urls)} agentów...")

    for i, url in enumerate(agent_urls):
        print(f"\n--- SYMULACJA DLA AGENTA {i+1} ({url}) ---")
        
        # Zakładamy, że URL to endpoint /action, więc ucinamy go, aby uzyskać bazę
        # np. http://localhost:8000/agent/action -> http://localhost:8000/agent
        base_url = url.rsplit('/', 1)[0]

        # 1. TEST ACTION
        print(f"\n[1] TEST ENDPOINTU /action")
        json_payload = create_dummy_payload()
        
        print(f"📤 GAME ENGINE SENDING REQUEST...")
        
        try:
            response = requests.post(url, json=json_payload)
            response.raise_for_status()
            
            print(f"📥 RECEIVED ACTION:")
            print(json.dumps(response.json(), indent=2))
        except requests.exceptions.RequestException as e:
            print(f"❌ Error sending request to {url}: {e}")
            print("   (Czy na pewno uruchomiłeś serwer agenta na tym porcie?)")
            continue

        # 2. TEST DESTROY
        print(f"\n[2] TEST ENDPOINTU /destroy")
        destroy_url = f"{base_url}/destroy"
        try:
            requests.post(destroy_url)
            print(f"✅ Wysłano sygnał zniszczenia do {destroy_url}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Błąd przy wysyłaniu destroy: {e}")

        # 3. TEST END
        print(f"\n[3] TEST ENDPOINTU /end")
        end_url = f"{base_url}/end"
        end_payload = create_end_payload()
        try:
            requests.post(end_url, json=end_payload)
            print(f"✅ Wysłano sygnał końca gry do {end_url} z wynikiem: {end_payload}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Błąd przy wysyłaniu end: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uruchom symulator silnika gry.")
    parser.add_argument("--urls", nargs="+", default=["http://localhost:8000/agent/action"], 
                        help="Lista URL-i agentów do odpytania (domyślnie: http://localhost:8000/agent/action)")
    
    args = parser.parse_args()
    
    run_mock_engine(args.urls)
