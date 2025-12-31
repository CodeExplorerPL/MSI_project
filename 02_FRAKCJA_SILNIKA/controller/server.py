"""
Główny plik serwera API dla agenta.

W tej architekturze AGENT działa jako SERWER HTTP.
Silnik gry (Game Engine) działa jako KLIENT, który wysyła zapytania do agenta.

Uruchomienie tego pliku (`python server.py`) startuje "mózg" agenta,
który nasłuchuje na porcie 8000 i czeka na dane z gry.
"""

from fastapi import FastAPI
import argparse
import uvicorn
import routes
from api import set_active_agent

# Tutaj importujemy konkretną implementację agenta, której chcemy użyć
from example_agent_logic import agent_controller

# Rejestracja agenta w systemie API
set_active_agent(agent_controller)

app = FastAPI(
    title="Serwer Agenta Czołgu",
    description="API, które agent musi zaimplementować, aby komunikować się z silnikiem gry.",
    version="1.0.0"
)

app.include_router(routes.router, prefix="/agent", tags=["Akcje Agenta"])

@app.get("/", tags=["Status"])
async def read_root():
    return {"message": "Serwer agenta jest uruchomiony. Oczekuje na połączenie od silnika gry."}

if __name__ == "__main__":
    # Obsługa argumentów linii poleceń, aby można było uruchomić wiele agentów na różnych portach
    parser = argparse.ArgumentParser(description="Uruchom serwer agenta czołgu.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Adres hosta (domyślnie 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port serwera (domyślnie 8000)")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)