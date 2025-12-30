"""
Główny plik serwera API dla agenta.

Uruchomienie tego pliku (`python server.py`) startuje serwer FastAPI,
który jest gotowy do przyjmowania zapytań od silnika gry.
"""

from fastapi import FastAPI
import uvicorn
import routes

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
    # Uruchomienie serwera Uvicorn. Agent będzie nasłuchiwał na porcie 8000.
    uvicorn.run(app, host="0.0.0.0", port=8000)