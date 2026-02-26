# API Agent Package

Struktura:
- `agent.py` - serwer HTTP agenta (`/`, `/agent/action`, `/agent/destroy`, `/agent/end`)
- `utils/` - moduły A*, TSK ruchu, TSK strzału, runtime mapy, adapter payloadu i model lufy NN

Model lufy NN:
- `utils/turret_aim_only.pt` (checkpoint)
- `utils/turret_nn_runtime.py` (runtime inferencji)
- `utils/dqn_neuro_anfis.py`, `utils/state_encoder.py` (architektura i encoder)

Uruchomienie:
```bash
python agent.py --host 0.0.0.0 --port 8001 --name AStarTSK_Agent
```

Silnik (`Engine/backend/engine/game_loop.py`) będzie łączył się pod:
- `GET /`
- `POST /agent/action`
- `POST /agent/destroy`
- `POST /agent/end`

Parametry i przełączniki:
- `utils/tsk_weights.json`
- sekcja `turret_nn` steruje ładowaniem checkpointu i parametrami runtime.
