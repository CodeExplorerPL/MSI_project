import math
import random
from typing import Dict, Any

# Zakładam, że te klasy można zaimportować z Twojej struktury projektu
from .strategy import StrategyType
from .observer import BattlefieldObserver
# from .agent import ActionCommand  # Pamiętaj o zaimportowaniu ActionCommand!

from pydantic import BaseModel

class ActionCommand(BaseModel):
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: str = None
    should_fire: bool = False

# ============================================================================
# FUNKCJE POMOCNICZE
# ============================================================================

def get_best_ammo(my_tank: Dict[str, Any]) -> str:
    """Wybiera typ amunicji, którego jest najwięcej."""
    ammo_data = my_tank.get("ammo", {})
    if not ammo_data:
        return "DEFAULT"
    return max(ammo_data, key=lambda k: ammo_data[k].get("count", 0))

def get_heading_to_pos(my_tank: Dict[str, Any], target_pos: Dict[str, float]) -> float:
    """Oblicza wymaganą rotację kadłuba, by skierować się w stronę target_pos."""
    # Engine exposes hull orientation as "heading" ("angle" kept as fallback).
    my_angle = my_tank.get("heading", my_tank.get("angle", 0.0))
    dx = target_pos["x"] - my_tank["position"]["x"]
    dy = target_pos["y"] - my_tank["position"]["y"]
    target_angle = math.degrees(math.atan2(dy, dx))
    
    diff = target_angle - my_angle
    return (diff + 180) % 360 - 180

# ============================================================================
# TAKTYKI DLA POSZCZEGÓLNYCH STRATEGII
# ============================================================================

def tactic_attack(summary: Dict[str, Any], observer: BattlefieldObserver) -> ActionCommand:
    """Cel: Zbliżyć się do wroga i go zniszczyć."""
    nearest = summary["radar"]["nearest_enemy"]
    can_fire = summary["tactical"]["can_fire"]
    barrel_rot = summary["tactical"]["rotation_to_target"]
    
    move_speed = 0.0
    heading_rot = 0.0
    
    if nearest:
        # Jeśli wróg jest daleko, jedź w jego stronę. Jeśli blisko - stój (lub manewruj) i strzelaj.
        if nearest["dist"] > 250.0:
            heading_rot = get_heading_to_pos(observer.my_tank, nearest["tank_data"]["position"])
            move_speed = 3.0
        else:
            # Delikatny "strafing" przy bliskim dystansie
            move_speed = random.choice([0.0, 1.0])
            heading_rot = random.choice([-15.0, 15.0])

    return ActionCommand(
        barrel_rotation_angle=barrel_rot,
        heading_rotation_angle=heading_rot,
        move_speed=move_speed,
        ammo_to_load=get_best_ammo(observer.my_tank),
        should_fire=can_fire
    )

def tactic_flee(summary: Dict[str, Any], observer: BattlefieldObserver) -> ActionCommand:
    """Cel: Ucieczka w przeciwnym kierunku do najbliższego wroga."""
    nearest = summary["radar"]["nearest_enemy"]
    
    if not nearest:
        return tactic_search(summary, observer) # Jeśli nie ma od kogo uciekać, szukaj

    # Rotacja wieżyczki w stronę wroga, by móc oddać strzał zaporowy
    barrel_rot = summary["tactical"]["rotation_to_target"]
    
    # Kąt ucieczki to kąt do wroga + 180 stopni
    heading_to_enemy = get_heading_to_pos(observer.my_tank, nearest["tank_data"]["position"])
    escape_heading = (heading_to_enemy + 180) % 360 - 180

    return ActionCommand(
        barrel_rotation_angle=barrel_rot,
        heading_rotation_angle=escape_heading,
        move_speed=5.0, # Maksymalna prędkość ucieczki
        ammo_to_load=get_best_ammo(observer.my_tank),
        should_fire=summary["tactical"]["can_fire"]
    )

def tactic_save(summary: Dict[str, Any], observer: BattlefieldObserver) -> ActionCommand:
    """Cel: Defensywa, unikanie ognia (zygzakowanie) i ewentualne strzelanie obronne."""
    # Podobne do Flee, ale z większym naciskiem na defensywne manewry
    return ActionCommand(
        barrel_rotation_angle=summary["tactical"]["rotation_to_target"],
        heading_rotation_angle=random.choice([-45.0, 45.0]), # Uniki
        move_speed=5.0,
        ammo_to_load=get_best_ammo(observer.my_tank),
        should_fire=summary["tactical"]["can_fire"]
    )

def tactic_search(summary: Dict[str, Any], observer: BattlefieldObserver) -> ActionCommand:
    """Cel: Patrolowanie i skanowanie terenu wieżyczką."""
    # Kręcenie wieżyczką, żeby radar kogoś wyłapał
    barrel_rot = 15.0 
    
    # Spokojny ruch patrolowy ze zmianą kierunku
    heading_rot = random.choice([-10.0, 0.0, 10.0])
    
    return ActionCommand(
        barrel_rotation_angle=barrel_rot,
        heading_rotation_angle=heading_rot,
        move_speed=3.0,
        ammo_to_load=get_best_ammo(observer.my_tank),
        should_fire=False
    )

def tactic_powerup(summary: Dict[str, Any], observer: BattlefieldObserver) -> ActionCommand:
    """Cel: Priorytetyzacja zebrania najbliższego powerupa."""
    powerups = summary["logistics"]["powerups"]
    
    if not powerups:
        return tactic_search(summary, observer) # Brak powerupów -> wracamy do szukania

    # Wybierz pierwszy z brzegu powerup (LogisticsModule sortuje/odfiltrowuje te najlepsze)
    closest_p_type = list(powerups.keys())[0]
    closest_pu = powerups[closest_p_type]

    heading_rot = get_heading_to_pos(observer.my_tank, closest_pu["pos"])

    return ActionCommand(
        barrel_rotation_angle=summary["tactical"]["rotation_to_target"], # Trzymaj na celowniku wroga jeśli jest
        heading_rotation_angle=heading_rot,
        move_speed=4.0,
        ammo_to_load=get_best_ammo(observer.my_tank),
        should_fire=summary["tactical"]["can_fire"]
    )

# ============================================================================
# GŁÓWNY ROUTER TAKTYK
# ============================================================================

def get_action_to_tactics(strategy: StrategyType, observer: BattlefieldObserver) -> ActionCommand:
    """
    Funkcja mapująca wybraną strategię (np. z sieci neuronowej/ANFIS) 
    na konkretne wywołanie taktyki sterującej.
    """
    summary = observer.get_summary()

    # Słownik pełniący rolę instrukcji `switch/match`
    tactics_map = {
        StrategyType.ATTACK: tactic_attack,
        StrategyType.FLEE: tactic_flee,
        StrategyType.SAVE: tactic_save,
        StrategyType.SEARCH: tactic_search,
        StrategyType.POWERUP: tactic_powerup,
    }

    # Pobieramy odpowiednią funkcję i ją uruchamiamy
    tactic_function = tactics_map.get(strategy, tactic_search)
    return tactic_function(summary, observer)



