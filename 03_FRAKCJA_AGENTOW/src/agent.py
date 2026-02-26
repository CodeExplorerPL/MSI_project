import json
import random
import numpy as np

from pydantic import BaseModel
from typing import Dict, Any

from .observer import BattlefieldObserver
from .strategy import StrategyType, StrategyModel, INPUTS_DEFINITION

from .genetic import ANFIS_Specimen

from .tactics import *
from .state import *

# ============================================================================
# ACTION COMMAND MODEL
# ============================================================================

class ActionCommand(BaseModel):
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: str = None
    should_fire: bool = False

# ============================================================================
# AGENT LOGIC
# ============================================================================

class Agent007:
    """
    Agent with more structured, stateful behavior for testing purposes.
    Drives in one direction for a while, then changes.
    Scans with its turret.
    """
    
    def __init__(self, name: str = "Bot007", specimen: ANFIS_Specimen = None, training: bool = False):
        self.name = name
        self.is_destroyed = False
        print(f"[{self.name}] Agent initialized")

        self.tactic_state = IterState()

        self.observer = BattlefieldObserver()
        self.score = 0
        self.shots = 0
        self.enemy_detected = 0
        self.enemy_in_range = 0
        self.enemy_aim = 0
        self.specimen = None
        self.training = training
        self.strategy_selector = StrategyModel(INPUTS_DEFINITION)
        self.strategy_counts = {s.name: 0 for s in StrategyType}
        if specimen:
            self.load_specimen(specimen)
    
    def set_training_mode(self, enabled: bool) -> None:
        self.training = enabled
        self.observer.set_training_mode(enabled)
            
    def load_specimen(self, specimen: ANFIS_Specimen):
        self.score = 0
        self.shots = 0
        self.enemy_detected = 0
        self.enemy_in_range = 0
        self.enemy_aim = 0
        self.specimen = specimen
        self.strategy_selector.set_params_from_genes(specimen)

    def _prepare_inputs(self, summary: dict) -> np.ndarray:
        """Mapuje dane z Observera na zakres [0, 1] dla ANFIS."""
        enemy = summary.get("radar", {}).get("nearest_enemy")
        enemy_dist = enemy.get("dist") if enemy else None

        powerups = summary.get("logistics", {}).get("powerups", {})
        nearest_powerup_dist = None
        for powerup in powerups.values():
            dist = powerup.get("dist")
            if dist is None:
                continue
            if nearest_powerup_dist is None or dist < nearest_powerup_dist:
                nearest_powerup_dist = dist

        feature_values = {
            "my_hp": summary.get("self", {}).get("hp_pct", 100.0) / 100.0,
            "enemy_dist": (enemy_dist / 300.0) if enemy_dist is not None else 1.0,
            "reload_status": summary.get("self", {}).get("reload_ticks", 0.0) / 10.0,
            # na bezwzględnej wartości obrażeń
            "terrain_risk": abs(float(summary.get("self", {}).get("terrain_damage", 0.0) or 0.0)) / 5.0,
            "enemies_left": min(float(summary.get("radar", {}).get("enemies_left", 0.0)) / 10.0, 1.0),
        }

        ordered_features = []
        for fuzzy_input in INPUTS_DEFINITION:
            feature_name = getattr(fuzzy_input, "name", "")
            value = feature_values.get(feature_name, 0.5)
            ordered_features.append(float(np.clip(value, 0.0, 1.0)))

        return np.array(ordered_features, dtype=float) 


    # def _prepare_inputs(self, summary: dict) -> np.ndarray:
    #     """Mapuje dane z Observera na zakres [0, 1] dla ANFIS."""
    #     # 1. HP (0-100 -> 0-1)
    #     hp = summary["self"]["hp_pct"] / 100.0

    #     # 2. Dystans do wroga (0-800 -> 0-1)
    #     enemy = summary["radar"]["nearest_enemy"]
    #     e_dist = min(enemy["dist"] / 800.0, 1.0) if enemy else 1.0
        
    #     # 3. Status przeładowania (0-1)
    #     reload = min(summary["self"]["reload_ticks"] / 60.0, 1.0)
        
    #     return np.array([hp, e_dist, reload])

    def decide_strategy(self, summary) -> StrategyType:
        """Główna metoda wyboru strategii."""
        input_vector = self._prepare_inputs(summary)
        prediction = self.strategy_selector.get_result(input_vector)
        val = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction

        return StrategyType(int(np.clip(np.floor(val), 0, 4)))

    def get_action(
        self, 
        current_tick: int, 
        my_tank_status: Dict[str, Any], 
        sensor_data: Dict[str, Any], 
        enemies_remaining: int
    ) -> ActionCommand:
        
        # NEW CODE =========================================================
        self.observer.update(my_tank_status, sensor_data, enemies_remaining)
        summary = self.observer.get_summary()

        current_strategy = self.decide_strategy(summary)
        self.strategy_counts[current_strategy.name] += 1
        # ==================================================================
        
        enemy = summary["radar"]["nearest_enemy"]

        if enemy is not None:
            current_strategy = StrategyType.ATTACK

        action = get_action_to_tactics(current_strategy, self.observer, self.tactic_state)

        if self.training:
            self.local_stats(summary, action)

        return action

    def local_stats(self, summary, action: ActionCommand):
        if action.should_fire and summary.get("self", {}).get("reload_ticks", 0.0) == 0.0:
            self.shots += 1

        if summary.get("radar", {}).get("nearest_enemy") is not None:
            self.enemy_detected += 1
            self.enemy_in_range += (summary.get("tactical", {}).get("can_fire", False))*1
            self.enemy_aim  += abs(summary.get("tactical", {}).get("rotation_to_target", 100)) < 10

    def destroy(self):
        """Called when tank is destroyed."""
        self.is_destroyed = True
        print(f"[{self.name}] Tank destroyed!")
    
    def end(self, damage_dealt: float, tanks_killed: int):
        """Called when game ends."""
        print(f"[{self.name}] Game ended!")
        print(f"[{self.name}] Damage dealt: {damage_dealt}")
        print(f"[{self.name}] Tanks killed: {tanks_killed}")

        if self.training:
            print(f"[{self.name}] Enemy detected ticks: {self.enemy_detected} | Times fired: {self.shots} | Time in range: {self.enemy_in_range} | Time aimed: {self.enemy_aim}")
                                                               

        if self.training and self.specimen:
            self._score_genotype(damage_dealt, tanks_killed)
            self._save_strategy_counts()


    # tutaj zmiany
    def _score_genotype(self, damage_dealt: float, tanks_killed: int) -> None:
        """
        Bounded fitness in [0, 100].
        - Strongly rewards kills + damage.
        - Rewards survival and remaining HP.
        - Penalizes terrain damage.
        - Penalizes "stalling": lots of SAVE with low impact.
        """
        summary = self.observer.get_summary()

        hp_pct = float(summary.get("self", {}).get("hp_pct", 0.0) or 0.0)
        hp_n = float(np.clip(hp_pct / 100.0, 0.0, 1.0))

        terrain_damage = float(
            summary.get("self", {}).get("terrain_damage", 0.0) or 0.0
        )
        terrain_abs = abs(terrain_damage)

        dmg = max(0.0, float(damage_dealt))
        kills = max(0, int(tanks_killed))

        # Smooth normalization (prevents outliers dominating).
        # Tune denominators (200.0, 2.0, 5.0) to your game's scale.
        damage_n = 1.0 - math.exp(-dmg / 200.0)          # 0..1
        kills_n = 1.0 - math.exp(-kills / 2.0)           # 0..1
        survive_n = 1.0 if not self.is_destroyed else 0.0
        terrain_n = 1.0 - math.exp(-terrain_abs / 5.0)   # 0..1

        total_choices = sum(self.strategy_counts.values())
        save_ratio = (
            self.strategy_counts.get("SAVE", 0) / total_choices
            if total_choices > 0
            else 0.0
        )

        # Only punish SAVE if it correlates with doing nothing useful.
        stalling = 1.0 if (kills == 0 and dmg < 50.0 and save_ratio > 0.35) else 0.0

        # Weights sum to 100 on the positive side.
        score = (
            45.0 * damage_n
            + 35.0 * kills_n
            + 10.0 * survive_n
            + 10.0 * hp_n
            - 10.0 * terrain_n
            - 10.0 * stalling
        )

        self.specimen.score = float(np.clip(score, 0.0, 100.0))
        self.specimen.save_to_file()

    def _save_strategy_counts(self) -> None:
        filename = f"strategy_counts_{self.name}.json"
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(self.strategy_counts, handle, indent=2)