from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from .geometry import (
    euclidean_distance,
    heading_to_angle_deg,
    normalize_angle_diff,
    to_xy,
)

THREAT_WEIGHTS = {
    "LIGHT": 3,
    "HEAVY": 7,
    "Sniper": 9,
}

OPTIMAL_ENGAGEMENT_RANGE = 50.0
COOLDOWN_TICKS = 10
AIMING_THRESHOLD_TIGHT = 2.5
FIRE_CONFIDENCE_THRESHOLD = 0.6

AMMO_SPECS: dict[str, dict[str, float]] = {
    "HEAVY": {"range": 25.0, "damage": 40.0, "reload": 10.0},
    "LIGHT": {"range": 50.0, "damage": 20.0, "reload": 5.0},
    "LONG_DISTANCE": {"range": 100.0, "damage": 25.0, "reload": 10.0},
}

LAST_SEEN_EXPIRY_TICKS = 40


class FuzzyTurretController:
    def __init__(
        self,
        max_barrel_spin_rate: float,
        vision_range: float,
        aim_threshold: float = AIMING_THRESHOLD_TIGHT,
    ):
        self.max_barrel_spin_rate = max_barrel_spin_rate
        self.vision_range = vision_range
        self.aim_threshold = aim_threshold

        self.cooldown_ticks = 0
        self.last_seen_direction: Optional[float] = None
        self.ticks_since_last_seen = 0

        self._init_target_selection_fuzzy()
        self._init_rotation_speed_fuzzy()
        self._init_firing_decision_fuzzy()
        self._init_adaptive_scan_fuzzy()

    def _init_target_selection_fuzzy(self):
        max_dist = max(self.vision_range * 1.5, 30)
        distance = ctrl.Antecedent(np.arange(0, max_dist + 1, 1), "distance")

        very_close_max = self.vision_range * 0.3
        close_min = self.vision_range * 0.2
        close_max = self.vision_range * 0.6
        medium_min = self.vision_range * 0.5
        medium_max = self.vision_range * 1.0
        far_min = self.vision_range * 0.8

        distance["very_close"] = fuzz.trapmf(
            distance.universe, [0, 0, very_close_max * 0.5, very_close_max]
        )
        distance["close"] = fuzz.trimf(
            distance.universe, [close_min, (close_min + close_max) / 2, close_max]
        )
        distance["medium"] = fuzz.trimf(
            distance.universe, [medium_min, (medium_min + medium_max) / 2, medium_max]
        )
        distance["far"] = fuzz.trapmf(
            distance.universe, [far_min, medium_max, max_dist, max_dist]
        )

        threat = ctrl.Antecedent(np.arange(0, 11, 1), "threat")
        threat["low"] = fuzz.trimf(threat.universe, [0, 0, 5])
        threat["medium"] = fuzz.trimf(threat.universe, [3, 5, 7])
        threat["high"] = fuzz.trimf(threat.universe, [5, 10, 10])

        priority = ctrl.Consequent(np.arange(0, 101, 1), "priority")
        priority["ignore"] = fuzz.trimf(priority.universe, [0, 0, 20])
        priority["low"] = fuzz.trimf(priority.universe, [10, 25, 40])
        priority["medium"] = fuzz.trimf(priority.universe, [30, 50, 70])
        priority["high"] = fuzz.trimf(priority.universe, [60, 75, 90])
        priority["critical"] = fuzz.trimf(priority.universe, [80, 100, 100])

        rules = [
            ctrl.Rule(distance["very_close"] & threat["high"], priority["critical"]),
            ctrl.Rule(distance["very_close"] & threat["medium"], priority["high"]),
            ctrl.Rule(distance["very_close"] & threat["low"], priority["medium"]),
            ctrl.Rule(distance["close"] & threat["high"], priority["critical"]),
            ctrl.Rule(distance["close"] & threat["medium"], priority["high"]),
            ctrl.Rule(distance["close"] & threat["low"], priority["medium"]),
            ctrl.Rule(distance["medium"] & threat["high"], priority["high"]),
            ctrl.Rule(distance["medium"] & threat["medium"], priority["medium"]),
            ctrl.Rule(distance["medium"] & threat["low"], priority["low"]),
            ctrl.Rule(distance["far"] & threat["high"], priority["medium"]),
            ctrl.Rule(distance["far"] & threat["medium"], priority["low"]),
            ctrl.Rule(distance["far"] & threat["low"], priority["ignore"]),
        ]

        self.target_selection_ctrl = ctrl.ControlSystem(rules)
        self.target_selection_sim = ctrl.ControlSystemSimulation(
            self.target_selection_ctrl
        )

    def _init_rotation_speed_fuzzy(self):
        angle_error = ctrl.Antecedent(np.arange(0, 181, 1), "angle_error")
        angle_error["small"] = fuzz.trapmf(angle_error.universe, [0, 0, 5, 15])
        angle_error["medium"] = fuzz.trimf(angle_error.universe, [10, 30, 60])
        angle_error["large"] = fuzz.trapmf(angle_error.universe, [45, 90, 180, 180])

        max_dist = max(self.vision_range * 1.5, 30)
        target_dist = ctrl.Antecedent(np.arange(0, max_dist + 1, 1), "target_distance")

        close_max = self.vision_range * 0.5
        medium_min = self.vision_range * 0.4
        medium_max = self.vision_range * 0.9
        far_min = self.vision_range * 0.7

        target_dist["close"] = fuzz.trapmf(
            target_dist.universe, [0, 0, close_max * 0.6, close_max]
        )
        target_dist["medium"] = fuzz.trimf(
            target_dist.universe,
            [medium_min, (medium_min + medium_max) / 2, medium_max],
        )
        target_dist["far"] = fuzz.trapmf(
            target_dist.universe, [far_min, medium_max, max_dist, max_dist]
        )

        speed_factor = ctrl.Consequent(np.arange(0, 1.01, 0.01), "speed_factor")
        speed_factor["very_slow"] = fuzz.trimf(speed_factor.universe, [0, 0, 0.25])
        speed_factor["slow"] = fuzz.trimf(speed_factor.universe, [0.15, 0.35, 0.55])
        speed_factor["medium"] = fuzz.trimf(speed_factor.universe, [0.45, 0.65, 0.85])
        speed_factor["fast"] = fuzz.trimf(speed_factor.universe, [0.75, 0.9, 1.0])
        speed_factor["very_fast"] = fuzz.trimf(speed_factor.universe, [0.9, 1.0, 1.0])

        rules = [
            ctrl.Rule(angle_error["small"], speed_factor["very_slow"]),
            ctrl.Rule(
                angle_error["medium"] & target_dist["close"], speed_factor["medium"]
            ),
            ctrl.Rule(
                angle_error["medium"] & target_dist["medium"], speed_factor["medium"]
            ),
            ctrl.Rule(angle_error["medium"] & target_dist["far"], speed_factor["fast"]),
            ctrl.Rule(
                angle_error["large"] & target_dist["close"], speed_factor["fast"]
            ),
            ctrl.Rule(
                angle_error["large"] & target_dist["medium"], speed_factor["fast"]
            ),
            ctrl.Rule(
                angle_error["large"] & target_dist["far"], speed_factor["very_fast"]
            ),
        ]

        self.rotation_speed_ctrl = ctrl.ControlSystem(rules)
        self.rotation_speed_sim = ctrl.ControlSystemSimulation(self.rotation_speed_ctrl)

    def _init_firing_decision_fuzzy(self):
        aiming_error = ctrl.Antecedent(np.arange(0, 11, 0.1), "aiming_error")
        aiming_error["perfect"] = fuzz.trapmf(aiming_error.universe, [0, 0, 1, 2])
        aiming_error["good"] = fuzz.trimf(aiming_error.universe, [1.5, 2.5, 4])
        aiming_error["acceptable"] = fuzz.trimf(aiming_error.universe, [3, 5, 7])
        aiming_error["poor"] = fuzz.trapmf(aiming_error.universe, [6, 8, 10, 10])

        max_dist = max(self.vision_range * 1.5, 30)
        firing_dist = ctrl.Antecedent(np.arange(0, max_dist + 1, 1), "firing_distance")

        optimal_peak = min(self.vision_range * 0.5, OPTIMAL_ENGAGEMENT_RANGE)
        optimal_end = self.vision_range * 0.7
        suboptimal_mid = self.vision_range * 0.9
        suboptimal_end = self.vision_range * 1.2
        extreme_start = self.vision_range * 1.0

        firing_dist["optimal"] = fuzz.trimf(
            firing_dist.universe, [0, optimal_peak, optimal_end]
        )
        firing_dist["suboptimal"] = fuzz.trimf(
            firing_dist.universe, [optimal_end * 0.8, suboptimal_mid, suboptimal_end]
        )
        firing_dist["extreme"] = fuzz.trapmf(
            firing_dist.universe, [extreme_start, suboptimal_end, max_dist, max_dist]
        )

        vulnerability = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "vulnerability")
        vulnerability["resilient"] = fuzz.trimf(vulnerability.universe, [0, 0, 0.4])
        vulnerability["normal"] = fuzz.trimf(vulnerability.universe, [0.3, 0.5, 0.7])
        vulnerability["vulnerable"] = fuzz.trimf(
            vulnerability.universe, [0.6, 1.0, 1.0]
        )

        fire_conf = ctrl.Consequent(np.arange(0, 1.01, 0.01), "fire_confidence")
        fire_conf["no"] = fuzz.trimf(fire_conf.universe, [0, 0, 0.3])
        fire_conf["maybe"] = fuzz.trimf(fire_conf.universe, [0.2, 0.5, 0.7])
        fire_conf["yes"] = fuzz.trimf(fire_conf.universe, [0.6, 1.0, 1.0])

        rules = [
            ctrl.Rule(
                aiming_error["perfect"] & firing_dist["optimal"], fire_conf["yes"]
            ),
            ctrl.Rule(
                aiming_error["perfect"] & firing_dist["suboptimal"], fire_conf["yes"]
            ),
            ctrl.Rule(
                aiming_error["perfect"] & firing_dist["extreme"], fire_conf["maybe"]
            ),
            ctrl.Rule(aiming_error["good"] & firing_dist["optimal"], fire_conf["yes"]),
            ctrl.Rule(
                aiming_error["good"] & firing_dist["suboptimal"], fire_conf["maybe"]
            ),
            ctrl.Rule(aiming_error["good"] & firing_dist["extreme"], fire_conf["no"]),
            ctrl.Rule(
                aiming_error["acceptable"]
                & firing_dist["optimal"]
                & vulnerability["vulnerable"],
                fire_conf["yes"],
            ),
            ctrl.Rule(
                aiming_error["acceptable"]
                & firing_dist["optimal"]
                & vulnerability["normal"],
                fire_conf["maybe"],
            ),
            ctrl.Rule(
                aiming_error["acceptable"]
                & firing_dist["optimal"]
                & vulnerability["resilient"],
                fire_conf["maybe"],
            ),
            ctrl.Rule(
                aiming_error["acceptable"] & firing_dist["suboptimal"],
                fire_conf["maybe"],
            ),
            ctrl.Rule(
                aiming_error["acceptable"] & firing_dist["extreme"], fire_conf["no"]
            ),
            ctrl.Rule(aiming_error["poor"], fire_conf["no"]),
            ctrl.Rule(firing_dist["extreme"] & aiming_error["poor"], fire_conf["no"]),
            ctrl.Rule(
                firing_dist["extreme"] & aiming_error["acceptable"], fire_conf["no"]
            ),
        ]

        self.firing_decision_ctrl = ctrl.ControlSystem(rules)
        self.firing_decision_sim = ctrl.ControlSystemSimulation(
            self.firing_decision_ctrl
        )

    def _init_adaptive_scan_fuzzy(self):
        time_unseen = ctrl.Antecedent(np.arange(0, 101, 1), "time_unseen")
        time_unseen["recent"] = fuzz.trapmf(time_unseen.universe, [0, 0, 10, 25])
        time_unseen["moderate"] = fuzz.trimf(time_unseen.universe, [15, 40, 65])
        time_unseen["long"] = fuzz.trapmf(time_unseen.universe, [50, 80, 100, 100])

        scan_error = ctrl.Antecedent(np.arange(0, 181, 1), "scan_error")
        scan_error["aligned"] = fuzz.trapmf(scan_error.universe, [0, 0, 20, 45])
        scan_error["misaligned"] = fuzz.trapmf(scan_error.universe, [30, 90, 180, 180])

        scan_speed = ctrl.Consequent(np.arange(0, 1.01, 0.01), "scan_speed")
        scan_speed["slow"] = fuzz.trimf(scan_speed.universe, [0, 0.2, 0.4])
        scan_speed["medium"] = fuzz.trimf(scan_speed.universe, [0.3, 0.5, 0.7])
        scan_speed["fast"] = fuzz.trimf(scan_speed.universe, [0.6, 0.8, 1.0])

        rules = [
            ctrl.Rule(
                time_unseen["recent"] & scan_error["misaligned"], scan_speed["fast"]
            ),
            ctrl.Rule(
                time_unseen["recent"] & scan_error["aligned"], scan_speed["slow"]
            ),
            ctrl.Rule(time_unseen["moderate"], scan_speed["medium"]),
            ctrl.Rule(time_unseen["long"], scan_speed["medium"]),
        ]

        self.adaptive_scan_ctrl = ctrl.ControlSystem(rules)
        self.adaptive_scan_sim = ctrl.ControlSystemSimulation(self.adaptive_scan_ctrl)

    def _select_destructible_obstacle(
        self,
        my_x: float,
        my_y: float,
        seen_obstacles: List[Any],
    ) -> Optional[Any]:
        """Pick the closest destructible obstacle in sight."""
        destructible = [
            o
            for o in seen_obstacles
            if (
                o.get("is_destructible", False)
                if isinstance(o, dict)
                else getattr(o, "is_destructible", False)
            )
        ]
        if not destructible:
            return None

        best = None
        best_dist = float("inf")
        for obs in destructible:
            x, y = to_xy(
                obs.get("position", {})
                if isinstance(obs, dict)
                else getattr(obs, "_position", getattr(obs, "position", None))
            )
            d = euclidean_distance(my_x, my_y, x, y)
            if d < best_dist:
                best_dist = d
                best = obs
        return best

    def _select_target(
        self,
        my_x: float,
        my_y: float,
        seen_tanks: List[Any],
    ) -> Optional[Any]:
        if not seen_tanks:
            return None

        best_target = None
        best_priority = -1

        for tank in seen_tanks:
            x, y = to_xy(
                tank.get("position", {})
                if isinstance(tank, dict)
                else getattr(tank, "position", None)
            )
            distance = euclidean_distance(my_x, my_y, x, y)
            tank_type = (
                tank.get("tank_type", "LIGHT")
                if isinstance(tank, dict)
                else getattr(tank, "tank_type", "LIGHT")
            )

            threat_level = THREAT_WEIGHTS.get(tank_type, 5)

            try:
                max_dist = max(self.vision_range * 1.5, 30)
                self.target_selection_sim.input["distance"] = min(distance, max_dist)
                self.target_selection_sim.input["threat"] = threat_level
                self.target_selection_sim.compute()
                priority = self.target_selection_sim.output["priority"]

                if priority > best_priority:
                    best_priority = priority
                    best_target = tank
            except Exception:
                if best_target is None or distance < euclidean_distance(
                    my_x,
                    my_y,
                    *to_xy(
                        best_target.get("position", {})
                        if isinstance(best_target, dict)
                        else getattr(best_target, "position", None)
                    ),
                ):
                    best_target = tank

        return best_target

    def _calculate_rotation_speed(
        self,
        angle_error: float,
        distance: float,
    ) -> float:
        try:
            max_dist = max(self.vision_range * 1.5, 30)
            self.rotation_speed_sim.input["angle_error"] = min(abs(angle_error), 180)
            self.rotation_speed_sim.input["target_distance"] = min(distance, max_dist)
            self.rotation_speed_sim.compute()

            return float(self.rotation_speed_sim.output["speed_factor"])
        except Exception:
            if abs(angle_error) < 5:
                return 0.2
            elif abs(angle_error) < 30:
                return 0.6
            else:
                return 1.0

    def _should_fire_fuzzy(
        self,
        angle_error: float,
        distance: float,
        is_damaged: bool,
    ) -> bool:
        if self.cooldown_ticks > 0:
            return False

        try:
            vulnerability_score = 0.5
            if is_damaged:
                vulnerability_score = 0.9
            close_threshold = self.vision_range * 0.3
            if distance < close_threshold:
                vulnerability_score = min(1.0, vulnerability_score + 0.2)

            max_dist = max(self.vision_range * 1.5, 30)
            self.firing_decision_sim.input["aiming_error"] = min(abs(angle_error), 10)
            self.firing_decision_sim.input["firing_distance"] = min(distance, max_dist)
            self.firing_decision_sim.input["vulnerability"] = vulnerability_score
            self.firing_decision_sim.compute()

            fire_confidence = self.firing_decision_sim.output["fire_confidence"]

            return bool(fire_confidence >= FIRE_CONFIDENCE_THRESHOLD)
        except Exception:
            return bool(abs(angle_error) <= self.aim_threshold)

    def _adaptive_scan(
        self,
        current_barrel_angle: float,
        max_rotation: float,
    ) -> float:
        self.ticks_since_last_seen += 1

        if self.ticks_since_last_seen >= LAST_SEEN_EXPIRY_TICKS:
            self.last_seen_direction = None

        scan_direction_error = 90.0
        if self.last_seen_direction is not None:
            scan_direction_error = abs(
                normalize_angle_diff(self.last_seen_direction, current_barrel_angle)
            )

        try:
            self.adaptive_scan_sim.input["time_unseen"] = min(
                self.ticks_since_last_seen, 100
            )
            self.adaptive_scan_sim.input["scan_error"] = min(scan_direction_error, 180)
            self.adaptive_scan_sim.compute()

            speed_factor = float(self.adaptive_scan_sim.output["scan_speed"])

            if self.last_seen_direction is not None and scan_direction_error > 15:
                direction_diff = normalize_angle_diff(
                    self.last_seen_direction, current_barrel_angle
                )
                rotation = speed_factor * max_rotation * np.sign(direction_diff)
            else:
                rotation = speed_factor * max_rotation

            return float(max(-max_rotation, min(max_rotation, rotation)))
        except Exception:
            return float(max(-max_rotation, min(max_rotation, 22.0)))

    @staticmethod
    def select_ammo(
        enemy_distance: Optional[float],
        ammo_stocks: dict[str, int],
        current_ammo: Optional[str],
    ) -> Optional[str]:
        for name, amount in ammo_stocks.items():
            if amount > 0:
                return name
        return None

    def update(
        self,
        my_x: float,
        my_y: float,
        my_heading: float,
        current_barrel_angle: float,
        seen_tanks: List[Any],
        max_barrel_rotation: float,
        ammo_stocks: Optional[dict[str, int]] = None,
        current_ammo: Optional[str] = None,
        seen_obstacles: Optional[List[Any]] = None,
    ) -> Tuple[float, bool, Optional[str]]:
        if self.cooldown_ticks > 0:
            self.cooldown_ticks -= 1

        target = self._select_target(my_x, my_y, seen_tanks) if seen_tanks else None

        if target is None:
            rotation = self._adaptive_scan(current_barrel_angle, max_barrel_rotation)
            ammo = self.select_ammo(None, ammo_stocks or {}, current_ammo)
            return rotation, False, ammo

        self.ticks_since_last_seen = 0

        target_x, target_y = to_xy(
            target.get("position", {})
            if isinstance(target, dict)
            else getattr(target, "position", None)
        )
        distance = euclidean_distance(my_x, my_y, target_x, target_y)

        absolute_angle = heading_to_angle_deg(my_x, my_y, target_x, target_y)
        relative_angle = normalize_angle_diff(absolute_angle, my_heading)
        self.last_seen_direction = relative_angle

        angle_error = normalize_angle_diff(relative_angle, current_barrel_angle)

        speed_factor = self._calculate_rotation_speed(angle_error, distance)
        rotation = speed_factor * self.max_barrel_spin_rate * np.sign(angle_error)
        rotation = max(-max_barrel_rotation, min(max_barrel_rotation, rotation))

        is_damaged = (
            target.get("is_damaged", False)
            if isinstance(target, dict)
            else getattr(target, "is_damaged", False)
        )
        should_fire = self._should_fire_fuzzy(abs(angle_error), distance, is_damaged)

        if should_fire:
            self.cooldown_ticks = COOLDOWN_TICKS

        ammo = self.select_ammo(distance, ammo_stocks or {}, current_ammo)

        return float(rotation), bool(should_fire), ammo
