from __future__ import annotations
import math
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .api import (
    Position, TankUnion, TankSensorData, ObstacleUnion, 
    TerrainUnion, PowerUpData, SeenTank, PowerUpType, AmmoType
)


class RadarModule:
    def __init__(self):
        self.enemies: List[Dict[str, Any]] = []
        self.allies: List[Dict[str, Any]] = []

    def update(self, my_tank: Dict, seen_tanks: List[Dict]):
        self.enemies.clear()
        self.allies.clear()
        
        for tank in seen_tanks:
            dist = tank.get("distance")
        
            if tank["team"] == my_tank["_team"]:
                self.allies.append(tank)
            else:
                self.enemies.append({
                    "tank_data": tank,
                    "dist": dist,
                    "is_low_hp": tank.get("is_damaged", False),
                    "is_aiming_at_me": self._is_aiming(tank, my_tank)
                })
        self.enemies.sort(key=lambda x: x['dist'])

    def _is_aiming(self, enemy: Dict, my_tank: Dict) -> bool:
        dx, dy = my_tank["position"]["x"] - enemy["position"]["x"], my_tank["position"]["y"] - enemy["position"]["y"]
        angle_to_me = math.degrees(math.atan2(dy, dx))
        diff = (enemy["barrel_angle"] - angle_to_me + 180) % 360 - 180
        return abs(diff) < 12.0

class LogisticsModule:
    def __init__(self):
        self.closest_powerups: Dict[str, Dict[str, Any]] = {}

    def update(self, my_pos: Dict, seen_powerups: List[Dict[str, Any]]) -> None:
        self.closest_powerups.clear()

        mx = float(my_pos["x"])
        my = float(my_pos["y"])

        for pu in seen_powerups:
            try:
                pos = pu["position"]
                px = float(pos["x"])
                py = float(pos["y"])
                dist = float(math.hypot(mx - px, my - py))

                raw_type = pu["powerup_type"]  # np. "PowerUpType.SHIELD"
                if not isinstance(raw_type, str):
                    # jeśli kiedyś wróci enum, to zadziała też na nim:
                    enum_obj = raw_type
                else:
                    enum_name = raw_type.split(".")[-1]  # "SHIELD"
                    enum_obj = PowerUpType[enum_name]

                meta = enum_obj.value  # {"Name": ..., "Value": ...}
                p_type = str(meta["Name"])
                val = int(meta["Value"])
                pu_id = str(pu["id"])
            except Exception:
                continue

            prev = self.closest_powerups.get(p_type)
            if prev is None or dist < float(prev["dist"]):
                self.closest_powerups[p_type] = {
                    "dist": dist,
                    "pos": {"x": px, "y": py},
                    "val": val,
                    "_id": pu_id,
                }


    # def update(self, my_pos: Dict, seen_powerups: List[Dict]):
    #     self.closest_powerups.clear()
    #     for pu in seen_powerups:
    #         try:
    #             p_type = pu.get("_powerup_type", {}).get("Name", "UNKNOWN")
    #             dist = math.sqrt((my_pos["x"] - pu["_position"]["x"])**2 + 
    #                             (my_pos["y"] - pu["_position"]["y"])**2)
                
    #             if p_type not in self.closest_powerups or dist < self.closest_powerups[p_type]['dist']:
    #                 self.closest_powerups[p_type] = {
    #                     "dist": dist,
    #                     "pos": pu["_position"],
    #                     "val": pu.get("value", 0)
    #                 }
    #         except KeyError:
    #             pass

class BallisticsModule:
    def get_range(self, tank: Dict) -> float:
        ammo_type = tank.get("ammo_loaded")
        if not ammo_type:
            return 0.0
        ranges = {"HEAVY": 12.5, "LIGHT": 40.0, "LONG_DISTANCE": 85.0}
        return float(ranges.get(ammo_type, 0.0))

    def get_rotation_to_target(self, my_tank: Dict, target_pos: Dict) -> float:
        dx, dy = target_pos["x"] - my_tank["position"]["x"], target_pos["y"] - my_tank["position"]["y"]
        
        # Korekta dla układu współrzędnych ekranowych (Y rośnie w dół):
        # W standardowym math.atan2(y, x), kąt rośnie CCW (przeciwnie do zegara).
        # W układzie ekranowym często kąty rosną CW (zgodnie z zegarem) lub Y jest odwrócone.
        # Jeśli 0 stopni = Prawo (X+), a 90 stopni = Dół (Y+), to atan2(dy, dx) zwróci poprawną wartość kąta,
        # ALE musimy upewnić się czy system gry używa tego samego zwrotu.
        # Zazwyczaj wystarczy po prostu atan2(dy, dx) jeśli 0 to 'X+', a kąty rosną w stronę 'Y+'.
        # Jeśli jednak kąty gry są "matematyczne" (Y+ to góra), a ekran ma Y+ w dół, to trzeba odwrócić Y przy obliczeniach.
        # Zakładamy tutaj, że skoro system jest Y-down (ekranowy), to kąt też powinien to odzwierciedlać.
        
        target_angle = math.degrees(math.atan2(dy, dx))
        
        # Normalizacja obu kątów przed odejmowaniem
        current_barrel_angle = (my_tank["barrel_angle"] + 180) % 360 - 180
        target_angle = (target_angle + 180) % 360 - 180
        
        diff = target_angle - current_barrel_angle
        return (diff + 180) % 360 - 180

    def is_line_of_fire_clear(self, my_pos: Dict, target_pos: Dict, allies: List[Dict]) -> bool:
        dx, dy = target_pos["x"] - my_pos["x"], target_pos["y"] - my_pos["y"]
        dist = math.sqrt(dx**2 + dy**2)
        if dist < 1: return True

        for ally in allies:
            ax, ay = ally["position"]["x"], ally["position"]["y"]
            d_ally = math.sqrt((ax - my_pos["x"])**2 + (ay - my_pos["y"])**2)
            if d_ally >= dist: continue
            
            cross_product = abs(dy * ax - dx * ay + target_pos["x"] * my_pos["y"] - target_pos["y"] * my_pos["x"])
            if (cross_product / dist) < 6.0:
                return False
        return True
    
class EnvironmentModule:
    def __init__(self, grid_size: int = 10, map_size: tuple = (500, 500)):
        self.obstacles: Dict[str, Dict] = {}
        self.terrains: Dict[str, Dict] = {}
        self.current_terrain: Optional[Dict] = None
        self.grid_size = grid_size
        self.map_w, self.map_h = map_size

    def update(self, my_pos: Dict, sensor_data: Dict):
        for obs in sensor_data.get("seen_obstacles", []):
            self.obstacles[obs["id"]] = obs
            
        for ter in sensor_data.get("seen_terrains", []):
            # Using grid coordinates as keys for faster terrain lookup
            gx, gy = self.world_to_grid(ter["position"])
            self.terrains[f"{gx}_{gy}"] = ter

        self.current_terrain = self._find_terrain_at(my_pos)

    def _find_terrain_at(self, pos: Dict) -> Optional[Dict]:
        gx, gy = self.world_to_grid(pos)
        return self.terrains.get(f"{gx}_{gy}")

    def get_navigation_graph(self) -> Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]:
        """
        Returns a graph where each node maps to a list of (neighbor_tuple, edge_weight).
        Weight = Step Distance + Terrain Damage.
        """
        graph = {}
        cols, rows = self.map_w // self.grid_size, self.map_h // self.grid_size
        
        for x in range(cols):
            for y in range(rows):
                if self._is_cell_blocked(x, y): 
                    continue
                
                node = (x, y)
                neighbors = []
                
                # Check 8-way movement
                for dx, dy in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                    nx, ny = x + dx, y + dy
                    
                    if 0 <= nx < cols and 0 <= ny < rows:
                        if not self._is_cell_blocked(nx, ny):

                            # Calculate movement cost
                            # 1. Base distance (1.0 for orthogonal, ~1.41 for diagonal)
                            base_cost = (dx**2 + dy**2)**0.5
                            
                            # 2. Add damage penalty from the destination cell
                            terrain = self.terrains.get(f"{nx}_{ny}")
                            danger_penalty = terrain.get("dmg", 0) if terrain else 0
                            
                            total_cost = base_cost + danger_penalty
                            neighbors.append(((nx, ny), total_cost))
                
                graph[node] = neighbors
        return graph

    def _is_cell_blocked(self, gx: int, gy: int) -> bool:
        wx, wy = gx * self.grid_size + 5, gy * self.grid_size + 5
        for obs in self.obstacles.values():
            ox, oy = obs["position"]["x"], obs["position"]["y"]
            # Simplified collision check
            if abs(ox - wx) < 8 and abs(oy - wy) < 8: #changed form 8 to 15
                return True
        return False

    def world_to_grid(self, pos: Dict) -> Tuple[int, int]:
        return (int(pos["x"] // self.grid_size), int(pos["y"] // self.grid_size))
    
    def get_movement_multiplier(self, pos: Optional[Dict] = None) -> float:
        terrain = self._find_terrain_at(pos) if pos else self.current_terrain
        if not terrain: 
            return 1.0
        return terrain.get("speed_modifier", 1.0)

    def get_terrain_danger(self, pos: Optional[Dict] = None) -> int:
        terrain = self._find_terrain_at(pos) if pos else self.current_terrain
        if not terrain: 
            return 0
        return terrain.get("dmg", 0)

class BattlefieldObserver:
    def __init__(self):
        self.radar = RadarModule()
        self.logistics = LogisticsModule()
        self.ballistics = BallisticsModule()
        self.env = EnvironmentModule()
        self.my_tank = {}
        self.enemies_left = 0
        
    def update(self, *data_packet: list):
        self.my_tank = data_packet[0]
        sensor_data = data_packet[1]
        self.enemies_left = data_packet[2]

        self.radar.update(self.my_tank, sensor_data.get("seen_tanks", []))
        self.logistics.update(self.my_tank["position"], sensor_data.get("seen_powerups", []))
        self.env.update(self.my_tank["position"], sensor_data)

    def set_training_mode(self, enabled: bool) -> None:
        self.training_mode = enabled

    def _can_shoot(self, nearest: Optional[Dict], w_range: float) -> bool:
        if not nearest or self.my_tank.get("_reload_timer", 0) > 0:
            return False
        
        in_range = nearest['dist'] < w_range
        clear_line = self.ballistics.is_line_of_fire_clear(
            self.my_tank["position"], nearest['tank_data']["position"], self.radar.allies
        )
        return in_range and clear_line

    def is_obstacle_ahead(self, max_distance: float = 30.0, step: float = 2.0) -> bool:
        """Sprawdza linię przed czołgiem w poszukiwaniu przeszkód (dokładne pozycje)."""
        if not self.my_tank: return False

        pos = self.my_tank["position"]
        heading_rad = math.radians(self.my_tank.get("heading", 0.0))
        
        # Skanujemy co 'step' (np. 2 jednostki), co daje dużą precyzję
        for d in np.arange(step, max_distance + step, step):
            target_x = pos["x"] + d * math.cos(heading_rad)
            target_y = pos["y"] + d * math.sin(heading_rad)
            
            # Przeszukujemy listę wszystkich znanych nam przeszkód
            for obs in self.env.obstacles.values():
                ox = obs["position"]["x"]
                oy = obs["position"]["y"]
                
                # METODA OKRĄGŁEGO HITBOXA (Zalecana)
                # Oblicza dokładną odległość w linii prostej między sprawdzanym punktem a przeszkodą.
                # Kolizja z okręgiem sprawia, że czołg płynniej ześlizguje się z narożników.
                dist = math.hypot(target_x - ox, target_y - oy)
                
                # 8.0 to tolerancja (wielkość przeszkody). 
                # W API mają rozmiar [10, 10], więc promień 7-8 jednostek jest optymalny.
                if dist < 8.0: 
                    return True
                    

        return False

    def get_summary(self) -> Dict[str, Any]:
        nearest = self.radar.enemies[0] if self.radar.enemies else None
        curr_range = self.ballistics.get_range(self.my_tank)
        reload_ticks = self.my_tank.get("_reload_timer", 0)
        
        return {
            "self": {
                "hp_pct": (self.my_tank["hp"] / self.my_tank["_max_hp"]) * 100,
                "is_ready": reload_ticks == 0,
                "reload_ticks": reload_ticks,
                "pos": self.my_tank["position"],
                "heading": self.my_tank["heading"],
                "barrel_angle": self.my_tank["barrel_angle"],
                "shield": self.my_tank["shield"],
                "speed_mod": self.env.get_movement_multiplier(),
                "terrain_damage": self.env.get_terrain_danger(),
                "obstacle_ahead": self.is_obstacle_ahead(max_distance=5, step=1)
            },
            "tactical": {
                "can_fire": self._can_shoot(nearest, curr_range),
                "rotation_to_target": self.ballistics.get_rotation_to_target(
                    self.my_tank, nearest['tank_data']["position"]
                ) if nearest else 0.0,
            },
            "radar": {
                "nearest_enemy": nearest,
                "enemies_left": self.enemies_left
            },
            "logistics": {
                "powerups": self.logistics.closest_powerups
            }
        }
