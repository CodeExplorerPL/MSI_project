from dataclasses import dataclass

@dataclass
class TankData:
    team: bool
    is_damaged: bool
    last_seen: int
    tank_id: str


@dataclass
class PowerupData:
    powerup_type: str
    last_seen: int


@dataclass
class TerrainData:
    is_obstacle: bool
    deal_dmg: bool
    is_destructible: bool
    speed_modifier: float


class Memory:
    def __init__(self):
        self._terrain_memory = {}
        self._powerups_memory = {}
        self._tanks_memory = {}

    def refresh_memory(self, payload: dict):
        my_team = payload['my_tank_status']['_team']
        for obstacle in payload['sensor_data']["seen_obstacles"]:
            position = (int(obstacle['position']['x']) // 10, int(obstacle['position']['y']) // 10)
            self._terrain_memory[position] = TerrainData(True, True, obstacle['is_destructible'], 1 if obstacle['is_destructible'] else 0)
        for terrain in payload['sensor_data']["seen_terrains"]:
            position = (int(terrain['position']['x']) // 10, int(terrain['position']['y']) // 10)
            self._terrain_memory[position] = TerrainData(False, True if terrain['dmg'] > 0 else False, False, terrain['speed_modifier'])
        for powerup in payload['sensor_data']["seen_powerups"]:
            position = (int(powerup['position']['x']) // 2, int(powerup['position']['y']) // 2)
            self._powerups_memory[position] = PowerupData(powerup['powerup_type'], payload['current_tick'])
        for tank in payload['sensor_data']["seen_tanks"]:
            for pos, tank_data in self._tanks_memory.items():
                if tank_data.tank_id == tank['id']:
                    del self._tanks_memory[pos]
                    break
            position = (int(tank['position']['x']) // 5, int(tank['position']['y']) // 5)
            self._tanks_memory[position] = TankData(tank['team'] == my_team, tank['is_damaged'], payload['current_tick'], tank['id'])

    def get_terrain_info(self, x, y) -> TerrainData | None:
        position = (int(x) // 10, int(y) // 10)
        return self._terrain_memory.get(position, None)

    def get_powerup_info(self, x, y) -> PowerupData | None:
        position = (int(x) // 2, int(y) // 2)
        return self._powerups_memory.get(position, None)

    def get_tank_info(self, x, y) -> TankData | None:
        position = (int(x) // 5, int(y) // 5)
        return self._tanks_memory.get(position, None)

    def get_neighbour_tanks(self, x, y) -> int:
        count = 0
        for dx in range(-40, 41, 5):
            for dy in range(-40, 41, 5):
                if self.get_tank_info(x+dx, y+dy) is not None:
                    count += 1

        return count

