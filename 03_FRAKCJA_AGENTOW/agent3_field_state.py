import numpy as np
from agent3_agent_memory import Memory
from agent3_agent_state import calculate_distance

MEMORY_DECAY_TICKS = 500
MAX_DIST_MOVE = 55
MAX_DIST_TANKS = 80
MAX_DIST_POWERUP = 40

def check_possible_fields(payload: dict, memory: Memory) -> dict:
    current_x = payload['my_tank_status']['position']['x']
    current_y = payload['my_tank_status']['position']['y']

    start_x = int(current_x - MAX_DIST_MOVE + 5) if current_x > MAX_DIST_MOVE else 5
    start_y = int(current_y - MAX_DIST_MOVE + 5) if current_y > MAX_DIST_MOVE else 5
    end_x = int(current_x + MAX_DIST_MOVE - 5) if current_x < 200 - MAX_DIST_MOVE else 195
    end_y = int(current_y + MAX_DIST_MOVE - 5) if current_y < 200 - MAX_DIST_MOVE else 195

    field_states = {}

    for x in range(start_x, end_x+1, 5):
        for y in range(start_y, end_y+1, 5):
            if calculate_distance((current_x, current_y), (x, y)) < MAX_DIST_MOVE:
                value = prepare_field_state_from_data(payload, (x, y), memory)
                if value is not None:
                    field_states[(x, y)] = np.array(value)

    return field_states


def check_decay(current_tick, last_seen_tick):
    return current_tick - last_seen_tick < MEMORY_DECAY_TICKS


def prepare_field_state_from_data(payload: dict, position: tuple, memory: Memory):
    tick = payload['current_tick']
    terrain_info = memory.get_terrain_info(position[0], position[1])
    tank_info1 = memory.get_tank_info(position[0], position[1])
    if terrain_info is not None and terrain_info.is_obstacle and not terrain_info.is_destructible:
        return None
    # cost in dmg
    if tank_info1 is not None and check_decay(tick, tank_info1.last_seen):
        field_cost = 1
    elif (terrain_info is not None and terrain_info.is_obstacle) or terrain_info is None:
        field_cost = 0.66
    elif terrain_info is not None and terrain_info.deal_dmg:
        field_cost = 0.33
    else:
        field_cost = 0
    # dist
    current_position = (payload['my_tank_status']['position']['x'], payload['my_tank_status']['position']['y'])
    dist = calculate_distance(current_position, position) / MAX_DIST_MOVE
    # enemies nearby
    # allies nearby
    start_x = int(position[0] - MAX_DIST_TANKS + 5) if position[0] > MAX_DIST_TANKS else 5
    start_y = int(position[1] - MAX_DIST_TANKS + 5) if position[1] > MAX_DIST_TANKS else 5
    end_x = int(position[0] + MAX_DIST_TANKS - 5) if position[0] < 200 - MAX_DIST_TANKS else 195
    end_y = int(position[1] + MAX_DIST_TANKS - 5) if position[1] < 200 - MAX_DIST_TANKS else 195
    enemies_close = 0
    allies_close = 0
    all_checked = 0
    certain = 0
    for x in range(start_x, end_x + 1, 5):
        for y in range(start_y, end_y + 1, 5):
            if calculate_distance(position, (x, y)) < MAX_DIST_TANKS:
                all_checked += 1
                tank_data = memory.get_tank_info(x, y)
                if tank_data is not None and check_decay(tick, tank_data.last_seen):
                    certain += (1 - ((tick - tank_data.last_seen) / MEMORY_DECAY_TICKS))
                    value = (0.5 if tank_data.is_damaged else 1) * (1 - ((tick - tank_data.last_seen) / MEMORY_DECAY_TICKS))
                    if tank_data.team:
                        allies_close += value
                    else:
                        enemies_close += value
        enemies_close /= 3
        allies_close /= 3
        enemies_close += 0.5*(certain / all_checked)
        allies_close += 0.2*(certain / all_checked)
        enemies_close = enemies_close if enemies_close < 1 else 1
        allies_close = allies_close if allies_close < 1 else 1
    # ammo nearby
    # medkit nearby
    # shield nearby
    start_x = int(position[0] - MAX_DIST_POWERUP + 1) if position[0] > MAX_DIST_POWERUP else 1
    start_y = int(position[1] - MAX_DIST_POWERUP + 1) if position[1] > MAX_DIST_POWERUP else 1
    end_x = int(position[0] + MAX_DIST_POWERUP - 1) if position[0] < 500 - MAX_DIST_POWERUP else 499
    end_y = int(position[1] + MAX_DIST_POWERUP - 1) if position[1] < 500 - MAX_DIST_POWERUP else 499
    closest_ammo = 10000000
    closest_medkit = 10000000
    closest_shield = 10000000
    closest_powerup = 10000000
    for x in range(start_x, end_x + 1, 5):
        for y in range(start_y, end_y + 1, 5):
            powerup_dist = calculate_distance(position, (x, y))
            if powerup_dist < MAX_DIST_POWERUP:
                powerup_data = memory.get_powerup_info(x, y)
                if powerup_data is not None and check_decay(tick, powerup_data.last_seen):
                    if powerup_dist < closest_powerup:
                        closest_powerup = powerup_dist
                    if powerup_data.powerup_type == 'PowerUpType.MEDKIT' and powerup_dist < closest_medkit:
                        closest_medkit = powerup_dist
                    elif powerup_data.powerup_type == 'PowerUpType.SHIELD' and powerup_dist < closest_shield:
                        closest_shield = powerup_dist
                    elif powerup_data.powerup_type.startswith('PowerUpType.AMMO') and powerup_dist < closest_ammo:
                        closest_ammo = powerup_dist

    closest_ammo /= MAX_DIST_POWERUP
    closest_medkit /= MAX_DIST_POWERUP
    closest_shield /= MAX_DIST_POWERUP
    closest_powerup /= MAX_DIST_POWERUP
    closest_ammo = closest_ammo if closest_ammo < 1 else 1
    closest_medkit = closest_medkit if closest_medkit < 1 else 1
    closest_shield = closest_shield if closest_shield < 1 else 1
    closest_powerup = closest_powerup if closest_powerup < 1 else 1

    # obstacles nearby
    obstacles_nearby = 0
    for x_change in [-10, 0, 10]:
        for y_change in [-10, 0, 10]:
            terrain = memory.get_terrain_info(current_position[0]+x_change, current_position[1]+y_change)
            if terrain is not None and terrain.is_obstacle is True:
                if terrain.is_destructible:
                    obstacles_nearby += 0.5
                else:
                    obstacles_nearby += 1

    obstacles_nearby /= 8

    return [
        field_cost,
        dist,
        enemies_close,
        allies_close,
        closest_ammo,
        closest_medkit,
        closest_shield,
        closest_powerup,
        obstacles_nearby
    ]
