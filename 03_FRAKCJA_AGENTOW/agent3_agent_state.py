import math
import numpy as np
from agent3_agent_memory import Memory


def calculate_angle_to_target(obj_from, obj_to) -> float:
    dx = obj_to['x'] - obj_from['x']
    dy = obj_to['y'] - obj_from['y']
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def normalize_angle(angle) -> float:
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def calculate_distance(pos1, pos2) -> float:
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return math.hypot(dx, dy)


def prepare_agent_state_from_data(payload: dict, memory: Memory) -> np.array:
    my_team = payload['my_tank_status']['_team']
    current_hp = payload['my_tank_status']['hp']
    max_hp = payload['my_tank_status']['_max_hp']
    current_shield = payload['my_tank_status']['shield']
    max_shield = payload['my_tank_status']['_max_shield']
    is_overcharged = 1 if payload['my_tank_status']['is_overcharged'] is True else 0
    ammo_count = sum([ammo['count'] for ammo in payload['my_tank_status']['ammo'].values()])

    enemies_tanks = [tank for tank in payload['sensor_data']['seen_tanks'] if tank['team'] != my_team]
    min_dist_to_enemy = min([tank['distance'] for tank in enemies_tanks]) \
        if len(enemies_tanks) > 0 else payload['my_tank_status']['_vision_range']
    enemies_health = sum([1 for tank in enemies_tanks if tank['is_damaged'] is False]) \
        if len(enemies_tanks) > 0 else 0

    aiming_enemies = 0
    for tank in enemies_tanks:
        shoot_direction = normalize_angle(tank['heading'] + tank['barrel_angle'])
        angle_to_target = normalize_angle(calculate_angle_to_target( payload['my_tank_status']['position'], tank['position']))
        if abs(normalize_angle(angle_to_target - shoot_direction)) <= 5:
            aiming_enemies += 1

    allies_tanks = len(payload['sensor_data']['seen_tanks']) - len(enemies_tanks)

    x = payload['my_tank_status']['position']['x']
    y = payload['my_tank_status']['position']['y']

    x_start = int(x-20 if x > 20 else 0)
    x_end = int(x+20 if x < 480 else 500)
    xs = [x for x in range(x_start, x_end+1, 10)]
    y_start = int(y-20 if y > 20 else 0)
    y_end = int(y+20 if y < 480 else 500)
    ys = [y for y in range(y_start, y_end+1, 10)]
    obstacle_close = 0
    for x in xs:
        if obstacle_close > 0:
            break
        for y in ys:
            terrain_info = memory.get_terrain_info(x, y)
            if terrain_info is not None and terrain_info.is_obstacle:
                obstacle_close = 1
                break

    state = [
        current_hp/max_hp,
        current_shield/max_shield,
        1 if ammo_count/20 >= 1 else ammo_count/20,
        is_overcharged,
        payload['enemies_remaining'] / 5,
        min_dist_to_enemy / payload['my_tank_status']['_vision_range'],
        enemies_health / payload['enemies_remaining'] if payload['enemies_remaining'] > 0 else 0,
        aiming_enemies / payload['enemies_remaining'] if payload['enemies_remaining'] > 0 else 0,
        allies_tanks / 4,
        obstacle_close,
    ]

    return state
