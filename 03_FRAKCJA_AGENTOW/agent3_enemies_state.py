import numpy as np
from agent3_agent_state import calculate_distance, calculate_angle_to_target, normalize_angle


def prepare_visible_tanks_states(payload: dict, position: tuple):
    features_by_tank = {}
    my_team = payload['my_tank_status']['_team']
    for visible_tank in payload['sensor_data']['seen_tanks']:
        if visible_tank['team'] == my_team:
            continue

        tank_position = (visible_tank['position']['x'], visible_tank['position']['y'])
        dist = 1 - calculate_distance(position, tank_position) / payload['my_tank_status']['_vision_range']

        damaged = 1 if visible_tank['is_damaged'] else 0

        heading_direction = calculate_angle_to_target(visible_tank['position'], {'x': position[0], 'y': position[1]})
        if abs(visible_tank['heading']-heading_direction) < 30:
            heading_coef = 1 - abs(visible_tank['heading']-heading_direction)/30
        elif 150 < abs(visible_tank['heading']-heading_direction) < 210:
            heading_coef = 1 - abs(180-abs(visible_tank['heading']-heading_direction))/30
        else:
            heading_coef = 0

        type_coef = 0.5 if visible_tank['tank_type'] == 'LIGHT' else (0.75 if visible_tank['tank_type'] == 'Sniper' else 1)

        shoot_direction = normalize_angle(visible_tank['heading'] + visible_tank['barrel_angle'])
        angle_to_target = normalize_angle(
            calculate_angle_to_target(visible_tank['position'], {'x': position[0], 'y': position[1]}))
        if 10 < abs(normalize_angle(angle_to_target - shoot_direction)) <= 15:
            aiming_coef = 0.33
        elif 5 < abs(normalize_angle(angle_to_target - shoot_direction)) <= 10:
            aiming_coef = 0.66
        elif abs(normalize_angle(angle_to_target - shoot_direction)) <= 5:
            aiming_coef = 1
        else:
            aiming_coef = 0

        features = [
            dist,
            damaged,
            heading_coef,
            type_coef,
            aiming_coef
        ]
        features_by_tank[visible_tank['id']] = np.array(features)

    return features_by_tank


def decide_if_should_shot(payload, tank_position, angle, max_dist):
    if payload['my_tank_status']['_reload_timer'] > 0:
        return False

    if payload['my_tank_status']['ammo_loaded'] is None:
        return False

    angle_to_target = normalize_angle(calculate_angle_to_target(payload['my_tank_status']['position'], tank_position))
    if abs(angle) > 1.5:
        return False

    dist = calculate_distance((payload['my_tank_status']['position']['x'], payload['my_tank_status']['position']['y']),
                              (tank_position['x'], tank_position['y']))
    current_ammo = payload['my_tank_status']['ammo_loaded']
    if dist > max_dist[current_ammo]:
        return False

    my_team = payload['my_tank_status']['_team']
    for tank in payload['sensor_data']['seen_tanks']:
        if tank['team'] == my_team:
            ally_dist = calculate_distance((payload['my_tank_status']['position']['x'], payload['my_tank_status']['position']['y']),
                                           (tank['position']['x'], tank['position']['y']))
            angle_to_ally = normalize_angle(calculate_angle_to_target(payload['my_tank_status']['position'], tank['position']))
            if ally_dist < dist and abs(angle_to_target - angle_to_ally) <= 5:
                return False

    return True
