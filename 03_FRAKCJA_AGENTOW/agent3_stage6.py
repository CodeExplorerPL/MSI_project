from agent3_stage4 import MyFuzzyRule, MyFuzzySimulation, MyFuzzyVariable
from agent3_agent_state import calculate_distance

dist = MyFuzzyVariable('dist', 0, 1)
damaged = MyFuzzyVariable('damaged', 0, 1)

dist.add_trapezium_membership_func('low', 0, 0, .2, .25)
dist.add_trapezium_membership_func('medium', .2, .25, .4, .5)
dist.add_trapezium_membership_func('high', .4, .5, 1, 1)

damaged.add_trapezium_membership_func('no', 0, 0, .3, .7)
damaged.add_trapezium_membership_func('yes', .3, .7, 1, 1)

rules = [
    MyFuzzyRule([(dist, 'high')], 1),
    MyFuzzyRule([(dist, 'medium'), (damaged, 'yes')], 0.4),
    MyFuzzyRule([(dist, 'medium'), (damaged, 'no')], 0.6),
    MyFuzzyRule([(dist, 'low'), (damaged, 'yes')], 0),
    MyFuzzyRule([(dist, 'low'), (damaged, 'no')], 0.25),
]

def create_fuzzy_model_stage6():
    sim = MyFuzzySimulation()
    sim.add_variable(dist)
    sim.add_variable(damaged)
    for rule in rules:
        sim.add_rule(rule)

    return sim

def get_preferable_ammo(sim, position, target, ammo_payload):
    if target is None:
        return max(list(ammo_payload.values()), key=lambda data: data['count'])['_ammo_type']

    if len(ammo_payload) == 1 or sum([1 for v in ammo_payload.values() if v['count'] > 0]) == 1:
        for ammo in ammo_payload.values():
            if ammo['count'] > 0:
                return ammo['_ammo_type']

    target_position = (target['position']['x'], target['position']['y'])
    distance = calculate_distance(position, target_position)

    distance /= 100
    distance = distance if distance <= 1 else 1

    is_damaged = 1 if target['is_damaged'] is True else 0

    sim.input(dist, distance)
    sim.input(damaged, is_damaged)
    value = sim.compute()
    if value <= 0.25 and ammo_payload['HEAVY']['count'] > 0:
        return 'HEAVY'
    elif value >= 0.75 and ammo_payload['LONG_DISTANCE']['count'] > 0:
        return 'LONG_DISTANCE'
    else:
        return 'LIGHT'
