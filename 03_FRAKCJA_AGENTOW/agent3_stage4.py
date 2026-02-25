class MyFuzzyVariable:
    def __init__(self, name: str, range_min: float, range_max: float):
        self.name = name
        self.range = (range_min, range_max)
        self.membership_funcs_triangle = {}
        self.membership_funcs_trapezium = {}

    @staticmethod
    def _triangle_membership_func(x: float, a: float, b: float, c: float):
        if x <= a or x >= c:
            return 0.0

        if x <= b:
            if a == b:
                return 1.0
            return (x - a) / (b - a)

        if b == c:
            return 1.0
        return (c - x) / (c - b)

    @staticmethod
    def _trapezium_membership_func(x: float, a: float, b: float, c: float, d: float):
        if x < a or x > d:
            return 0.0

        if a == b and x <= b:
            return 1.0

        if x < b:
            return (x - a) / (b - a)

        if x <= c:
            return 1.0

        if c == d:
            return 1.0

        return (d - x) / (d - c)

    def add_triangle_membership_func(self, name: str, a: float, b: float, c: float):
        self.membership_funcs_triangle[name] = (lambda x: self._triangle_membership_func(x, a, b, c))

    def add_trapezium_membership_func(self, name: str, a: float, b: float, c: float, d: float):
        self.membership_funcs_trapezium[name] = (lambda x: self._trapezium_membership_func(x, a, b, c, d))

    def membership(self, name: str, x: float):
        if name in self.membership_funcs_triangle.keys():
            return self.membership_funcs_triangle[name](x)
        elif name in self.membership_funcs_trapezium.keys():
            return self.membership_funcs_trapezium[name](x)
        else:
            raise ValueError(f'No membership function with name = {name}')

class MyFuzzyRule:
    def __init__(self, antecedents: list[list | tuple], consequent: float):
        self.antecedents = antecedents
        self.consequent = consequent

    def check(self, inputs: dict):
        return MyFuzzyRule._weight(inputs, self.antecedents, True)

    @staticmethod
    def _weight(inputs: dict, antecedents: tuple | list, minimum: bool):
        if minimum:
            full_membership = 1
        else:
            full_membership = 0
        for antecedent in antecedents:
            if type(antecedent) == tuple:
                variable, name = antecedent
                membership = variable.membership(name, inputs[variable.name])
            else:
                membership = MyFuzzyRule._weight(inputs, antecedent, False)
            if membership < full_membership and minimum:
                full_membership = membership
            elif membership > full_membership and not minimum:
                full_membership = membership
        return full_membership


class MyFuzzySimulation:
    def __init__(self):
        self.variables = []
        self.rules = []
        self.inputs = {}

    def add_variable(self, variable: MyFuzzyVariable):
        self.variables.append(variable)

    def add_rule(self, rule: MyFuzzyRule):
        self.rules.append(rule)

    def input(self, input_variable: MyFuzzyVariable, input_value: float):
        self.inputs[input_variable.name] = input_value

    def compute(self):
        weighted_sum = 0
        weight_sum = 0
        for rule in self.rules:
            weight = rule.check(self.inputs)
            weighted_sum += weight * rule.consequent
            weight_sum += weight
        return weighted_sum / weight_sum if weight_sum != 0 else 0


health = MyFuzzyVariable('health + shield', 0, 1)
enemies_around = MyFuzzyVariable('enemies_around', 0, 1)
enemies_threat = MyFuzzyVariable('enemies_threat', 0, 1)

health.add_trapezium_membership_func('low', 0, 0, .2, .3)
health.add_trapezium_membership_func('medium', .2, .3, .5, .75)
health.add_trapezium_membership_func('high', .5, .75, 1, 1)

enemies_around.add_triangle_membership_func('none', 0, 0, .4)
enemies_around.add_triangle_membership_func('few', 0, .4, .7)
enemies_around.add_trapezium_membership_func('many', .4, .7, 1, 1)

enemies_threat.add_trapezium_membership_func('none', 0, 0, .1, .2)
enemies_threat.add_triangle_membership_func('low', .1, .2, .6)
enemies_threat.add_triangle_membership_func('high', .2, .6, .8)
enemies_threat.add_trapezium_membership_func('max', .6, .8, 1, 1)


rules = [
    MyFuzzyRule([(health, 'low'), (enemies_around, 'none')], .1),
    MyFuzzyRule([(health, 'low'), (enemies_around, 'few'), [(enemies_threat, 'none'), (enemies_threat, 'low')]], .25),
    MyFuzzyRule([(health, 'low'), (enemies_around, 'few'), [(enemies_threat, 'high'), (enemies_threat, 'max')]], .5),
    MyFuzzyRule([(health, 'low'), (enemies_around, 'many'), [(enemies_threat, 'none'), (enemies_threat, 'low')]], .4),
    MyFuzzyRule([(health, 'low'), (enemies_around, 'many'), [(enemies_threat, 'high'), (enemies_threat, 'max')]], .6),
    MyFuzzyRule([(health, 'medium'), (enemies_around, 'none')], .3),
    MyFuzzyRule([(health, 'medium'), (enemies_around, 'few'), [(enemies_threat, 'none'), (enemies_threat, 'low')]], .5),
    MyFuzzyRule([(health, 'medium'), (enemies_around, 'few'), [(enemies_threat, 'high'), (enemies_threat, 'max')]], .66),
    MyFuzzyRule([(health, 'medium'), (enemies_around, 'many'), [(enemies_threat, 'none'), (enemies_threat, 'low')]], .6),
    MyFuzzyRule([(health, 'medium'), (enemies_around, 'many'), [(enemies_threat, 'high'), (enemies_threat, 'max')]], .75),
    MyFuzzyRule([(health, 'high'), (enemies_around, 'none')], .65),
    MyFuzzyRule([(health, 'high'), (enemies_around, 'few'), [(enemies_threat, 'none'), (enemies_threat, 'low')]], .7),
    MyFuzzyRule([(health, 'high'), (enemies_around, 'few'), [(enemies_threat, 'high'), (enemies_threat, 'max')]], .8),
    MyFuzzyRule([(health, 'high'), (enemies_around, 'many'), [(enemies_threat, 'none'), (enemies_threat, 'low')]], .75),
    MyFuzzyRule([(health, 'high'), (enemies_around, 'many'), [(enemies_threat, 'high'), (enemies_threat, 'max')]], .9),
]

def create_fuzzy_model():
    sim = MyFuzzySimulation()
    sim.add_variable(health)
    sim.add_variable(enemies_around)
    sim.add_variable(enemies_threat)
    for rule in rules:
        sim.add_rule(rule)

    return sim

def get_alpha(sim, state, payload, memory):
    max_hp = payload['my_tank_status']['_max_hp']
    max_shield = payload['my_tank_status']['_max_shield']
    health_shield = (state[0] * max_hp + state[1] * max_shield) / (max_hp + max_shield)

    enemies_count = memory.get_neighbour_tanks(payload['my_tank_status']['position']['x'],
                                               payload['my_tank_status']['position']['y']) / 5

    enemies_power = 0.33 * (1-state[5]) + 0.33 * (1-state[6]) + 0.33 * state[7]
    sim.input(health, health_shield)
    sim.input(enemies_around, enemies_count)
    sim.input(enemies_threat, enemies_power)
    alpha = sim.compute()
    return alpha
