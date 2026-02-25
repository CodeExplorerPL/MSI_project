import bisect
import copy
from agent3_agent_state import calculate_distance
from agent3_field_state import check_decay, MEMORY_DECAY_TICKS


class AStarAgent:
    def __init__(self, start_position, goal_position, alpha, memory, current_tick, my_team, my_id):
        self.my_id = my_id
        self.my_team = my_team
        self.current_tick = current_tick
        self.memory = memory
        self.start = start_position
        self.goal = goal_position
        self.alpha = alpha
        self.beta = 1 - alpha
        self.max_route_length = calculate_distance(self.start, self.goal) * 2
        self.current_route = None

    def _is_goal_state(self, route: list):
        return calculate_distance(route[-1], self.goal) < 4

    def get_possible_moves(self, route: list, visited: set):
        last_position = route[-1]
        possible_positions = [(last_position[0]-2, last_position[1]), (last_position[0], last_position[1]-2),
                              (last_position[0]+2, last_position[1]), (last_position[0], last_position[1]+2),
                              (last_position[0]-2, last_position[1]-2), (last_position[0]+2, last_position[1]-2),
                              (last_position[0]+1, last_position[1]+2), (last_position[0]-2, last_position[1]+2)]

        possible_positions = [pos for pos in possible_positions if 0 <= pos[0] <= 200 and 0 <= pos[1] <= 200 and pos not in visited]
        if len(possible_positions) == 0:
            return []
        if len(route) == self.max_route_length:
            return []
        possible_moves = []
        for pos in possible_positions:
            possible_moves.append((pos, self.calc_cost(last_position, pos)))

        return possible_moves

    def get_current_route(self):
        return self.current_route

    def search(self):
        frontier = [State([self.start], 0, 0 + self.heuristic([self.start]))]
        i = 1
        visited = set()
        while frontier:
            i += 1
            current_state = frontier.pop()
            if current_state.route[-1] in visited:
                continue
            self.current_route = current_state.route
            possible_moves = self.get_possible_moves(self.current_route, visited)
            if self._is_goal_state(self.current_route):
                return self.current_route, current_state.cost
            for move, dist in possible_moves:
                route_copy = copy.deepcopy(self.current_route)
                route_copy.append(move)
                bisect.insort(frontier, State(route_copy, current_state.cost + dist,
                                              current_state.cost + dist + self.heuristic(route_copy)))
            visited.add(current_state.route[-1])
        return None, None

    def calc_cost(self, from_position, to_position):
        check_position = ((from_position[0] + to_position[0])/2, (from_position[1] + to_position[1])/2)

        terrain_info = self.memory.get_terrain_info(check_position[0], check_position[1])
        if terrain_info is None:
            time_factor = 0.5
        elif terrain_info.is_obstacle and not terrain_info.is_destructible:
            time_factor = 1e-4
        elif terrain_info.is_obstacle and terrain_info.is_destructible:
            time_factor = 1
        else:
            time_factor = terrain_info.speed_modifier

        tank_info = self.memory.get_tank_info(check_position[0], check_position[1])
        if tank_info is not None and check_decay(self.current_tick, tank_info.last_seen):
            coef = (1 - ((self.current_tick - tank_info.last_seen) / MEMORY_DECAY_TICKS))
            time_factor -= coef
        if time_factor <= 0:
            time_factor = 1e-4
        time_factor = 1 / time_factor

        allies_risk = 0
        enemies_risk = 0
        for dx in range(-10, 11, 5):
            for dy in range(-10, 11, 5):
                tank_info = self.memory.get_tank_info(check_position[0]+dx, check_position[1]+dy)
                if tank_info is not None and check_decay(self.current_tick, tank_info.last_seen):
                    coef = (1 - ((self.current_tick - tank_info.last_seen) / MEMORY_DECAY_TICKS))
                    if tank_info.team == self.my_team and tank_info.tank_id != self.my_id:
                        allies_risk += coef
                    elif tank_info.team != self.my_team:
                        enemies_risk += coef

        risk_factor = enemies_risk - allies_risk
        risk_factor = risk_factor if risk_factor > 0 else 0
        terrain_info = self.memory.get_terrain_info(check_position[0], check_position[1])
        if terrain_info is None:
            risk_factor += 0.5
        elif terrain_info.is_obstacle and not terrain_info.is_destructible:
            risk_factor += 10000
        elif terrain_info.is_obstacle and terrain_info.is_destructible:
            risk_factor += 0.5
        elif terrain_info.deal_dmg:
            time_factor += 0.5

        return self.alpha * time_factor + self.beta * risk_factor

    def heuristic(self, state):
        dist = calculate_distance(state[-1], self.goal)
        time_factor = 2/3
        heuristic_time = dist * time_factor

        heuristic_risk = 0.5 * (dist // 2)

        return self.alpha * heuristic_time + self.beta * heuristic_risk


class State:
    def __init__(self, route: list, cost: float, estimated: float):
        self.route = route
        self.cost = cost
        self.estimated = estimated

    def __lt__(self, other):
        return self.estimated > other.estimated

    def __repr__(self):
        return f'({self.route}, {self.cost}, {self.estimated})'


def get_path(current_position, destinations_scored, alpha, memory, current_tick, my_team, my_id):
    sorted_destinations_scored = {k: v for k, v in sorted(destinations_scored.items(), key=lambda item: item[1])}
    for destination, destination_score in reversed(list(sorted_destinations_scored.items())):
        terrain_info = memory.get_terrain_info(current_position[0], current_position[1])
        route, cost = make_search(current_position, destination, alpha, memory, current_tick, my_team, my_id)
        if route is not None and (cost < 1000 or (terrain_info is not None and terrain_info.is_obstacle)):
            return copy.deepcopy(route)

    return None

def make_search(start_position, goal_position, alpha, memory, current_tick, my_team, my_id):
    start_position = (round(start_position[0] / 2) * 2, round(start_position[1] / 2) * 2)
    goal_position = (round(goal_position[0] / 2) * 2, round(goal_position[1] / 2) * 2)
    mean_agent = AStarAgent(start_position, goal_position, alpha, memory, current_tick, my_team, my_id)
    res = mean_agent.search()
    return res
