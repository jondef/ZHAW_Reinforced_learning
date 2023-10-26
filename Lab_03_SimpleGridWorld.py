from collections import namedtuple
from enum import Enum

Point = namedtuple('Point', ['x', 'y'])


class Direction(Enum):
    NORTH = "⬆"
    EAST = "⮕"
    SOUTH = "⬇"
    WEST = "⬅"

    @classmethod
    def values(self):
        return [v for v in self]


class SimpleGridWorld(object):
    def __init__(self, width: int = 5, height: int = 5, debug: bool = False):
        self.width = width
        self.height = height
        self.debug = debug
        self.action_space = [d for d in Direction]
        self.reset()

    def reset(self):
        self.cur_pos = Point(x=0, y=(self.height - 1))
        self.goal = Point(x=(self.width - 1), y=0)
        # If debug, print state
        if self.debug:
            print(self)
        return self.cur_pos, 0, False

    def step(self, action: Direction):
        # Depending on the action, mutate the environment state
        if action == Direction.NORTH:
            self.cur_pos = Point(self.cur_pos.x, self.cur_pos.y + 1)
        elif action == Direction.EAST:
            self.cur_pos = Point(self.cur_pos.x + 1, self.cur_pos.y)
        elif action == Direction.SOUTH:
            self.cur_pos = Point(self.cur_pos.x, self.cur_pos.y - 1)
        elif action == Direction.WEST:
            self.cur_pos = Point(self.cur_pos.x - 1, self.cur_pos.y)
        # Check if out of bounds
        if self.cur_pos.x >= self.width:
            self.cur_pos = Point(self.width - 1, self.cur_pos.y)
        if self.cur_pos.y >= self.height:
            self.cur_pos = Point(self.cur_pos.x, self.height - 1)
        if self.cur_pos.x < 0:
            self.cur_pos = Point(0, self.cur_pos.y)
        if self.cur_pos.y < 0:
            self.cur_pos = Point(self.cur_pos.x, 0)

        # If at goal, terminate
        is_terminal = self.cur_pos == self.goal

        # Constant -1 reward to promote speed-to-goal
        reward = -1

        # If debug, print state
        if self.debug:
            print(self)

        return self.cur_pos, reward, is_terminal

    def __repr__(self):
        res = ""
        for y in reversed(range(self.height)):
            for x in range(self.width):
                if self.goal.x == x and self.goal.y == y:
                    if self.cur_pos.x == x and self.cur_pos.y == y:
                        res += "@"
                    else:
                        res += "o"
                    continue
                if self.cur_pos.x == x and self.cur_pos.y == y:
                    res += "x"
                else:
                    res += "_"
            res += "\n"
        return res

    def next_state(self, state: Point, action: Direction) -> Point:
        """Compute the next state given a current state and action."""

        if action == Direction.NORTH:
            new_pos = Point(state.x, state.y + 1)
        elif action == Direction.EAST:
            new_pos = Point(state.x + 1, state.y)
        elif action == Direction.SOUTH:
            new_pos = Point(state.x, state.y - 1)
        elif action == Direction.WEST:
            new_pos = Point(state.x - 1, state.y)
        else:
            raise ValueError("Invalid action")

        # Check if out of bounds and adjust the position if needed
        if new_pos.x >= self.width:
            new_pos = Point(self.width - 1, new_pos.y)
        if new_pos.y >= self.height:
            new_pos = Point(new_pos.x, self.height - 1)
        if new_pos.x < 0:
            new_pos = Point(0, new_pos.y)
        if new_pos.y < 0:
            new_pos = Point(new_pos.x, 0)

        return new_pos


def state_value_2d(env: SimpleGridWorld, agent):
    """
    :param env: The environment object that represents the 2D grid world.
    :param agent: The agent object that interacts with the environment.

    :return: A string representation (2D grid) of the state values for each state in the environment. Each grid cell represents the average state value calculated based on the agent's action values.

    This method iterates through each state in the environment and calculates the average state value by summing the action values for each action in the agent's action space and dividing by the number of actions. The state values are then formatted as a string and returned as the result.
    """
    res = ""
    for y in reversed(range(env.height)):
        for x in range(env.width):
            if env.goal.x == x and env.goal.y == y:
                res += "   @  "
            else:
                state_value = sum([agent.action_value(Point(x, y), d) for d in env.action_space]) / len(env.action_space)
                res += f'{state_value:6.2f}'
            res += " | "
        res += "\n"
    return res


def argmax(a):
    return max(range(len(a)), key=lambda x: a[x])


def next_best_value_2d(env, agent):
    """

    :param env: The environment object that represents the 2D grid world.
    :param agent: The agent object that interacts with the environment.

    :return: A string representation of the 2D grid world with the next best action values for each state.

    """
    res = ""
    for y in reversed(range(env.height)):
        for x in range(env.width):
            if env.goal.x == x and env.goal.y == y:
                res += "@"
            else:
                # Find the action that has the highest value
                loc = argmax([agent.action_value(Point(x, y), d) for d in env.action_space])
                res += f'{env.action_space[loc].value}'
            res += " | "
        res += "\n"
    return res
