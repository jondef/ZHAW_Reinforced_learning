from collections import defaultdict, namedtuple
from enum import Enum
from typing import Tuple, List
import random
from IPython.display import clear_output
import time

Point = namedtuple('Point', ['x', 'y'])

"""
Modify the agent to hold state values instead of state-action values.
Use ε-Greedy to select actions based on state values (or a proxy using action values if you don't have a transition model).
At the end of each episode, calculate returns for states and update the state values.
"""

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


class MonteCarloGeneration(object):
    """
    This class `MonteCarloGeneration` represents a Monte Carlo method for generating samples from an environment.

    Attributes:
    - `env`: The environment object
    - `max_steps`: The maximum number of steps to run before terminating
    - `debug`: Boolean flag indicating whether to print debug information

    Methods:
    - `run() -> List`: Runs the Monte Carlo generation process and returns a list of samples.

    Example usage:
    ```python
    env = Environment()
    generator = MonteCarloGeneration(env, max_steps=1000, debug=True)
    samples = generator.run()
    ```
    ```"""

    def __init__(self, env: SimpleGridWorld, agent=None, epsilon=0.1, max_steps: int = 1000, debug: bool = False):
        self.env = env
        self.agent = agent  # The agent object that interacts with the environment
        self.epsilon = epsilon  # for epsilon-greedy policy
        self.max_steps = max_steps
        self.debug = debug

    def set_agent(self, agent):
        self.agent = agent

    def run(self) -> List:
        buffer = []
        n_steps = 0
        state, _, _ = self.env.reset()
        terminal = False
        while not terminal:
            if random.random() < self.epsilon:  # exploration / exploitation tradeoff, sometimes explore, sometimes exploit
                action = random.choice(self.env.action_space)
            else:  # here is exploitation
                # Instead of random, choose best action based on next state's value
                values = [self.agent.action_value(self.env.next_state(state, a), a) for a in self.env.action_space]
                best_action_idx = argmax(values)
                action = self.env.action_space[best_action_idx]

            next_state, reward, terminal = self.env.step(action)
            buffer.append((state, action, reward))
            state = next_state
            n_steps += 1
            if n_steps >= self.max_steps:
                if self.debug:
                    print("Terminated early due to large number of steps")
                terminal = True
        return buffer


class MonteCarloExperiment(object):
    """
    This class represents a Monte Carlo Experiment.

    Attributes:
        generator (MonteCarloGeneration): A MonteCarloGeneration object used to generate trajectories.
        values (defaultdict): A dictionary that stores the total value of each state-action pair.
        counts (defaultdict): A dictionary that stores the number of times each state-action pair has been encountered.

    Methods:
        _to_key(state, action): Converts a state-action pair into a key.
        action_value(state, action): Calculates the value of a state-action pair.
        run_episode(): Runs a single episode of the experiment.
    """

    def __init__(self, generator: MonteCarloGeneration):
        self.generator = generator
        self.values = defaultdict(float)
        self.counts = defaultdict(float)

    def _to_key(self, state, action):
        return (state, action)

    def action_value(self, state, action) -> float:
        """
        :param state: The current state of the environment.
        :param action: The action to be taken in the current state.
        :return: The estimated action value for the given state-action pair.

        This method calculates the estimated action value for a given state-action pair using the Monte Carlo method. It checks if the state-action pair has been visited before and if so, returns the average value calculated so far. If the state-action pair has not been visited before, it returns 0.0.

        """
        key = self._to_key(state, action)
        if self.counts[key] > 0:
            return self.values[key] / self.counts[key]
        else:
            return 0.0

    def run_episode(self) -> None:
        trajectory = self.generator.run()  # Generate a trajectory
        episode_reward = 0
        """
        accumulate rewards from the current step until the end of the episode for
        each state-action pair, and then add that accumulated reward.
        """
        for i, t in enumerate(reversed(trajectory)):  # Starting from the terminal state
            state, action, reward = t
            key = self._to_key(state, action)
            episode_reward += reward  # Add the reward to the buffer
            self.values[key] += episode_reward  # And add this to the value of this action
            self.counts[key] += 1  # Increment counter


def state_value_2d(env: SimpleGridWorld, agent: MonteCarloExperiment):
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



epsilon = 1  # How often to explore (take a random action)
env = SimpleGridWorld()  # Instantiate the environment

# Instantiate the trajectory generator with the environment and epsilon (without the agent for now)
generator = MonteCarloGeneration(env=env, epsilon=epsilon)

# Instantiate the agent with the generator
agent = MonteCarloExperiment(generator=generator)

# Now, set the agent in the generator
generator.set_agent(agent)


for i in range(4000):
    clear_output(wait=True)
    agent.run_episode()
    print(f"Iteration: {i}")
    print(state_value_2d(env, agent))
    print(next_best_value_2d(env, agent), flush=True)
    #time.sleep(0.1)  # Uncomment this line if you want to see the iterations


