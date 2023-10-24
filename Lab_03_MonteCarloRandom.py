from collections import defaultdict, namedtuple
from enum import Enum
from typing import Tuple, List
import random
from IPython.display import clear_output
import time

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


"""s = SimpleGridWorld(debug=True)
print("☝ This shows a simple visualisation of the environment state.\n")
s.step(Direction.SOUTH)
print(s.step(Direction.SOUTH), "⬅ This displays the state and reward from the environment 𝐀𝐅𝐓𝐄𝐑 moving.\n")
s.step(Direction.SOUTH)
s.step(Direction.SOUTH)
s.step(Direction.EAST)
s.step(Direction.EAST)
s.step(Direction.EAST)
s.step(Direction.EAST)"""


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

    def __init__(self, env: SimpleGridWorld, max_steps: int = 1000, debug: bool = False):
        self.env = env
        self.max_steps = max_steps
        self.debug = debug

    def run(self) -> List:
        buffer = []
        n_steps = 0  # Keep track of the number of steps so I can bail out if it takes too long
        state, _, _ = self.env.reset()  # Reset environment back to start
        terminal = False
        while not terminal:  # Run until terminal state
            action = random.choice(self.env.action_space)  # Random action. Try replacing this with Direction.EAST
            next_state, reward, terminal = self.env.step(action)  # Take action in environment
            buffer.append((state, action, reward))  # Store the result
            state = next_state  # Ready for the next step
            n_steps += 1
            if n_steps >= self.max_steps:
                if self.debug:
                    print("Terminated early due to large number of steps")
                terminal = True  # Bail out if we've been working for too long
        return buffer


"""env = SimpleGridWorld(debug=True)  # Instantiate the environment
generator = MonteCarloGeneration(env=env, max_steps=20, debug=True)  # Instantiate the generation
trajectory = generator.run()  # Generate a trajectory
print([t[1].value for t in trajectory])  # Print chosen actions
print(f"total reward: {sum([t[2] for t in trajectory])}")  # Print final reward"""


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


"""env = SimpleGridWorld(debug=False)  # Instantiate the environment - set the debug to true to see the actual movement of the agent.
generator = MonteCarloGeneration(env=env, debug=True)  # Instantiate the trajectory generator
agent = MonteCarloExperiment(generator=generator)
for i in range(4):
    agent.run_episode()
    # print the value of each action for the state (3,0)
    #print(f"Run {i}: ", [agent.action_value(Point(3, 0), d) for d in env.action_space])
print(state_value_2d(env, agent))"""

"""env = SimpleGridWorld()  # Instantiate the environment
generator = MonteCarloGeneration(env=env)  # Instantiate the trajectory generator
agent = MonteCarloExperiment(generator=generator)
for i in range(1000):
    clear_output(wait=True)
    agent.run_episode()
    print(f"Iteration: {i}")
    print([agent.action_value(Point(0,4), d) for d in env.action_space]) # Uncomment this line to see the actual values for a particular state
    print(state_value_2d(env, agent), flush=True)
    #time.sleep(0.1) # Uncomment this line if you want to see every episode
print(next_best_value_2d(env, agent))"""


env = SimpleGridWorld()  # Instantiate the environment
generator = MonteCarloGeneration(env=env)  # Instantiate the trajectory generator
agent = MonteCarloExperiment(generator=generator)
for i in range(4000):
    clear_output(wait=True)
    agent.run_episode()
    print(f"Iteration: {i}")
    print(state_value_2d(env, agent))
    print(next_best_value_2d(env, agent), flush=True)
    #time.sleep(0.1)  # Uncomment this line if you want to see the iterations


