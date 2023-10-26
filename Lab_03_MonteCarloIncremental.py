import random
from collections import defaultdict
from typing import List

from IPython.display import clear_output

from Lab_03_SimpleGridWorld import SimpleGridWorld, state_value_2d, next_best_value_2d, argmax


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

    def best_action(self, state):
        values = [self.agent.action_value(self.env.next_state(state, a), a) for a in self.env.action_space]
        best_action_idx = argmax(values)
        return env.action_space[best_action_idx]

    def run(self) -> List:
        """
        To implement the Incremental Monte Carlo:
        Start your episode and take an action.
        After each step, calculate the estimated return from the current state.
        Update the value of the state based on this estimated return without waiting for the episode to end.
        Use the updated state values to guide action selection in the ongoing episode.
        Repeat until the episode ends.

        After each step within an episode, estimate the return for the current state and update its value.
        Use the updated values to guide action selection within the ongoing episode.
        """
        print("Running episode...")
        buffer = []
        n_steps = 0
        episode_reward = 0
        state, _, _ = self.env.reset()
        terminal = False
        while not terminal:
            if random.random() < self.epsilon:  # exploration
                action = random.choice(self.env.action_space)
            else:  # exploitation
                action = self.best_action(state)

            # Take a step in the environment
            next_state, reward, terminal = self.env.step(action)
            buffer.append((state, action, reward))

            # ???????????
            key = self.agent._to_key(state, action)
            episode_reward += reward  # Add the reward to the buffer
            self.agent.values[key] += episode_reward  # And add this to the value of this action
            self.agent.counts[key] += 1  # Increment counter

            state = next_state
            n_steps += 1
            if n_steps >= self.max_steps:
                if self.debug:
                    print("Terminated early due to large number of steps")
                terminal = True
        return buffer


class MonteCarloIncrementalAgent(object):
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

    def __init__(self, generator: MonteCarloGeneration, gamma: float = 0.9):
        self.generator = generator
        self.values = defaultdict(float)
        self.counts = defaultdict(float)
        self.gamma = gamma

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
        trajectory = self.generator.run()  # Generate a trajectory using Monte Carlo method


epsilon = 1  # How often to explore (take a random action)
gamma = 0.9  # Discount factor
env = SimpleGridWorld()  # Instantiate the environment

# Instantiate the trajectory generator with the environment and epsilon (without the agent for now)
generator = MonteCarloGeneration(env=env, epsilon=epsilon)

# Instantiate the agent with the generator
agent = MonteCarloIncrementalAgent(generator=generator, gamma=gamma)

# Now, set the agent in the generator
generator.set_agent(agent)

for i in range(4000):
    clear_output(wait=True)
    agent.run_episode()
    print(f"Iteration: {i}")
    print(state_value_2d(env, agent))
    print(next_best_value_2d(env, agent), flush=True)
    # time.sleep(0.1)  # Uncomment this line if you want to see the iterations
