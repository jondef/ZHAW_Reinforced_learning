import random
from collections import defaultdict

from IPython.display import clear_output

from Lab_03_SimpleGridWorld import SimpleGridWorld, state_value_2d, next_best_value_2d, argmax

"""
The Q-Learning update rule is:
Q(s,a) ← Q(s,a) + α( reward+γmax_a′Q(s′,a′) − Q(s,a) )

Where:
- Q(s,a) is the current Q-value of state s and action a
- α is the learning rate
- γ is the discount factor
- s′ is the next state
- max_a′Q(s′, a′) is the maximum Q-value for the next state across all possible actions
"""


class QLearningAgent(object):
    """
    This class holds the Q-Values for each state-action pair and implements the Q-Learning update rule.
    """

    def __init__(self, alpha=0.1, gamma=0.99, action_space=None):
        self.alpha = alpha
        self.gamma = gamma
        self.Q_Values = defaultdict(float)
        self.action_space = action_space

    def action_value(self, state, action):
        return self.Q_Values[(state, action)]

    def best_action(self, state):
        # Return the action with the maximum Q-value for the state
        values = [self.action_value(state, action) for action in self.action_space]
        best_idx = argmax(values)
        return self.action_space[best_idx]

    def learn(self, state, action, reward, next_state):
        # Update the Q-value using the Q-learning update rule above
        max_next_value = max([self.action_value(next_state, next_action) for next_action in self.action_space])
        self.Q_Values[(state, action)] += self.alpha * (reward + self.gamma * max_next_value - self.action_value(state, action))


class QLearningGeneration(object):
    def __init__(self, env, agent, epsilon=0.1, epsilon_decay=0.01, epsilon_min=0.99, max_steps=1000, debug=False):
        self.env = env
        self.agent = agent
        self.epsilon = epsilon
        self.epsilon_min = epsilon_decay
        self.epsilon_decay = epsilon_min
        self.max_steps = max_steps
        self.debug = debug

    def run(self):
        state, _, _ = self.env.reset()
        n_steps = 0
        terminal = False
        while not terminal:
            # Epsilon-greedy policy
            if random.random() < self.epsilon:
                action = random.choice(self.env.action_space)
            else:
                action = self.agent.best_action(state)

            # Take a step in the environment
            next_state, reward, terminal = self.env.step(action)

            # Update Q-values
            self.agent.learn(state, action, reward, next_state)

            # Update current state
            state = next_state
            n_steps += 1
            if n_steps >= self.max_steps:
                if self.debug:
                    print("Terminated early due to large number of steps")
                terminal = True

            # Epsilon decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


epsilon = 1  # How often to explore (take a random action)
epsilon_decay = 0.9  # How much to reduce exploration
epsilon_min = 0.1  # Minimum exploration rate
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor

env = SimpleGridWorld()  # Instantiate the environment
q_agent = QLearningAgent(action_space=env.action_space, alpha=alpha, gamma=gamma)
generator = QLearningGeneration(env, q_agent, epsilon, epsilon_decay, epsilon_min)

for i in range(1000):
    clear_output(wait=True)
    generator.run()
    print(f"Iteration: {i}")
    print(state_value_2d(env, q_agent))
    print(next_best_value_2d(env, q_agent), flush=True)
    # time.sleep(0.1)  # Uncomment this line if you want to see the iterations
