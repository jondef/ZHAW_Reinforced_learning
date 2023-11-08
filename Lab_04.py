import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax


class CartPoleREINFORCEAgent:

    def __init__(self, num_episodes=500, learning_rate=0.01, gamma=0.99):
        self.num_episodes = num_episodes
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        # Initialize policy parameters theta randomly
        self.theta = np.random.rand(self.state_size, self.action_size)

    def policy(self, state):
        z = state.dot(self.theta)
        return softmax(z)

    def choose_action(self, state):
        prob = self.policy(state)
        return np.random.choice(self.action_size, p=prob)

    def train(self):
        episode_rewards = []

        for episode in range(self.num_episodes):
            state = self.env.reset()[0]
            done = False
            transitions = []

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                transitions.append((state, action, reward))
                state = next_state

            episode_rewards.append(sum(x[2] for x in transitions))
            self.learn(transitions)

            if (episode + 1) % 10 == 0:
                print(f'Episode: {episode + 1}/{self.num_episodes}, Total reward: {episode_rewards[-1]}')

        print('Finished training!')
        return episode_rewards

    def learn(self, transitions):
        for i, (state, action, reward) in enumerate(transitions):
            # Calculate the discounted return
            # G_t = R_(t+1) + gamma * R_(t+2) + gamma^2 * R_(t+3) + ... + gamma^(T-t-1) * R_T
            # R_(t+1) is the reward received after taking an action at time step t.
            # gamma is the discount factor, which is a number between 0 and 1, which determines the present value of future rewards
            # T is the final time step of the episode.
            G = sum(self.gamma ** t * r for t, (_, _, r) in enumerate(transitions[i:]))

            # Calculate the policy gradient
            probs = self.policy(state)  # probability of taking action a in state s with current policy
            dsoftmax = softmax(state.dot(self.theta)) * (1 - softmax(state.dot(self.theta)))
            dlog = dsoftmax / probs  # derivative of log pi(a|s, theta)
            grad = state.reshape(-1, 1) * dlog  # gradient of log pi(a|s, theta) * G

            # Update policy parameters
            self.theta[:, action] += self.learning_rate * grad[:, action] * G

    def plot_learning(self, episode_rewards):
        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.show()


def load_reinforce():
    num_episodes = 500
    learning_rate = 0.01
    gamma = 0.99

    agent = CartPoleREINFORCEAgent(num_episodes, learning_rate, gamma)
    episode_rewards = agent.train()
    agent.plot_learning(episode_rewards)
    return agent


agent = load_reinforce()
