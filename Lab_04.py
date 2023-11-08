import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax


class CartPoleREINFORCEAgent:
    """
    REINFORCE Agent for the CartPole environment
    """
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
                next_state = np.array(next_state)  # Ensure next_state is a NumPy array
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
            G = sum(self.gamma ** t * r for t, (_, _, r) in enumerate(transitions[i:]))

            # Calculate the policy gradient
            probs = self.policy(state)
            dsoftmax = probs * (1 - probs[action])
            dlog = -dsoftmax / probs[action]
            grad = np.outer(dlog, state).T

            # Update policy parameters
            self.theta += self.learning_rate * grad * G

    def plot_learning(self, episode_rewards):
        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.show()


def load_reinforce():
    agent = CartPoleREINFORCEAgent()
    episode_rewards = agent.train()
    agent.plot_learning(episode_rewards)
    return agent


agent = load_reinforce()
