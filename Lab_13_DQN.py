import os
import random
from collections import deque

import numpy as np
from tensorflow import gather_nd
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import RMSprop
import tensorflow as tf

class DQN:

    def __init__(self, policy: str, env, gamma, epsilon, min_epsilon=0.01):
        """
        :param env: stable baseline env
        :param gamma: discount factor
        :param epsilon: for epsilon-greedy policy
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.model_name = "lab_13_trained_model_temp"

        self.stateDimension = 8
        self.actionDimension = 4

        self.replayBuffer = deque(maxlen=1024)  # max size of the replay buffer
        self.trainingBatchSize = 256

        self.onlineNetwork = self.createNetwork()
        self.targetNetwork = self.createNetwork()
        self.targetNetwork.set_weights(self.onlineNetwork.get_weights())

        # update targetNetwork after ... episodes
        self.updateTargetNetworkPeriod = 100  # needs to be a multiple of num_envs

        # env step counter
        self.timestep_count = 0

        # this list is used in the cost function to select certain entries of the
        # predicted and true sample matrices in order to form the loss
        self.actionsAppend = []

        # this sum is used to store the sum of rewards obtained during each training episode
        self.sumRewardsEpisode = []

    def save(self):
        self.onlineNetwork.save(self.model_name)

    def load(self):
        if os.path.exists(self.model_name):
            tf.keras.models.load_model(self.model_name,
                                       custom_objects={'dqn_loss': self.dqn_loss} )

    def dqn_loss(self, y_true, y_pred):
        """
        :param y_true: matrix of dimension (self.batchReplayBufferSize,2) - this is the target
        :param y_pred: matrix of dimension (self.batchReplayBufferSize,2) - this is predicted by the network
        :return: - loss -
        """
        loss = tf.reduce_mean(tf.square(y_true - y_pred))
        return loss

    def createNetwork(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.stateDimension, activation='relu'))
        model.add(Dense(56, activation='relu'))
        model.add(Dense(self.actionDimension, activation='linear'))
        # compile the network with the custom loss defined in my_loss_fn
        model.compile(optimizer=RMSprop(), loss=self.dqn_loss)
        return model

    def learn(self, total_timesteps):
        num_envs = len(self.env.reset())  # Assuming this returns a list of initial states for each env
        self.timestep_count = 0  # Initialize timestep counter
        episodeIndex = 0

        # Episode loop
        while self.timestep_count < total_timesteps:
            currentState: list = self.env.reset()  # currentState for all envs
            episodeDone = [False] * num_envs  # Initialize terminal states for all environments
            episodeReward = []  # good for keeping track of convergence

            while not all(episodeDone) and self.timestep_count < total_timesteps: # exit once all envs are terminated
                print(f"Episode: {episodeIndex}\ttimesteps: {self.timestep_count}\tepisodeDone: {episodeDone}")

                if np.random.random() < self.epsilon:
                    actions = np.array([np.random.choice(self.actionDimension) for _ in range(num_envs)])
                else:  # greedy action
                    actions, _ = self.predict(currentState)

                nextState, reward, terminalState, _ = self.env.step(actions)  # This is for all envs
                self.timestep_count += num_envs  # Increment timestep count by the number of environments

                # add transitions from all envs to the replay buffer only if the env is not terminated
                for i in range(num_envs):
                    if not episodeDone[i]:  # Process only if the episode for this env isn't done
                        self.replayBuffer.append((currentState[i], actions[i], reward[i], nextState[i], terminalState[i]))
                        episodeReward.append(reward[i])
                        if terminalState[i]:  # Mark episode as done for this env
                            episodeDone[i] = True

                # train network only if reply buffer has at least batchReplayBufferSize elements
                if len(self.replayBuffer) > self.trainingBatchSize:
                    self.trainNetwork()

                currentState = nextState

            # epsilon decay after each episode
            self.epsilon = max(self.epsilon * 0.999, self.min_epsilon)

            total_reward = np.sum(episodeReward)
            print(f"Sum of rewards {total_reward}")
            self.sumRewardsEpisode.append(total_reward)
            episodeIndex += 1

    def predict(self, observations: list[list], state=None, episode_start=None, deterministic=False):
        """
        This function selects an action based on the current state
        :param observations: current state for which to compute the action (list of lists for vecenvs)
        :param state: (optional) used for stateful models like RNNs
        :param episode_start: (optional) boolean indicating the start of an episode
        :param deterministic: (optional) whether to use a deterministic policy
        :return: action
        """
        Qvalues = self.onlineNetwork.predict(observations, verbose=0)

        actions = np.zeros(len(observations), dtype=np.int32)

        for i in range(len(observations)):
            # If deterministic, choose the action with the highest Q-value, first index
            if deterministic:
                actions[i] = np.argmax(Qvalues[i, :])
            else:
                # If not deterministic, randomly choose action with max Q-values
                actions[i] = np.random.choice(np.where(Qvalues[i, :] == np.max(Qvalues[i, :]))[0])

        return actions, state

    def trainNetwork(self):
        # sample a batch from the replay buffer
        randomSampleBatch = random.sample(self.replayBuffer, self.trainingBatchSize)

        # Extract components from the sample batch using vectorized operations
        currentStateBatch, actionBatch, rewardBatch, nextStateBatch, terminatedBatch = map(np.array, zip(*randomSampleBatch))

        # predict Q-values using networks
        QcurrentStateOnlineNetwork = self.onlineNetwork.predict(currentStateBatch, verbose=0)
        QnextStateTargetNetwork = self.targetNetwork.predict(nextStateBatch, verbose=0)

        # Compute target Q-value (y)
        maxQnextState = np.max(QnextStateTargetNetwork, axis=1)  # get highest action value for each sample in the batch
        y = rewardBatch + self.gamma * maxQnextState * (~terminatedBatch)

        # Only update the action that were taken (Q-value for other actions stays the same)
        QcurrentStateOnlineNetwork[np.arange(self.trainingBatchSize), actionBatch] = y

        # Train the online network
        self.actionsAppend = actionBatch  # for cost function
        self.onlineNetwork.fit(x=currentStateBatch, y=QcurrentStateOnlineNetwork, batch_size=self.trainingBatchSize, verbose=0, epochs=1)

        # after n steps, update the weights of the target network
        if self.timestep_count % self.updateTargetNetworkPeriod == 0:
            self.targetNetwork.set_weights(self.onlineNetwork.get_weights())
            print("Target network updated!")
