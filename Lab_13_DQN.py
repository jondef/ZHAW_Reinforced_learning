import random
from collections import deque

import numpy as np
from tensorflow import gather_nd
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import RMSprop


class DQN:

    def __init__(self, policy: str, env, gamma, epsilon):
        """
        :param env: stable baseline env
        :param gamma: discount factor
        :param epsilon: for epsilon-greedy policy
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

        self.stateDimension = 8
        self.actionDimension = 4

        self.replayBufferMaxSize = 300
        self.batchReplayBufferSize = 100
        self.replayBuffer = deque(maxlen=self.replayBufferMaxSize)

        self.onlineNetwork = self.createNetwork()
        self.targetNetwork = self.createNetwork()
        self.targetNetwork.set_weights(self.onlineNetwork.get_weights())

        # update targetNetwork after ... episodes
        self.updateTargetNetworkPeriod = 100
        self.updateTargetNetworkCounter = 0

        # this list is used in the cost function to select certain entries of the
        # predicted and true sample matrices in order to form the loss
        self.actionsAppend = []

        # this sum is used to store the sum of rewards obtained during each training episode
        self.sumRewardsEpisode = []

    def my_loss_fn(self, y_true, y_pred):
        """
        This function will select certain row entries from y_true and y_pred to form the output
        the selection is performed on the basis of the action indices in the list  self.actionsAppend
        this function is used in createNetwork(self) to create the network

        :param y_true: matrix of dimension (self.batchReplayBufferSize,2) - this is the target
        :param y_pred: matrix of dimension (self.batchReplayBufferSize,2) - this is predicted by the network
        :return: - loss - watch out here, this is a vector of (self.batchReplayBufferSize,1),
                   with each entry being the squared error between the entries of y_true and y_pred
                   later on, the tensor flow will compute the scalar out of this vector (mean squared error)
        """
        s1, s2 = y_true.shape
        print(f"y_true shape: {y_true.shape}")

        # this matrix defines indices of a set of entries that we want to
        # extract from y_true and y_pred
        # s2=2
        # s1=self.batchReplayBufferSize
        indices = np.zeros(shape=(s1, 2))
        indices[:, 0] = np.arange(s1)
        indices[:, 1] = self.actionsAppend

        y_true_gathered = gather_nd(y_true, indices=indices.astype(int))
        y_pred_gathered = gather_nd(y_pred, indices=indices.astype(int))

        loss = mean_squared_error(y_true_gathered, y_pred_gathered)
        return loss

    def createNetwork(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.stateDimension, activation='relu'))
        model.add(Dense(56, activation='relu'))
        model.add(Dense(self.actionDimension, activation='linear'))
        # compile the network with the custom loss defined in my_loss_fn
        model.compile(optimizer=RMSprop(), loss=self.my_loss_fn, metrics=['accuracy'])
        return model

    def learn(self, total_timesteps):

        for episodeIndex in range(total_timesteps):

            # list that stores rewards per episode - this is necessary for keeping track of convergence
            episodeReward = []

            print("Simulating episode {}".format(episodeIndex))

            currentState: list = self.env.reset()  # currentState for all envs

            terminalState = False
            while not terminalState:

                if episodeIndex < 1 or np.random.random() < self.epsilon:
                    actions = np.array([np.random.choice(self.actionDimension) for _ in range(len(currentState))])  # len(currentState) is the number of envs
                else:  # greedy action
                    actions, _ = self.predict(currentState)

                if episodeIndex > 200:
                    self.epsilon = 0.999 * self.epsilon

                (nextState, reward, terminalState, _) = self.env.step(actions)  # This is for all envs
                terminalState = terminalState[0]
                reward = reward[0]

                episodeReward.append(reward)
                # todo: add here all transitions for all envs???
                self.replayBuffer.append((currentState[0], actions[0], reward, nextState[0], terminalState))

                # train network only if reply buffer has at least batchReplayBufferSize elements
                if len(self.replayBuffer) > self.batchReplayBufferSize:
                    self.trainNetwork()

                currentState = nextState

            print("Sum of rewards {}".format(np.sum(episodeReward)))
            self.sumRewardsEpisode.append(np.sum(episodeReward))

    def predict(self, observations: list[list], state=None, episode_start=None, deterministic=False):
        """
        This function selects an action based on the current state
        :param observations: current state for which to compute the action (list of lists for vecenvs)
        :param state: (optional) used for stateful models like RNNs
        :param episode_start: (optional) boolean indicating the start of an episode
        :param deterministic: (optional) whether to use a deterministic policy
        :return: action
        """
        Qvalues = self.onlineNetwork.predict(np.array(observations), verbose=0)

        actions = np.zeros(len(observations), dtype=np.int32)

        for i in range(len(observations)):
            # If deterministic, choose the action with the highest Q-value, first index
            if deterministic:
                actions[i] = np.argmax(Qvalues[0, :])
            else:
                # If not deterministic, handle the possibility of multiple max Q-values
                actions[i] = np.random.choice(np.where(Qvalues[0, :] == np.max(Qvalues[0, :]))[0])

        return actions, state

    def trainNetwork(self):
        # sample a batch from the replay buffer
        randomSampleBatch = random.sample(self.replayBuffer, self.batchReplayBufferSize)

        # here we form current state batch
        # and next state batch
        # they are used as inputs for prediction
        currentStateBatch = np.zeros(shape=(self.batchReplayBufferSize, self.stateDimension))
        nextStateBatch = np.zeros(shape=(self.batchReplayBufferSize, self.stateDimension))
        # this will enumerate the tuple entries of the randomSampleBatch
        # index will loop through the number of tuples
        for index, tupleS in enumerate(randomSampleBatch):
            # first entry of the tuple is the current state
            currentStateBatch[index, :] = tupleS[0]
            # fourth entry of the tuple is the next state
            nextStateBatch[index, :] = tupleS[3]

        # here, use the target network to predict Q-values
        QnextStateTargetNetwork = self.targetNetwork.predict(nextStateBatch)
        # here, use the main network to predict Q-values
        QcurrentStateMainNetwork = self.onlineNetwork.predict(currentStateBatch)

        # now, we form batches for training
        # input for training
        inputNetwork = currentStateBatch
        # output for training
        outputNetwork = np.zeros(shape=(self.batchReplayBufferSize, self.actionDimension))

        # this list will contain the actions that are selected from the batch
        # this list is used in my_loss_fn to define the loss-function
        self.actionsAppend = []
        for index, (currentState, action, reward, nextState, terminated) in enumerate(randomSampleBatch):

            if terminated:  # if the next state is the terminal state
                y = reward
            else:
                y = reward + self.gamma * np.max(QnextStateTargetNetwork[index])

            # this is necessary for defining the cost function
            self.actionsAppend.append(action)

            # this actually does not matter since we do not use all the entries in the cost function
            outputNetwork[index] = QcurrentStateMainNetwork[index]
            # this is what matters
            outputNetwork[index, action] = y

        # here, we train the network
        self.onlineNetwork.fit(inputNetwork, outputNetwork, batch_size=self.batchReplayBufferSize, verbose=0, epochs=100)

        # after updateTargetNetworkPeriod training sessions, update the coefficients
        # of the target network
        # increase the counter for training the target network
        self.updateTargetNetworkCounter += 1
        if (self.updateTargetNetworkCounter > (self.updateTargetNetworkPeriod - 1)):
            # copy the weights to targetNetwork
            self.targetNetwork.set_weights(self.onlineNetwork.get_weights())
            print("Target network updated!")
            print("Counter value {}".format(self.updateTargetNetworkCounter))
            # reset the counter
            self.updateTargetNetworkCounter = 0
