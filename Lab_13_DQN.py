# import the necessary libraries
import random
from collections import deque

import numpy as np
from tensorflow import gather_nd
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop


class DQN:

    def __init__(self, env, gamma, epsilon):
        """
        :param env: stable baseline env
        :param gamma: discount factor
        :param epsilon: for epsilon-greedy policy
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

        # state dimension
        self.stateDimension = 8
        # action dimension
        self.actionDimension = 4
        # this is the maximum size of the replay buffer
        self.replayBufferSize = 300
        # this is the size of the training batch that is randomly sampled from the replay buffer
        self.batchReplayBufferSize = 100

        # number of training episodes it takes to update the target network parameters
        # that is, every updateTargetNetworkPeriod we update the target network parameters
        self.updateTargetNetworkPeriod = 100

        # this is the counter for updating the target network
        # if this counter exceeds (updateTargetNetworkPeriod-1) we update the network
        # parameters and reset the counter to zero, this process is repeated until the end of the training process
        self.counterUpdateTargetNetwork = 0

        # this sum is used to store the sum of rewards obtained during each training episode
        self.sumRewardsEpisode = []

        # replay buffer
        self.replayBuffer = deque(maxlen=self.replayBufferSize)

        # this is the main network
        # create network
        self.onlineNetwork = self.createNetwork()

        # this is the target network
        # create network
        self.targetNetwork = self.createNetwork()

        # copy the initial weights to targetNetwork
        self.targetNetwork.set_weights(self.onlineNetwork.get_weights())

        # this list is used in the cost function to select certain entries of the
        # predicted and true sample matrices in order to form the loss
        self.actionsAppend = []

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
        print(f"y_pred shape: {y_pred.shape}")

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
        print(loss)
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

        # here we loop through the episodes
        for indexEpisode in range(total_timesteps):

            # list that stores rewards per episode - this is necessary for keeping track of convergence
            rewardsEpisode = []

            print("Simulating episode {}".format(indexEpisode))

            # reset the environment at the beginning of every episode
            currentState = self.env.reset()[0]  # fixme: for vectorized envs > 1 returns S of all envs

            # here we step from one state to another
            # this will loop until a terminal state is reached
            terminalState = False
            while not terminalState:

                # select an action on the basis of the current state, denoted by currentState
                # fixme: here action should be an array for vectorized envs
                action = self.predict(currentState, indexEpisode)

                # here we step and return the state, reward, and boolean denoting if the state is a terminal state
                (nextState, reward, terminalState, _) = self.env.step([action])
                rewardsEpisode.append(reward)

                # add current state, action, reward, next state, and terminal flag to the replay buffer
                self.replayBuffer.append((currentState, action, reward, nextState, terminalState))

                # train network
                self.trainNetwork()

                # set the current state for the next step
                currentState = nextState

            print("Sum of rewards {}".format(np.sum(rewardsEpisode)))
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode))

    def predict(self, state, episode_index):
        """
        This function selects an action on the basis of the current state
        :param state: state for which to compute the action
        :param episode_index:  index of the current episode
        :return:
        """

        # first index episodes we select completely random actions to have enough exploration
        # change this
        if episode_index < 1:
            return np.random.choice(self.actionDimension)

            # Returns a random real number in the half-open interval [0.0, 1.0)
        # this number is used for the epsilon greedy approach
        randomNumber = np.random.random()

        # after index episodes, we slowly start to decrease the epsilon parameter
        if episode_index > 200:
            self.epsilon = 0.999 * self.epsilon

        # if this condition is satisfied, we are exploring, that is, we select random actions
        if randomNumber < self.epsilon:
            # returns a random action selected from: 0,1,...,actionNumber-1
            return np.random.choice(self.actionDimension)

            # otherwise, we are selecting greedy actions
        else:
            # we return the index where Qvalues[state,:] has the max value
            # that is, since the index denotes an action, we select greedy actions

            Qvalues = self.onlineNetwork.predict(state.reshape(1, self.stateDimension))

            return np.random.choice(np.where(Qvalues[0, :] == np.max(Qvalues[0, :]))[0])
            # here we need to return the minimum index since it can happen
            # that there are several identical maximal entries, for example
            # import numpy as np
            # a=[0,1,1,0]
            # np.where(a==np.max(a))
            # this will return [1,2], but we only need a single index
            # that is why we need to have np.random.choice(np.where(a==np.max(a))[0])
            # note that zero has to be added here since np.where() returns a tuple

    def trainNetwork(self):
        # if the replay buffer has at least batchReplayBufferSize elements,
        # then train the model
        # otherwise wait until the size of the elements exceeds batchReplayBufferSize
        if (len(self.replayBuffer) > self.batchReplayBufferSize):

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

                # if the next state is the terminal state
                if terminated:
                    y = reward
                    # if the next state if not the terminal state
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
            self.counterUpdateTargetNetwork += 1
            if (self.counterUpdateTargetNetwork > (self.updateTargetNetworkPeriod - 1)):
                # copy the weights to targetNetwork
                self.targetNetwork.set_weights(self.onlineNetwork.get_weights())
                print("Target network updated!")
                print("Counter value {}".format(self.counterUpdateTargetNetwork))
                # reset the counter
                self.counterUpdateTargetNetwork = 0
