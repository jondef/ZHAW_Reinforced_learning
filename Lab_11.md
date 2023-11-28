# Week 11: DQN for CartPole

## Lab session

Welcome to the eleventh lab session of our Reinforcement Learning course! This week, we are diving further into the topic of Deep Q-Network (DQN). You'll receive a new notebook for the CartPole environment specifically designed for this lab, where your task will be to complete the implementation of the DQN algorithm. Your task will be to complete the missing parts of the code and compare the final outcome with your previous CartPole exercises. Please note that this notebook requires a GPU to run efficiently, and Google Colab is recommended for ease of access.

## Educational Objectives

- Understand how deep learning can be integrated with the DQN algorithm to solve reinforcement learning problems.
- Evaluate the impact of different batch sizes and network sizes on the learning performance of the DQN algorithm.
- Compare the performance and efficiency of DQN with your previous implementations that used the CartPole environment.

## Getting Started

Please group up in pairs and download the notebook for this week's lab either on Google Colab or on your local machine. To begin, please access the notebook titled "DQN_on_cartpole" provided for this lab session. This notebook is designed to guide you through implementing the DQN algorithm for the CartPole environment. Note that Google Colab is recommended for this exercise due to the requirement for GPU acceleration.

## Tasks

- **Complete DQN Implementation**: Complete the missing parts of the code in the notebook to implement the DQN algorithm for CartPole. Understand how neural networks are used to approximate the action-value function. Note that we use PyTorch to implement the neural network.
- **Comparative Analysis**: Compare the performance and behavior of the DQN implementation with your previous CartPole exercises. Analyze differences in learning curves, convergence, and overall stability.
- **Hyperparameter Tuning**: Experiment with different hyperparameters, such as batch sizes, learning rates, TAU, epsilon, discount factors, etc. Observe how they affect training convergence and performance. Make sure to play around with the size of the hidden layers and try to see what happens when you add or remove hidden layers.

## Key Takeaways

- **Deep Learning in RL**: Integrating deep learning with reinforcement learning enables more complex and scalable solutions for challenging problems.
- **Algorithm Extensions**: Implementing variations like Dueling DQN enhances your understanding of advanced concepts and their impact on reinforcement learning algorithms.
- **Analyzing Learning Curves**: Analyze learning curves, rewards per episode, and agent behavior to gain insights into the learning process and algorithm performance.
- **Comparative Analysis**: Comparing the performance of DQN with your previous implementations offers insights into the advantages and challenges of using neural networks in RL.
- **Hyperparameter Exploration**: Experimentation with batch sizes, network architectures, TAU, and epsilon values helps understand their influence on DQN's learning performance.
