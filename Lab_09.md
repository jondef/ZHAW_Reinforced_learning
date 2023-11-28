# Week 09: Double Q-Learning and Off-Policy MC Prediction

## Lab session

Welcome to the eighth week of our Reinforcement Learning course! This week, we will delve into the implementation of two important algorithms from Sutton's book (2020 version): Double Q-learning (section 6.7) and Off-policy Monte Carlo (MC) prediction (section 5.6) in the GridWorld environment. These algorithms introduce the concept of using one policy for behavior while learning value or policy functions in another. This understanding will be crucial as we approach Deep Reinforcement Learning in two weeks.

## Educational Objectives

- Be able to implement the Double Q-learning algorithm.
- Be able to implement the Off-policy MC prediction algorithm.
- Gain practical experience in adapting algorithms from theoretical descriptions to functional code.
- Experiment with hyperparameters and observe their influence on the training process.

## Getting Started

Please group up in pairs and make two copies of the GridWorld notebook that you have used before either on Google Colab or on your local machine. You're supposed to implement the two algorithms separately.

## Tasks

- **Double Q-learning Implementation**: Implement the Double Q-learning algorithm in the first copy of the GridWorld notebook. Understand how using two sets of Q-values can mitigate overestimation issues. Play with different grid sizes to better see how the Double Q-learning agent performs. Refer to the pseudo-code implementation from the slides or Sutton's RL-book.
- **Off-policy MC Prediction Implementation**: Implement the Off-policy MC Prediction algorithm in the second copy of the GridWorld notebook. Explore the concept of using a different policy for behavior and learning. Refer to the pseudo-code implementation from the slides or Sutton's RL-book.
- **Hyperparameter Tuning**: Experiment with different hyperparameters, such as learning rates, discount factors, etc. Observe how they affect training convergence and performance.

## Key Takeaways

- **Double Q-learning**: This algorithm addresses the overestimation bias in Q-learning by using two sets of Q-values, providing a more accurate estimation of state-action values.
- **Off-policy MC Prediction**: Off-policy MC Prediction allows learning from an old policy while following a new policy. This separation of behavior and learning policies is a key concept in reinforcement learning.
- **Practical Implementation**: Bridging the gap between theoretical algorithms and practical code is an essential skill. This lab provides an opportunity to strengthen this skill.
- **Hyperparameter Exploration**: The choice of hyperparameters can significantly impact the training process and agent performance. Experimentation is key to finding the right settings.
- **Preparation for Deep RL**: The concepts explored in this lab will become even more relevant as we transition to Deep Reinforcement Learning in the upcoming weeks.

## Practical Examples

- **Robotics**: Off-policy learning (including Off-policy MC prediction) is crucial in training robotic systems. The ability to learn from past experiences (off-policy) allows robots to leverage existing data to improve their performance over time.
- **Financial Trading**: In the context of algorithmic trading, off-policy learning can be applied to optimize trading strategies. Traders can learn from historical market data (off-policy) to develop more robust and adaptive trading algorithms.
