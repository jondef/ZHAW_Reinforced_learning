# Week 08: Actor-Critic (A2C), Annealing, and Upper Confidence Bound

## Lab session

Welcome to the eighth week of our Reinforcement Learning course! In this lab, we will provide you with a notebook by Dr. Janis Klaise that implements the REINFORCE algorithm in the CartPole environment. Your task will be to compare this implementation with your own, make modifications to print intermediate results, and experiment with hyperparameters to understand their impact on training.

## Educational Objectives

- Gain hands-on experience in modifying the provided REINFORCE algorithm to print intermediate results.
- Experiment with hyperparameters and observe their influence on the training process.
- Be able to implement A2C with the provided REINFORCE implementation.

## Getting Started

Please group up in pairs and open the notebook for this week either on Google Colab or on your local machine.

## Tasks

This lab offers the opportunity to experiment with A2C and explore different exploration strategies.

- Compare Implementations: Compare your own REINFORCE implementation with the provided code. Identify similarities and differences and understand how the algorithm works.
- Print Intermediate Results: Modify the code to print intermediate results during training. Track metrics such as returns per episode to gain insight into the agent's learning progress.
- Hyperparameter Tuning: Experiment with different hyperparameters, such as learning rates, discount factors, etc. Observe how they affect training convergence and performance.
- A2C: Implement the A2C algorithm. Instead of using the episode return, use the concept of "advantage," introduced in week 5, where you use the average of multiple episodes. This will make the training more stable.

## Key Takeaways

- Comparing Implementations: Comparing your implementation with existing code is a great way to understand different approaches to solving the same problem.
- Hyperparameter Exploration: The choice of hyperparameters can significantly impact the training process and agent performance. Experimentation is key to finding the right settings.
