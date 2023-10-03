Week 02: Exploration

 

Lab session

Welcome to the second lab session of our Reinforcement Learning course! This time you will try your hand on the Q-Learning algorithm, which is a step up from last week's Monte Carlo method. This lab will build upon the concepts introduced in our lecture and help you apply your knowledge to a simple problem.

 

Educational Objectives


	
Know how to put the learning in Reinforcement Learning.
	
Know the basis of Markoc decision processes, Monte Carlo algorithms, Dynamic programming, and Time-Differences.
	
Explain how value-functions and action-value functions can be learned using these approaches.
	
Be able to implement action-value based Reinforcement Learning.


 

Getting Started

Please group up in pairs (alone is just as fine) and open the notebook in either Google Colab or on your local machine.

 

Tasks

This lab is designed for you to explore and experiment with Q-Learning. Read the explanations in the notebook, let the code run and observe the resulting plot. Then, do the following tasks:


	
Where is the Q-Learning algorithm?: Find out where the training is happening. Can you identify the main loop? Find out where the Q-table gets updated. What is the value of the learning rate?
	
Mess with the code: Set the learning rate to 0.1. Retrain and look at the performance. How does it change? Can you explain the influence of the initial implementation?
	
Extend the plotting: Plot both the learning rate and the epsilon over the course of the training.
	
Compare: Write a simple algorithm to choose an action by hand and compare it to the Q-Learning agent. How does it compare? Consider also that the Cartpole problem is theoretically a very easy and simple problem, which can be solved by calculating the physics of the problem, but an RL-agent does not have that prior knowledge.


 

Key Takeaways


	
Q-Learning: Q-Learning is a powerful model-free reinforcement learning algorithm that helps agents learn the values of state-action pairs to make optimal decisions.
	
Building on Foundations: Q-Learning builds on the foundational concepts of MDPs, Monte Carlo algorithms, Dynamic programming, and Time-Differences.
	

		
Monte Carlo learns from entire trajectories.
		
Dynamic Programming learns from looking one step ahead on all actions
		
Q-Learning looks one step ahead on one action.
	
	
	
Value Functions and Action-Value Functions: Q-Learning is an action-value based approach that helps agents learn both value functions (V) and action-value functions (Q) to make informed decisions.
	
Experiment and Iterate: The best way to grasp the nuances of Q-Learning is through experimentation. Try different scenarios, hyperparameters, and modifications to gain a deeper understanding.
	
Real-World Applications: Q-Learning has real-world applications in robotics, game AI, finance, and many other domains.


 

Prominent RL examples


	
Robotics: Google DeepMind used Q-Learning to teach virtual humans and creatures to move over obstacles in 2017 (video, publication)
	
QT-Opt: Google AI used a variant of deep Q-Learning to teach robots to grasp objects, with 96% successful grasp attempts in 2018 (video, preprint).
	
TD-Gammon: Using Temporal Difference Learning, TD-Gammon almost rivaled the top human backgammon players in 1992, with the latest version making significantly less errors than the top backgammon player Malcolm Davis in 1998.
	
Deep Q-Network (DQN): DeepMind trained DQN to play Atari games, beating competitive RL agents and even human players by a landslide in 2013 (paper).