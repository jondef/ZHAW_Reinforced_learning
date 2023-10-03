Week 01: Introduction

 


Welcome to the first lab session of our Reinforcement Learning course! In today's session, you will dive into the exciting world of reinforcement learning and get hands-on experience with an example problem. This lab will build upon the concepts introduced in our lecture and help you apply your knowledge to a simple problem.

 
Educational Objectives
 


	
Know what reinforcement learning is and how it differs from supervised learning.



	
Know real-world applications of reinforcement learning.



	
Explain the basic concepts of reinforcement learning.



	
Be able to run and start working on simple RL implementations.

 
Getting Started
 
Please group up in pairs (alone is just as fine) and open the introductory notebook (see in Files) in either Google Colab or on your local machine. Note that we modified this notebook from Prof. Dr. Winder's original material. In this notebook, you will find a 2D text-based world where an agent can move in cardinal directions to reach a goal.
The goal of this lab is to use reinforcement learning techniques to help the agent find the optimal policy to reach its goal.
 
Tasks
 
This lab is designed for you to explore and experiment with various aspects of reinforcement learning.
After working through the notebook from top to bottom, try to get a better feel for how you can change the scenario to make it harder or what changes can help the agent learn better.
Here are some modifications you can try:
 


	
Grid World Modifications: Experiment with different grid sizes, add more terminating states, or change the rewards associated with specific states. This will help you understand how RL agents adapt to different environments.



	
Simulate Challenges: Add obstacles, cliffs, holes, or walls to the grid. See how these modifications impact the agent's behavior and learning process.



	
Policy Implementation: After optimizing for the optimal policy using the Monte Carlo algorithm, try implementing the policy by hand to see how well it performs in comparison.

 
Key Takeaways
 


	
Experimentation is Key: Experimentation is crucial to understand how RL agents learn.



	
Real-World Applications: Reinforcement Learning is not only applicable to games, but also has a real-world impact on robotics, recommendation systems, finance, and autonomous driving.

 
Prominent RL example
 
Go:


	
AlphaGo Fan defeated Fan Hui 5:0 in 2015, (176 GPUs, Elo rating 3144)



	
AlphaGo Lee defeated Lee Sedong 4:5 in 2016 (48 TPUs, Elo rating 3739, documentary on Youtube)



	
AlphaGo Master won 60:0 against professional players, including 3:0 against Ke Jie in 2017 (4 TPUs single machine, Elo rating 4858)



	
AlphaGo Zero beats AlphaGo Lee 100:0 and AlphaGo Master 89:11 in 2017 (4 TPUs single machine, Elo rating 5185, watch TwoMinutePapers video)



	
AlphaZero can generalize to chess, go, and shogi:



	
Chess: Won against Stockfish 8 for 28 out of 100 matches, with the remainind 72 being draws



	
Go: Beat AlphaGo Zero 60:40 (4 TPUs single machine, Elo rating 5018)



	
Shogi: Won against Elmo 90 times, lost 8 times, and drew twice.