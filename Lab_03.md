Week 03: Graded Lab

Lab session

Welcome to the third week of our Reinforcement Learning course! In this lab session, we will delve into more advanced reinforcement learning techniques and challenge you to implement and extend what you've learned in the previous two weeks. This graded lab is designed to be more time-consuming and challenging, but it will be a rewarding experience as you tackle complex problems in reinforcement learning.

Educational Objectives


	
Be able to extend the Monte Carlo algorithm to work with a state value function while maintaining a balance between exploration and exploitation.
	
Be able to implement the incremental Monte Carlo algorithm.
	
Be able to apply the Q-Learning algorithm from scratch.


Getting Started

Please group up in pairs and open the notebook from week 01 in either Google Colab or on your local machine.

Tasks

This graded lab is designed for you to prove your understanding of the learned methods. The tasks are threefold:


	
Monte Carlo with State Value Function: For the first task, you are required to take the Monte Carlo algorithm from the first week's lab (Grid World) and modify it to work with a state-value function instead of a random choice. This means that instead of randomly sampling from all possible actions, you sample according to the estimated state-value function. You should also incorporate a strategy to balance exploration and exploitation using ε-Greedy.
	
Incremental Monte Carlo: In the second task, extend the Monte Carlo algorithm from the first task to an incremental Monte Carlo method. Incremental Monte Carlo updates the value estimates incrementally, reducing the need to wait until the end of an episode to update values. This should lead to more efficient learning.
	
Q-Learning Integration: For the final task, you should replace the Monte Carlo method used and extended in the first two tasks with the Q-Learning algorithm, as introduced last week. This will allow you to compare the performance of Q-Learning with the Monte Carlo approach and observe how the learning process differs.


Submission

The deadline for completing this graded lab is the start of the lab session in week 7 (31st of October, 16:00). If you are attending the lecture in person, please be prepared to show your results upfront. If you are attending online, you should send an email to embe@zhaw.ch with your lab results attached. Make sure to include the names of both students if you are working in a pair.

Key Takeaways


	
Advanced Techniques: You will be applying advanced reinforcement learning techniques, including state value functions, incremental Monte Carlo, and Q-Learning.
	
Efficient Learning: Incremental Monte Carlo is designed to make learning more efficient by updating value estimates incrementally.
	
Comparative Analysis: By implementing Q-Learning for the same GridWorld environment, you will gain insights into the performance differences between the Monte Carlo method and Q-Learning in action.