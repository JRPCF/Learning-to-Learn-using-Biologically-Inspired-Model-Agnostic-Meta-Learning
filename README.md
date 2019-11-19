# Learning-to-Learn-using-Model-Agnostic-Meta-Learning-with-Non-Episodic-Memory

This is a project that plans on discovering how a Meta-RL architecture with non-episodic memory might perform on learning to learn and transferring knowledge to adjacent tasks. The project is mostly focused on merging Recurrent Geometric Neural Nets and Model-Agnostic MetaLearning and observing how agents perform on Open-AI Gym Atari Games. 

When  Deep  Mind  demonstrated  their  groundbreaking  results  on  training  algorithms  to  play Atari video games, their agents depended on hours of  training  in  order  to  accomplish  human  level capacity.  Humans  on  the  other  hand,  although never  reaching  super-human  capacity,  are  able  to comprehend  the  game  fairly  quickly. The  brain does  this  by  learning  to  learn  (or  Meta-Learning) to  play  games.  Recreating  this  in  artificial  agents has  resulted  in  Meta-RL  which  has been shown to closely  resemble how our brain uses dopamine to help our prefrontal cortex  learn. The next step  would be to see how memory influences this process of learning to learn. Learning through analogy is an essential part of learning to learn in humans. Humans naturally utilize relevant knowledge from  related  tasks  to  learn  new  tasks.

However, it is challenging to implement a form of memory that resembles the brain’s hierarchical temporal memory. This project aims to test how agents learn to learn a new task if they possess memory of related tasks. I plan on using a Geometric Neural Network,  which allows for non-episodic memory that is reliably and predictably stored using attention mechanisms, as the neural net component in my version of MAML. The analogy in the brain would be that this neural net would behave as the pre-frontal cortex with access to a specific  memory location in the hippocampus. This holds information the prefrontal cortex uses to preform a related task

The project mainly consists of three steps:

1. Creating the new version of MAML. To reduce the amount of coding I need to do I will be utilizing the MAML implementation by tristandeleu, [pytorch-maml-rl](https://github.com/tristandeleu/pytorch-maml-rl).

2. Training an regular MAML agent and the Geometric MAML agent on both [Demon Attack](https://en.wikipedia.org/wiki/Demon_Attack) and [SpaceInvaders](https://en.wikipedia.org/wiki/Space_Invaders) which both possess similar game dynamics. Compare each other

3. Training pretrained Geometric MAML agents on the opposite game. Compare to non pretrained agents.
