# Navigation Project

Within the Udacity Deep Reinforcement Learning nano-degree program, the Navigation project is the first major
project for students to build on their own.  The project is to build and train an agent using DQN techniques
to play the Banana game in OpenAI's gym environment.  The object of this game is for the agent to move around
the provided 2D surface that has bananas scattered about, collecting yellow bananas and avoiding the blue
bananas.  The agent has four commands available to it:  move forward, move backward, turn left, turn right.
The environment provides a 37-element state vector for the agent, which includes ray-based perception info
(we are not using video pixels as inputs).  The agent collects +1 reward point for each yellow banana and -1
point for each blue banana.  A successful solution is considered if it maintains an average of at least +13
points over 100 consecutive game episodes.

This project explores several enhancements to the vanilla DQN algorithm (originally described in
http://files.davidqiu.com//research/nature14236.pdf), including 
- Fixed Q targets
- Experience replay buffer
- Prioritized experience replay (PER) (https://arxiv.org/pdf/1511.05952.pdf)
- Double DQN (https://arxiv.org/pdf/1509.06461.pdf)

### To use this code

Code for this project lives in two places within this GitHub directory.  The first is the Jupyter notebook,
_Navigation.ipynb_ , with some support code in the flat files, 'PrioritizedMemory.py' and 'SumTree.py',
which, together, provide the bulk of the PER capability.

