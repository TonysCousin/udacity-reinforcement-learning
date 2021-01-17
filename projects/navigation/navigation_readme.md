# Navigation Project

Within the Udacity Deep Reinforcement Learning nano-degree program, the Navigation project is the first major
project for students to build on their own.  The project is to build and train an agent using DQN techniques
to play the Banana game in the Unity ML Agents environment.  The object of this game is for the agent to move
around the provided 2D surface that has bananas scattered about, collecting yellow bananas and avoiding the
blue bananas.  The agent has four commands available to it:  move forward, move backward, turn left, turn
right.  The environment provides a 37-element state vector for the agent, which includes ray-based perception
info (we are not using video pixels as inputs).  The agent collects +1 reward point for each yellow banana
and -1 point for each blue banana.  A successful solution is considered if it maintains an average of at
least +13 points over 100 consecutive game episodes.

This project explores several enhancements to the vanilla DQN algorithm ([originally described here](http://files.davidqiu.com//research/nature14236.pdf)), including 
- Fixed Q targets
- Experience replay buffer
- Prioritized experience replay (PER) (https://arxiv.org/pdf/1511.05952.pdf)
- Double DQN (https://arxiv.org/pdf/1509.06461.pdf)

The project report can be found in the Jupyter notebook, _Report.ipynb_ , located in this project directory.

### To use this code

Code for this project lives in two places within this GitHub directory.  The first is the Jupyter notebook,
_Navigation.ipynb_ , with some support code in the flat files, `PrioritizedMemory.py` and `SumTree.py`,
which, together, provide the bulk of the PER capability.  So the first step in using it is to clone this
repo (https://github.com/TonysCousin/udacity-reinforcement-learning).  Within that repo, go to the
`projects/navigation` directory to find all of the relevant files, including this readme.


The environment needed to run the code can be set up in a few minutes.  Beyond installing Jupyter Notebook,
[NumPy](http://www.numpy.org) and the [Unity ML agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md),
it will require installation of some dependencies from OpenAI Gym described in
https://github.com/udacity/deep-reinforcement-learning#dependencies.

Once these dependencies are in place, simply open the _Navigation_ notebook, select the drlnd kernel, and
run the entire notebook.  It will train the agent given the hyperparameters that appear in the code, and
show a plot of its training history.

Note that also included in this repository are several checkpoints that hold pre-trained models with various
DQN features, indicated by their filenames.  The checkpoints are stored in `*.pt` files.  It should be a
simple exercise to slightly modify the notebook code to read in one of the checkpoint files and use that 
model for inference purposes if desired.
