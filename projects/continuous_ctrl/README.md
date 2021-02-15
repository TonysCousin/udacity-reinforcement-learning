# Continuous Control Project

Within the Udacity Deep Reinforcement Learning nano-degree program, Continuous Control is the second major
project for students to build on their own.  The project is to build and train an agent using any policy-based techniques
to control the Reacher two-jointed robot arm in the Unity ML Agents environment.  The object is to move the arm
in such a way that its end point is always in the vicinity of the environment's moving target.  The target is
a sphere that randomly moves in a circular pattern in a plane parallel to the base table.  It moves either
clockwise or counter-clockwise, and varies its speed.  Sometimes it is essentially stationary.  This is not an
episodic task, so we arbitrarily limit trajectory length to a certain number of time steps and call it an episode.

The agent's possible actions comprise a vector of 4 real numbers in [-1, 1], which represent rotations of each
of the two arm joints.  The environment provides a state vector of 33 real values representing the sphere's
position and velocity as well as the positions, velocities and accelerations of each arm segment.  The arm is
not spatially constrained, and can fold back on itself (i.e. both segments can exist in the same space).  The
agent collects a small reward for each time step that the arm's end is "within the target vicinity".
Empirical evidence suggests that this means the end is within the target sphere.  The goal is to achieve an
average score of +30.0 or more over 100 consecutive episodes (there is no guidance on how long these episodes
must be).  My observation shows that the nominal reward is 0.04 per time step within the sphere, so for an
episode of 1000 time steps the maximum possible reward would be 40 points.


I use the DDPG algorithm to train the agent.  The project report with additional details can be found in the Jupyter notebook, _Report.ipynb_ , located in this project directory.

### To use this code

Code for this project lives in two places within this GitHub directory.  The first is the Jupyter notebook,
_cont-ctrl-project.ipynb_ , with the main logic in the flat files, `ddpg_agent.py` and `model.py`.
So the first step in using it is to clone [this repo](https://github.com/TonysCousin/udacity-reacher). 

The environment needed to run the code can be set up in a few minutes.  Beyond installing Jupyter Notebook,
[NumPy](http://www.numpy.org) and the [Unity ML agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md),
it will require installation of some dependencies from OpenAI Gym described in
https://github.com/udacity/deep-reinforcement-learning#dependencies.

Once these dependencies are in place, simply open the _cont-ctrl-project_ notebook, select the drlnd kernel, and
run the entire notebook.  It will train the agent given the hyperparameters that appear in the code, and
show a plot of its training history.

Also included in this repository are checkpoint files that hold the pre-trained models, one for the actor network
and one for the critic network.  The final cell in the notebook will read in these files and play an inference
episode with the trained models.

