# Multi-Agent Cooperation/Competition Project (Tennis)

![Tennis_trained](https://user-images.githubusercontent.com/9048857/114322523-9006ac00-9ad5-11eb-922d-82cd6067cd90.gif)

Within the Udacity Deep Reinforcement Learning nano-degree program, controling multiple agents in a cooperative or competetive environment is the third and final major project for students to build on their own.  The project is to build and train an agent using any policy-based techniques to control two players playing a game that, at first, looks like tennis.  There is one player (agent) on each side of the net, but they are actually cooperating, not competing.  The object is not to play the game of tennis, rather to hit the ball back and forth, keeping it in the air as long as possible.  

## Scoring and Winning

If a player misses the ball or hits it out of bounds, then that player receives a reward of -0.01 for that time step.  When a player makes a successful hit
(the ball stays in bounds), that player receives a reward of +0.1.  Rewards are accumulated throughout an episode,
which will last until the ball goes out of bounds or hits the ground, or until the maximum number of time steps is 
reached.  The score of that episode is the max of each agent's accumulated reward (one agent may have a few more hits
than the other, such as hitting it vertically to itself).

The agents are considered successfully trained when they can play 100 consecutive episodes with an average score of 0.5 or greater.

## The Environment

The game is played using the Unity ML-Agents gaming framework, and uses a custom environment built by Udacity instructors.  The game is a 2D tennis court.
The agents can move horizontally toward the net and away from it, within their court bounds, and they can also move their racquet up and down.
The angle of the racquet is fixed.
At each time step the environment returns a state vector comprised of 24 elements for each agent, describing that agent's point of view on the game.  It contains X and Y positions and velocities of both the agent and the ball over three consecutive time steps (8 elements for each time step).
Action vectors input to the environment have two elements for each agent, representing its desired horizontal and vertical motion.
Action values must be in the range [-1, 1].

## Solution

I used the Multi-Agent DDPG algorithm to train the agents.  See the [project report](docs/Report.md) for details. Note that this is a classroom project, and does not claim to be production-quality code. I got a huge education out of building and running it, but that is where its purpose ends. I hope you will find it an interesting and valuable reference to build upon or learn from.

## Installation

This project uses a pre-built Unity ML-Agents envronment, which means it is unnecessary to install Unity itself. The environment is included in this repo.

**Python 3.6 is required.**

After cloning this repo, cd into the project directory and run

	`pip install .`

## Run the Code

The main.py file contains the main program to run:

	`python3.6 main.py`

It will either run in training mode or inference mode.
As delivered, it will run my solution model in inference mode as a demonstration.
To choose a different model from the stored checkpoints, or to train a new model, simply edit main.py to change the hyperparameters as desired.
Since training on this model was fairly difficult, and required a lot of exploration of various hyperparameters, the logic here performs an exploration of hyperparameter combinations over several runs for a given "configuration" (code base).
Note that I have included a sample for how to set up for training, in _explore_example.py_ .
You may want to use this as a starting point for more refined training.  Just run

	`python3.6 explore_example.py`



----- REPLACE BELOW -----


### To use this code

Code for this project lives in two places within this directory.  The first is the Jupyter notebook,
_cont-ctrl-project.ipynb_ , with the main logic in the flat files, `ddpg_agent.py` and `model.py`.
Once these dependencies are in place, simply open the notebook, select the drlnd kernel, and
run the entire notebook.  It will train the agent given the hyperparameters that appear in the code, and
show a plot of its training history.

Also included in this repository are checkpoint files that hold the pre-trained models, one for the actor network
and one for the critic network.  The final two cells in the notebook will read in these files and play an inference
episode with the trained models.

