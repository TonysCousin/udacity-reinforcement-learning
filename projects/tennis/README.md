# Multi-Agent Cooperation/Competition Project (Tennis)

![Tennis Game](Reacher-snapshot.png) JOHN FIX THIS!

Within the Udacity Deep Reinforcement Learning nano-degree program, controling multiple agents in a cooperative or
competetive environment is the third and final major
project for students to build on their own.  The project is to build and train an agent using any policy-based techniques
to control two players playing a game that, at first, looks like tennis.  There is one player (agent) on each side
of the net, but they are actually cooperating, not competing.  The object is not to play the game of tennis, rather to
hit the ball back and forth, keeping it in the air as long as possible.  

## Scoring and Winning

If a player misses the ball or hits it
out of bounds, then that player receives a reward of -0.01 for that time step.  When a player makes a successful hit
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
JOHN - fill in the report link

I used the Multi-Agent DDPG algorithm to train the agents.  See the [project report](XXX) for details.
Note that this is a classroom project, and does not claim to be production-quality code.  
I got a huge education out of building and running it, but that is where its purpose ends.
I hope you will find it an interesting and valuable reference to build upon or learn from.

## Installation


## Run the Code
JOHN - include video clip here

----- REPLACE BELOW -----

I use the DDPG algorithm to train the agent.  The project report with additional details can be found in the Jupyter notebook, _Report.ipynb_ , located in this project directory.

### To use this code

The environment needed to run the code can be set up in a few minutes.  Beyond installing Jupyter Notebook, here are the steps
(use of conda is optional, but Python 3.6 is mandatory):
```
conda create --name drlnd python=3.6
conda activate drlnd
```

Next, pick a location to clone this repository, clone it, install its dependencies, and set up the Jupyter kernel:
```
cd <location>
git clone https://github.com/TonysCousin/udacity-reacher.git .
pip install .
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
 
Finally, install the Unity ML-Agents environment pre-built for this project (which means you don't need to install Unity directly).
Download the environment [from here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip),
place it in your project directory and unzip it.

Code for this project lives in two places within this directory.  The first is the Jupyter notebook,
_cont-ctrl-project.ipynb_ , with the main logic in the flat files, `ddpg_agent.py` and `model.py`.
Once these dependencies are in place, simply open the notebook, select the drlnd kernel, and
run the entire notebook.  It will train the agent given the hyperparameters that appear in the code, and
show a plot of its training history.

Also included in this repository are checkpoint files that hold the pre-trained models, one for the actor network
and one for the critic network.  The final two cells in the notebook will read in these files and play an inference
episode with the trained models.

