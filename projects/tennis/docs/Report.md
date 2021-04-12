# Multi-Agent Cooperation and Competition Project Report

_4/11/2021_

Within the Udacity Deep Reinforcement Learning nano-degree program, controling multiple agents in a cooperative or competetive environment is the third and final major project for students to build on their own.  The project is to build and train an agent using any policy-based techniques to control two players playing a game that, at first, looks like tennis.  There is one player (agent) on each side of the net, but they are actually cooperating, not competing.  The object is not to play the game of tennis, per se, rather to hit the ball back and forth, keeping it in the air as long as possible.

[The project's README file](../README.md) describes how to install and run the code.
This report describes the technical aspects of how it works.

## Learning Algorithm

I used the Multi-Agent Deep Deterministic Policy gradient (MADDPG) algorithm to train the two bots.
Whereas the original DDPG was intended for a single agent, it used an actor-critic approach.
The actor neural network (NN) attempts to provide the agent's desired outputs (i.e. solving the problem at hand), based on observations of the environment.
Meanwhile, the critic NN observes the environment as well, but also observes the actor's outputs, and attemts to learn an effective _Q function_ to describe the losses that tell the actor how close it is to the target.
Both the actor and the critic are instantiated twice; one embodies the actual policy that is being learned, and one acts as a "target", whose outputs become a reference to which the policy can be compared.
To make this inherently unstable situation more tenable, the target NN's parameters are very slowly updated to move ever closer to those of the policy NN (whose parameters are also changing as learning evolves).
A key training hyperparameter is the _tau_ value that controls the rate that the target moves toward the policy network.

In MADDPG the above concept is extended so that essentially a copy of the actor-critic structure exists for each agent in the problem space.
When multiple agents are trying to simultaenously learn how to act in the same space, they are observing each other's learning experienc as well.
As such, their comrades' actions may be poorly chosen, and therefore poor inputs to the local agent's training algorithm.
This situation is called "non-stationarity", and is a major hurdle that causes instability in multi-agent training.
The key that makes MADDPG stable and successful is that, while each agent's actor NN only observes the environment from that agent's point of view (i.e. it may not see the whole picture), each critic NN is fed the entire state of the whole system of systems.
The critics have access to all state variables of all agents and of the world they operate in.
This visibility helps them to "coach" their actors toward reasonable solutions in a more smooth approach.
Since the critic networks are only used for training purposes, (they are not used in inference mode), this unnatural visibility is not "cheating".
When trained agents are tested in an operational envrionment (inference mode) they will continue to function well with just their own local observations, as they did during training.
MADDPG is described in detail [in this paper](https://arxiv.org/pdf/1706.02275.pdf).

## Neural Networks Used

**Each agent has two actor NNs**, a policy network and a target network.
These have identical structure so that the parameters (weights and biases) can be copied from one into the other (gradually merged).  The structure is three fully-connected layers:
- Layer 1 has 400 nodes, ReLU activation, and takes in 24 state variables (3 time steps containing 8 states each)
- Layer 2 has 256 nodes and a ReLU activation
- Layer 3 has 2 nodes, representing the horizontal and vertical motion of the racquet, and uses a _tanh_ activation to ensure outputs are in [-1, 1]
- There is 20% dropout after layers 1 and 2
- Parameters for the policy NN are updated with an Adam optimizer using an actor-specific learning rate.

**Each agent also has two critic NNs**, a policy network and a target network, whose structures are identical so that they can be copied like the actors are copied.  The structure is also three fully-connected layers:
- Layer 1 has 400 nodes, leaky ReLU activation, and takes in the same 24 state variables for the local agent as are input to the actor NN
- Layer 2 has 256 nodes, leaky ReLU activation, and blends the outputs of layer 1 with the actor's two output actions (402 total inputs)
- Layer 3 has 1 node and linear activation; its output represents a Q value used for training the actor NN.
- There is 20% dropout after layers 1 and 2
- Parameters for the policy NN are updated with an Adam optimizer using a critic-specific learning rate.

### Additional algorithmic notes

Classroom examples were provided that used (and the previous project also used) the Ornstein-Uhlenbeck noise generator.
As one of the many options I altered when trying to get the solution to converge, I replaced this generator with a simple Gaussian generator.
It is not clear how much that change contributed to the final success, but the Gaussian generator is being used in the final solution.
The OU generator code is in the repo, and can be dropped in by changing two lines of code (same interfaces) if further comparisons are desired.

I also added a learning rate scheduler for the actor NN and another for the critic NN, which enables annealing of the learning rate.
Experimenting with this eventually confirmed that annealing is not needed, so the hyperparameters I've chosen render it inactive, but it could be used if desired.

As is typical with reinforcement learning, in order to avoid temporal biases due to correlation between states of consecutive time steps, training relies on an experience replay buffer.
I began with a simple replay buffer (no prioritizing), but quickly observed that the vast majority of experiences (virtually all) resulted in negative reward.
Thinking that learning would occur more quickly if the agents saw positive rewards, I modified the replay buffer to limit the number of negative experiences it stores.
In addition, I added a "priming" feature to the replay buffer; the first N experiences it takes in are considered priming, and consist of experiences built entirely from random actions by the agents.
A replay buffer needs to have at least buffer_size experiences stored before it can start serving batches for training, but this additionally requires that the priming quota be filled (which I set at 5000, substantially larger than the batch size of 256).
The driving concern here was that my early learning experiences only involved a very small range of motion of the racquets, and starting with a substantial set of training experiences that covered the full range of possible motion would help to force agents to explore "corner cases" more.

## Hyperparameters Used

While wide ranges of most of these quantities were experimented with, these are the values that finally achieved success for me:

- Discount rate = 0.99
- Weight decay for the critics' Adam optimizer = 1e-5
- Actor learning rate = 0.0001281
- Critic learning rate = 0.0001176
- Tau (target update rate) = 0.00869
- Learn every 2 time steps with only 1 learning iteration at each opportunity
- Bad experience storage probabiliy = 0.15, meaning only 15% of experiences with a reward <= 0 were retained in the replay buffer (randomly chosen).
- Noise decay rate = 0.999231; using this with an initial noise scale of only 0.02, the noise reached insignificant values within the first 350 episodes.
- Replay buffer size was 100,000 experiences, but few training runs ever filled it; for some of those that did, learning performed well, so I don't think this was a signifncant factor.

With so many hyperparameters to tune, in addition to various code configurations (e.g. NN structures), it took a large amount of time to find a solution.
I began using a random search algorithm to help explore the hyperparameter space more efficiently, which became a very useful tool.  It is implemented in the _main.py_ and in the _random_sampler.py_ code.

## Results

Once all code was confirmed correct, and reasonable structure was in place, hyperparameter tuning led to several acceptable solutions.
I chose the one I named M46.20 to be featured in this report, as its learning history looked a bit more robust than the others.
Its NN paremeters are stored in the [checkpoint M46.20_1983.pt](checkpoint/M46/M46.20_1983.pt).
Its learning history is shown in the following two plots, the first being raw scores from each episode, and the second being the 100-point moving average score.

![Raw scores](docs/M46.20_scores.png)

![Moving avg)(docs/M46.20_avg_score.png)

I used this checkpoint as the baseline for some additional training (stored in the checkpoint/M46X directory), which took the model to new heights - trained until the average reward (over 100 consecutive episodes) exceeded 1.0, twice the requirement for this project.
This additional training performed very well and only took a few hundred extra episodes.

## Future Considerations

- Given the amount of time invested in this project, I did not have time to pursue the optional assignment that I really wanted to, training a pair of competing soccer teams.
I will extend my multi-agent education by solving that problem as well.
- It would be interesting to try using the TD3 training algorithm, which is supposed to be more stable due to avoiding an overestimation bias in the critic evaluation of the Q function.
- I also plan to begin extending these ideas into real world situations in my work with [cooperative driving automation](https://github.com/usdot-fhwa-stol/carma-platform).
