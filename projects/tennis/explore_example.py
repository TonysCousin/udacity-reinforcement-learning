# Main function for the Tennis project if running from the command line.
# Edit the hyperparameters in here as desired, then run the program
# with no arguments.
#
# Running this function is identical to doing the training from the Jupyter
# notebook, but it does not attempt to plot the resulting scores.

import numpy as np
import time

from unityagents import UnityEnvironment

from train import train
from maddpg import Maddpg
from random_sampler import RandomSampler

def main():

    """
    EXPLORE determines whether the notebook does exploratory training or inference demonstration.
        * True runs a hyperparameter exploration loop to generate many training runs with a random
        search algorithm. To use this well, you should study that cell and specify the ranges of
        hyperparameters to be explored.
        * False runs a few inference episodes of a pretrained model and opens a visualization window
        to watch it play.

    config_name: the name of a model configuration to be loaded from a checkpoint to begin the
    exercise.
        * If EXPLORE = True, this is optional, and tells the training loop to start from this
        pre-trained model and continue refining it; if the value is None then the training starts
        from a randomly initialized model.
        * If EXPLORE = False, then this must reflect the name of a legitimate config/run (e.g.
        "M37.01").

    run_number:  sequential run ID number from a given configuration. Only used if config_name is
    not None.

    checkpoint_episode: if a checkpoint is being used to start the exercise, then this number
    reflects what episode that checkpoint was captured from. The checkpoint_name and
    checkpoint_episode together are required to completely identify the checkpoint file.

    training_viz controls whether training will be visualized or not (only has meaning when
    EXPLORE = True). With visualization off training will run a bit faster.
    """

    EXPLORE            = True
    config_name        = None # Must be None if not using!
    run_number         = 1
    checkpoint_episode = 5005
    training_viz       = True


    #---------- Set up game environment & get info about it

    initial_episode = checkpoint_episode
    checkpoint_path = "checkpoint/{}/".format(config_name)
    tag = "{}.{:02d}".format(config_name, run_number)

    if EXPLORE:
        turn_off_graphics = not training_viz
        initial_episode = 0
        unity_train_mode = True
        if config_name != None:
            initial_episode = checkpoint_episode
    else:
        turn_off_graphics = False
        unity_train_mode = False

    # create a new Unity environment
    # it needs to be done once, outside any loop, as closing an environment then restarting causes
    # a Unity exception about the handle no longer being active.
    env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64", seed=0, 
                           no_graphics=turn_off_graphics)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]                       
    env_info = env.reset(train_mode=unity_train_mode)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]


    #---------- Explore hyperparameter combinations for training solutions

    # Use a random search for the hyperparams

    TIME_STEPS         = 600

    if EXPLORE:

        # fixed for the session:
        RUN_PREFIX        = "EXAMPLE"
        EPISODES          = 10001
        NUM_RUNS          = 6
        BUFFER_PRIME_SIZE = 50
        WEIGHT_DECAY      = 1.0e-5 #was 1.0e-5
        GAMMA             = 0.99
        LR_ANNEAL_FREQ    = 10000 #episodes
        LR_ANNEAL_MULT    = 1.0
        WIN_SCORE         = 1.0
        SEED              = 44939 #(0, 111, 468, 5555, 23100, 44939)

        # session variables:
        vars = [
                ["discrete",         0.15,      1.00],      #BAD_STEP_PROB
                ["continuous-float", 0.999000,  0.999900],  #NOISE_DECAY
                ["discrete",         0.010],                #NOISE_SCALE (was 0.040, 1.0)
                ["continuous-float", 0.000800,  0.000900],  #LR_ACTOR  (was 0.000010, 0.000080)
                ["continuous-float", 0.1,      0.3],        #LR_RATIO (determines LR_CRITIC)
                ["discrete",         2,        4],          #LEARN_EVERY
                ["continuous-int",   1,         2],         #LEARN_ITER
                ["continuous-float", 0.00100,   0.01000],   #TAU
                ["discrete",         256]                   #BATCH
               ]
        rs = RandomSampler(vars)

        print("Ready to train {} over {} training sets for {} episodes each, with fixed params:"
              .format(RUN_PREFIX, NUM_RUNS, EPISODES))
        print("    Max episodes   = ", EPISODES)
        print("    Weight decay   = ", WEIGHT_DECAY)
        print("    Gamma          = ", GAMMA)
        print("    LR anneal freq = ", LR_ANNEAL_FREQ)
        print("    LR anneal mult = ", LR_ANNEAL_MULT)
        print("    Buf prime size = ", BUFFER_PRIME_SIZE)

        for set_id in range(NUM_RUNS):

            # sample the variables
            v = rs.sample()
            BAD_STEP_PROB = v[0]
            NOISE_DECAY   = v[1]
            NOISE_SCALE   = v[2]
            LR_ACTOR      = v[3]
            LR_CRITIC     = v[4] * LR_ACTOR
            LEARN_EVERY   = v[5]
            LEARN_ITER    = v[6]
            TAU           = v[7]
            BATCH         = v[8]

            buffer_size = 100000

            RUN_NAME = "{}.{:02d}".format(RUN_PREFIX, set_id)
            print("\n///// Beginning training set ", RUN_NAME, " with:")
            print("      Batch size       = {:d}".format(BATCH))
            print("      Buffer size      = {:d}".format(buffer_size))
            print("      Bad step prob    = {:.4f}".format(BAD_STEP_PROB))
            print("      Noise decay      = {:.6f}".format(NOISE_DECAY))
            print("      Noise scale      = {:.3f}".format(NOISE_SCALE))
            print("      LR actor         = {:.7f}".format(LR_ACTOR))
            print("      LR critic        = {:.7f}".format(LR_CRITIC))
            print("      Learning every     ", LEARN_EVERY, " time steps")
            print("      Learn iterations = ", LEARN_ITER)
            print("      Tau              = {:.5f}".format(TAU))
            print("      Seed             = ", SEED)

            ##### instantiate the agents and perform the training

            maddpg = Maddpg(state_size, action_size, 2, bad_step_prob=BAD_STEP_PROB,
                            random_seed=SEED, batch_size=BATCH, buffer_size=buffer_size,
                            noise_decay=NOISE_DECAY, buffer_prime_size=BUFFER_PRIME_SIZE,
                            learn_every=LEARN_EVERY, 
                            learn_iter=LEARN_ITER, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC,
                            lr_anneal_freq=LR_ANNEAL_FREQ, lr_anneal_mult=LR_ANNEAL_MULT,
                            weight_decay=WEIGHT_DECAY, gamma=GAMMA, noise_scale=NOISE_SCALE,
                            tau=TAU)

            if config_name != None:
                print("///// Beginning training from checkpoint for {}, episode {}" \
                      .format(tag, initial_episode))
                maddpg.restore_checkpoint(checkpoint_path, tag, initial_episode)

            scores, avgs = train(maddpg, env, run_name=RUN_NAME, starting_episode=initial_episode,
                                 max_episodes=EPISODES, sleeping=training_viz, winning_score=WIN_SCORE,
                                 max_time_steps=TIME_STEPS, checkpoint_interval=1000)

        print("\n\nDONE TRAINING!")


    #---------- Run an existing model in inference mode to watch it perform

    else: #not EXPLORE

        # load the pre-trained model
        model = Maddpg(state_size, action_size, 2)
        model.restore_checkpoint(checkpoint_path, tag, initial_episode)

        for i in range(10):                                        # play game for several episodes
            env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
            states = env_info.vector_observations                  # get the current state (for each agent)
            scores = np.zeros(num_agents)                          # initialize the score (for each agent)
            num_steps = 0
            while True:
                actions = model.act(states, is_inference=True, add_noise=False)
                env_info = env.step(actions)[brain_name]           # send all actions to tne environment
                next_states = env_info.vector_observations         # get next state (for each agent)
                rewards = env_info.rewards                         # get reward (for each agent)
                dones = env_info.local_done                        # see if episode finished
                scores += env_info.rewards                         # update the score (for each agent)
                states = next_states                               # roll over states to next time step
                num_steps += 1
                if np.any(dones):                                  # exit loop if episode finished
                    time.sleep(2)
                    break
            print('Episode {}: {:5.3f}, took {} steps'.format(i, np.max(scores), num_steps))

    #---------- release the environment resources

    env.close()


if __name__ == "__main__":
    main()

