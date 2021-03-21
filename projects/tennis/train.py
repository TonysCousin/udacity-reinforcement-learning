# Performs the training of both agents for the Tennis project, based on the Unity ML-Agents
# simulation environment.

import numpy as np
import torch
import time
from collections import deque

from maddpg      import Maddpg

AVG_SCORE_EXTENT = 100 #number of episodes over which running average scores are computed
CHECKPOINT_PATH = "checkpoint/" #can be empty string, but if a dir is named, needs trailing '/'


def train(maddpg, env, run_name="UNDEF", starting_episode=0, max_episodes=2, max_time_steps=100,
          sleeping=False, winning_score=0.0, checkpoint_interval=1000):

    """Trains a set of DRL agents in a Unity ML-Agents environment.

       Params:
           maddpg (Maddpg):           the manager of the agents and learning process
           env (UnityEnvironment):    the game environment
           run_name (string):         description of training session (used in status reporting
                                        and checkpoint filenames)
           starting_episode (int):    episode # to begin counting from (used if restarting training
                                        from a checkpoint)
           max_episodes (int):        max number of episodes to train if winning condition isn't met
           max_time_steps (int):      number of time steps allowed in an episode before forcing end
           sleeping (bool):           should the code pause for a few sec after selected episodes?
                                        (doing so allows for better visualizing of the environment)
           winning_score (float):     game score above which is considered a win (averaged over
                                        AVG_SCORE_EXTENT consecutive episodes)
           checkpoint_interval (int): number of episodes between checkpoint files being stored

       Return: list of episode scores (list of float)
    """

    # Initialize Unity simulation environment
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0]) #num states for one agent
    action_size = brain.vector_action_space_size #num actions for one agent
    #print("train: state_size = {}, action_size = {}".format(state_size, action_size)) #debug

    scores = []
    sum_steps = 0 #accumulates number of time steps exercised
    max_steps_experienced = 0
    recent_scores = deque(maxlen=AVG_SCORE_EXTENT)
    start_time = 0
    first_episode_timed = -1

    # loop on episodes
    for e in range(starting_episode, max_episodes):
        
        # Reset the enviroment & agents and get the initial state of environment & agents
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations #returns tensor(2, state_size)
        score = 0 #total score for this episode
        maddpg.reset()

        # start the timer after several episodes have passed, since early ones just fill the
        # replay buffer and go quickly; once learning starts, the pace slows down, and this
        # timer is used to predict total run duration
        if first_episode_timed < 0  and  maddpg.is_learning_underway():
            first_episode_timed = e
            start_time = time.perf_counter()
        #print("Top of episode {}: states = ".format(e), states) #debug

        # loop over time steps
        for i in range(max_time_steps):

            # Predict the best actions for the current state and store them in a single tensor
            actions = maddpg.act(states) #returns ndarray, one row for each agent
            #print("train: actions = ", actions, type(actions)) #debug

            # get the new state & reward based on this action
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations #returns ndarray, one row for each agent
            rewards = env_info.rewards #returns list of floats, one for each agent
            dones = env_info.local_done #returns list of bools, one for each agent
            #print("      next_states = ", next_states, type(next_states)) #debug
            #print("      rewards = ", rewards, type(rewards))
            #print("      dones = ", dones, type(dones))

            # update the agents with this new info
            maddpg.step(states, actions, rewards, next_states, dones) 

            # roll over new state and check for episode completion
            states = next_states
            score += np.max(rewards) #use the highest reward from the agents
            #print("       score = ", score) #debug
            if np.any(dones):
                sum_steps += i
                if i > max_steps_experienced:
                    max_steps_experienced = i
                break

        # determine epoch duration and estimate remaining time
        current_time = time.perf_counter()
        if start_time > 0:
            timed_episodes = e - first_episode_timed - starting_episode + 1
            avg_duration = (current_time - start_time) / timed_episodes / 60.0 #minutes
            remaining_time_minutes = (starting_episode + max_episodes - e - 1) * avg_duration
            rem_time = remaining_time_minutes / 60.0
            time_est_msg = "{:4.1f} hr rem".format(rem_time)
        else:
            avg_duration = 1.0 #avoids divide-by-zero
            time_est_msg = "???"

        # update score bookkeeping, report status and decide if training is complete
        scores.append(score)
        recent_scores.append(score)
        avg_score = np.mean(recent_scores)
        max_recent = np.max(recent_scores)
        mem_stats = maddpg.get_memory_stats() #element 0 is total size, 1 is num good experiences
        mem_pct = 0.0
        if mem_stats[0] > 0:
            mem_pct = min(100.0*float(mem_stats[1])/mem_stats[0], 99.9)
        print("\r{}\tRunning avg/max: {:.3f}/{:.3f},  mem: {:6d}/{:6d} ({:4.1f}%), avg {:.1f} eps/min   "
              .format(e, avg_score, max_recent, mem_stats[0], mem_stats[1], mem_pct, 
                      1.0/avg_duration), end="")
        if e > 0  and  e % checkpoint_interval == 0:
            maddpg.save_checkpoint(CHECKPOINT_PATH, run_name, e)
            print("\r{}\tAverage score:   {:.3f},        mem: {:6d}/{:6d} ({:4.1f}%), avg {:.1f} eps/min; {}   "
                  .format(e, avg_score, mem_stats[0], mem_stats[1], mem_pct,
                          1.0/avg_duration, time_est_msg))

        # if sleeping is chosen, then pause for viewing after selected episodes
        if sleeping:
            if e % 100 < 5:
                time.sleep(1) #allow time to view the Unity window

        if e > 100  and  avg_score >= winning_score:
            print("\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}"
                  .format(e, avg_score))
            maddpg.save_checkpoint(CHECKPOINT_PATH, run_name, e)
            break

    print("\nAvg/max time steps/episode = {:.1f}/{:d}"
          .format(float(sum_steps)/float(max_episodes-starting_episode),
                  max_steps_experienced))
    return scores

