# Performs the training of both agents for the Tennis project, based on the Unity ML-Agents
# simulation environment.

import numpy as np
import torch
import time
from collections import deque

from maddpg      import Maddpg

AVG_SCORE_EXTENT = 100 #number of episodes over which running average scores are computed
CHECKPOINT_PATH = "checkpoint/" #can be empty string, but if a dir is named, needs trailing '/'
ABORT_EPISODE = 4000


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

       Return: tuple of (list of episode scores (floats), list of running avg scores (floats))
    """

    # Initialize Unity simulation environment
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0]) #num states for one agent
    action_size = brain.vector_action_space_size #num actions for one agent
    num_agents = len(env_info.agents)
    states = env_info.vector_observations #returns ndarray(2, state_size)

    actions = np.ndarray((num_agents, action_size))
    next_states = np.ndarray((num_agents, state_size))
    rewards = []
    dones = []
    # collect raw & running avg scores (over 100 episodes) at each episode
    scores = []
    avg_scores = []
    sum_steps = 0 #accumulates number of time steps exercised
    max_steps_experienced = 0
    recent_scores = deque(maxlen=AVG_SCORE_EXTENT)
    start_time = 0

    # run the simulation several time steps to prime the replay buffer
    print("Priming the replay buffer", end="")
    pc = 0
    while not maddpg.is_learning_underway():
        states, actions, rewards, next_states, dones = \
          advance_time_step(maddpg, env, brain_name, states, actions, rewards, next_states, dones)
        if pc % 4000 == 0:
            print(".", end="")
        pc += 1
        if any(dones):
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations #returns ndarray(2, state_size)
    print("!\n")


    # loop on episodes for training
    start_time = time.perf_counter()
    for e in range(starting_episode, max_episodes):
        
        # Reset the enviroment & agents and get their initial states
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations #returns ndarray(2, state_size)
        score = 0 #total score for this episode
        maddpg.reset()

        # loop over time steps
        for i in range(max_time_steps):

            # advance the MADDPG model and its environment by one time step
            states, actions, rewards, next_states, dones = \
              advance_time_step(maddpg, env, brain_name, states, actions, rewards, next_states, dones)

            # check for episode completion
            score += np.max(rewards) #use the highest reward from all agents
            if np.any(dones):
                sum_steps += i
                if i > max_steps_experienced:
                    max_steps_experienced = i
                break

        # determine episode duration and estimate remaining time
        current_time = time.perf_counter()
        rem_time = 0.0
        if start_time > 0:
            timed_episodes = e - starting_episode + 1
            avg_duration = (current_time - start_time) / timed_episodes / 60.0 #minutes
            remaining_time_minutes = (starting_episode + max_episodes - e - 1) * avg_duration
            rem_time = remaining_time_minutes / 60.0
            time_est_msg = "{:4.1f} hr rem".format(rem_time)
        else:
            avg_duration = 1.0 #avoids divide-by-zero
            time_est_msg = "???"

        # update score bookkeeping, report status
        scores.append(score)
        recent_scores.append(score)
        # don't compute avg score until several episodes have completed to avoid a meaningless
        # spike near the very beginning
        avg_score = 0.0
        if e > 50:
            avg_score = np.mean(recent_scores)
        max_recent = np.max(recent_scores)
        avg_scores.append(avg_score)
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
            if e % 100 < 20:
                time.sleep(1) #allow time to view the Unity window

        # if we have met the winning criterion, save a checkpoint and terminate
        if e > 100  and  avg_score >= winning_score:
            print("\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}"
                  .format(e, avg_score))
            maddpg.save_checkpoint(CHECKPOINT_PATH, run_name, e)
            print("\nMost recent individual episode scores:")
            for j, sc in enumerate(recent_scores):
                print("{:2d}: {:.2f}".format(j, sc))
            break

        # if this solution is clearly going nowhere, then abort early
        if e > starting_episode + ABORT_EPISODE:
            hit_rate = float(mem_stats[1]) / e
            if hit_rate < 0.01  or  (rem_time > 1.0  and  hit_rate < 0.05):
                print("\n* Aborting due to inadequate progress.")
                break

    print("\nAvg/max time steps/episode = {:.1f}/{:d}"
          .format(float(sum_steps)/float(max_episodes-starting_episode),
                  max_steps_experienced))
    return (scores, avg_scores)


def advance_time_step(model, env, brain_name, states, actions, rewards, next_states, dones):
    """Advances the agents' model and the environment to the next time step, passing data
       between the two as needed.

       Params
           model (Maddpg):         the MADDPG model that manages all agents
           env (UnityEnvironment): the environment object in which all action occurs
           brain_name (string):    an index into the Unity data structure for this environment
           states (ndarray):       array of current states of all agents and environment [n, x]
           actions (ndarray):      array of actions by all agents [n, x]
           rewards (list):         list of rewards from all agents [n]
           next_states (ndarray):  array of next states (after action applied) [n, x]
           dones (list):           list of done flags (int, 1=done, 0=in work) [n]
       where, in each param, n is the number of agents and x is the number of items per agent.

       Returns: tuple of (s, a, r, s', done) values
    """

    # Predict the best actions for the current state and store them in a single ndarray
    actions = model.act(states) #returns ndarray, one row for each agent

    # get the new state & reward based on this action
    env_info = env.step(actions)[brain_name]
    next_states = env_info.vector_observations #returns ndarray, one row for each agent
    rewards = env_info.rewards #returns list of floats, one for each agent
    dones = env_info.local_done #returns list of bools, one for each agent

    # update the agents with this new info
    model.step(states, actions, rewards, next_states, dones) 

    # roll over new state
    states = next_states

    return (states, actions, rewards, next_states, dones)
