# Performs the training of both agents for the Tennis project, based on the Unity ML-Agents
# simulation environment.
#
# Parameters:
# agent0              - the first tennis-playing agent in the game [DdpgAgent]
# agent1              - the second tennis-playing agent in the game [DdpgAgent]
# env                 - the Unity game environment [UnityEnvironment]
# run_name            - an arbitrary identifier used in the results output and checkpoint
#                         filenames [string]
# max_episodes        - max number of episodes to train before forced to give up [int]
# max_time_steps      - max number of time steps in each episode [int]
# sleeping            - do we desire a pause after each episode? (helpful for visualizing) [bool]
# winning_score       - target reward that needs to be exceeded over 100 consecutive episodes
#                         to consider training complete [float]
# checkpoint_interval - number of episodes between checkpoint storage [int]
#
# Returns: list of scores (rewards) from each episode [list of int]

import numpy as np
import torch
import time
from collections import deque

from ddpg_agent  import DdpgAgent

AVG_SCORE_EXTENT = 100 #number of episodes over which running average scores are computed
CHECKPOINT_PATH = "checkpoint/" #can be empty string, but if a dir is named, needs trailing /


def train(agent0, agent1, env, run_name="UNDEF", max_episodes=2,
          max_time_steps=100, sleeping=False, winning_score=0.0, checkpoint_interval=1000):

    # Initialize Unity simulation environment
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0]) #num states for one agent
    action_size = brain.vector_action_space_size #num actions for one agent
    #print("train: state_size = {}, action_size = {}".format(state_size, action_size)) #debug

    scores = []
    sum_steps = 0 #accumulates number of time steps exercised
    recent_scores = deque(maxlen=AVG_SCORE_EXTENT)
    start_time = time.perf_counter()
    starting_episode = 0 #could be used for continuing training from a checkpoint

    # loop on episodes
    for e in range(starting_episode, max_episodes):
        
        # Reset the enviroment and get the initial state of environment & agents
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations #returns tensor(2, state_size)
        score = 0 #total score for this episode
        #print("Top of episode {}: states = ".format(e), states) #debug

        # loop over time steps
        for i in range(max_time_steps):

            # Predict the best actions for the current state and store them in a single tensor
            action0 = agent0.act(states[0]) #returns ndarray(1, 2)
            action1 = agent1.act(states[1])
            actions = np.concatenate((action0.reshape(1, action_size),
                                      action1.reshape(1, action_size)), 0)
            #print("Step", i, "action0 = ", action0) #debug
            #print("       action1 = ", action1)
            #print("       actions = ", actions)

            # get the new state & reward based on this action
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations #returns ndarray(2, state_size)
            rewards = env_info.rewards #returns list
            dones = env_info.local_done #returns list
            #print("       next_states = ", next_states, type(next_states)) #debug
            #print("       rewards = ", rewards, type(rewards))
            #print("       dones = ", dones)

            # update the agents with this new info
            agent0.step(states[0], actions[0], rewards[0], next_states[0], dones[0]) 
            agent1.step(states[1], actions[1], rewards[1], next_states[1], dones[1]) 

            # roll over new state and check for episode completion
            states = next_states
            score += np.max(rewards) #use the highest reward from the agents
            #print("       score = ", score) #debug
            if np.any(dones):
                sum_steps += i
                break

        # determine epoch duration and estimate remaining time
        current_time = time.perf_counter()
        avg_duration = (current_time - start_time) / (e - starting_episode + 1) / 60.0 #minutes
        remaining_time_minutes = (starting_episode + max_episodes - e - 1) * avg_duration
        rem_time = remaining_time_minutes / 60.0
        time_est_msg = "{:4.1f} hr rem.".format(rem_time)

        # update score bookkeeping, report status and decide if training is complete
        scores.append(score)
        recent_scores.append(score)
        avg_score = np.mean(recent_scores)
        print('\rEpisode {}\tAverage Score: {:.3f}, avg {:.0f} episodes/min'
              .format(e, avg_score, 1.0/avg_duration), end="")
        if e > 0  and  e % checkpoint_interval == 0:
            torch.save(agent0.actor_local.state_dict(),  '{}{}_checkpoint0a_{:d}.pt'
                       .format(CHECKPOINT_PATH, run_name, e))
            torch.save(agent0.critic_local.state_dict(), '{}{}_checkpoint0c_{:d}.pt'
                       .format(CHECKPOINT_PATH, run_name, e))
            torch.save(agent1.actor_local.state_dict(),  '{}{}_checkpoint1a_{:d}.pt'
                       .format(CHECKPOINT_PATH, run_name, e))
            torch.save(agent1.critic_local.state_dict(), '{}{}_checkpoint1c_{:d}.pt'
                       .format(CHECKPOINT_PATH, run_name, e))
            print('\rEpisode {}\tAverage Score: {:.2f}\t{}             '
                  .format(e, avg_score, time_est_msg))

        if sleeping:
            if e % 50 < 5:
                time.sleep(1) #allow time to view the Unity window

        if e > 100  and  avg_score >= winning_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e, avg_score))
            torch.save(agent0.actor_local.state_dict(),  '{}{}_checkpoint0a.pt'
                       .format(CHECKPOINT_PATH, run_name))
            torch.save(agent0.critic_local.state_dict(), '{}{}_checkpoint0c.pt'
                       .format(CHECKPOINT_PATH, run_name) 
            torch.save(agent1.actor_local.state_dict(),  '{}{}_checkpoint1a.pt'
                       .format(CHECKPOINT_PATH, run_name))
            torch.save(agent1.critic_local.state_dict(), '{}{}_checkpoint1c.pt'
                       .format(CHECKPOINT_PATH, run_name))
            break

    print("Avg time steps/episode = {:.1f}"
          .format(float(sum_steps)/float(max_episodes-starting_episode)))
    return scores

