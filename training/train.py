import torch
from collections import namedtuple, deque
import numpy as np
import matplotlib.pyplot as plt


def train(env, agent, n_episodes= 1800, max_t = 1000, eps_start=1.0, eps_end = 0.01,
       eps_decay=0.996):
    """Deep Q-Learning

    Params
    ======
        n_episodes (int): maximum number of training epsiodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): mutiplicative factor (per episode) for decreasing epsilon

    """
    brain_name = env.brain_names[0]

    scores = [] # list containing score from each episode
    scores_window = deque(maxlen=100) # last 100 scores
    eps = eps_start
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state,eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]
            agent.step(state,action,reward,next_state,done)
            ## above step decides whether we will train(learn) the network
            ## actor (local_qnetwork) or we will fill the replay buffer
            ## if len replay buffer is equal to the batch size then we will
            ## train the network or otherwise we will add experience tuple in our
            ## replay buffer.
            state = next_state
            score += reward
            if done:
                break
            scores_window.append(score) ## save the most recent score
            scores.append(score) ## sae the most recent score
            eps = max(eps*eps_decay,eps_end)## decrease the epsilon
            print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)), end="")
            if i_episode %100==0:
                print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)))

            if np.mean(scores_window)>=13.0:
                print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                torch.save(agent.qnetwork_local.state_dict(),'checkpoint.pth')
                break
    return scores
