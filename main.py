from unityagents import UnityEnvironment
import numpy as np
import random
from agents.dqn_agent import DQNAgent
import torch
from training.train import train
import matplotlib.pyplot as plt

def main():

    env = UnityEnvironment(file_name="/home/faten/projects/deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86_64")

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    state_size = len(state)

    agent = DQNAgent(state_size,action_size,seed=0)

    scores = train(env, agent)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)),scores)
    plt.ylabel('Score')
    plt.xlabel('Epsiode #')
    plt.show()

    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

    for i in range(3):
        state = env.reset()
        for j in range(200):
            action = agent.act(state)
            env.render()
            state, reward, done, _ = env.step(action)
            if done:
                break

    env.close()

main()

    # # instantiate the environment
    # env = UnityEnvironment(file_name="/home/faten/projects/deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86_64")

    # brain_name = env.brain_names[0]
    # brain = env.brains[brain_name]

    # action_size = brain.vector_action_space_size

    # env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    # state = env_info.vector_observations[0]            # get the current state
    # state_size = len(state)

    # #load the weights from file
    # agent = DQNAgent(state_size,action_size,seed=0)
    # agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

    # score = 0

    # for i in range(3):
    #     for j in range(200):
    #         action = agent.act(state)
    #         env_info = env.step(action)[brain_name]        # send the action to the environment
    #         next_state = env_info.vector_observations[0]   # get the next state
    #         reward = env_info.rewards[0]                   # get the reward
    #         done = env_info.local_done[0]                  # see if episode has finished
    #         score += reward                                # update the score
    #         state = next_state
    #         state,reward,done,_ = env.step(action)
    #         if done:
    #             break

    # print("Score: {}".format(score))
    # env.close()