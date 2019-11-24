from unityagents import UnityEnvironment
import numpy as np

# instantiate the environment
env = UnityEnvironment(file_name="/home/faten/projects/deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86_64")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

action_size = brain.vector_action_space_size

# get the current state
state = env_info.vector_observations[0]

score = 0
while True:
    # select a random action
    action = np.random.randint(action_size)
    # send the action to the environment
    env_info = env.step(action)[brain_name]
    # get the next state
    next_state = env_info.vector_observations[0]
    # get the reward
    reward = env_info.rewards[0]
    # see if episode has finished
    done = env_info.local_done[0]
    # update the score
    score += reward
    # roll over the state to the next time step
    state = next_state
    # exit loop if the episode finished
    if done:
        break

print("Score: {}".format(score))
