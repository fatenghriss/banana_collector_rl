# Banana Collector

This project consists in developing a reinforcement learning algorithm to train an agent to navigate and collect bananas in a large square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Installation
First, and in order to have a working environment that is as clean as possible, let's:
1. Create and activate a new environment with Python 3.6
- __Linux__ or __Mac__:
```
conda create --name bc python=3.6
source activate bc
```
- __Windows__:
```
conda create --name bc python=3.6
activate bc
```
2. Install dependencies:

```
pip install .
```

## Training

In order to train your agent and test your agent, all you have to do is the following:

```
python main.py
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
