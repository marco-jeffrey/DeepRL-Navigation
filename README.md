# DeepRL-Navigation

![pretrained banana finder](https://github.com/MJPansa/DeepRL-Navigation/blob/master/banana_navigation.gif)

This is the first project of the Udacity Deep Reinforcement Learning Nanodegree. I used DeepQ Learning to train an agent to collect bananas in the provided Unity environment

## About
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

This implementation can either be run in training mode to train an intelligent agent from scratch, or it can be run in evaluation mode to see the performance of the pretrained agent.

## Installation

_NOTE: This environment **only** works with **Linux**_

ml-agents demands the usage of Python 3.6. The easiest way is to use a new conda environment like so:
```
conda create -name bananas python=3.6
conda activate bananas
```
Then clone the repo and install the requirements:
```
git clone https://github.com/MJPansa/DeepRL-Navigation
pip install -r requirements
```
## Usage
* Use `python main.py train` for training the agent from scratch

* Use `python main.py eval` for using the evaluation mode with the pretrained agent

## Results
The environment counts as solved after reaching an average score of at least 13 points over 100 episodes.
![Training results](https://github.com/MJPansa/DeepRL-Navigation/blob/master/training_stats.png)
* This implementation solves the environment after around 500 episodes.
