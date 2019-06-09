# Project Report

The algorithm used for solving the environment is DQN as published by DeepMind in this [paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).
This is the standard implementation without any fancy tricks like Double DQN or Prioritized Experience Replay.

## Training

The environment was solved after around 500 episodes. Training performance can be judged by the picture below:
![Training results](https://github.com/MJPansa/DeepRL-Navigation/blob/master/training_stats.png)

#### DQN-Parameters

* maximum timesteps per episode: max_t = 10000
* epsilon start/end/decay: eps_max = 1 ; eps_end=0.01 ; eps_decay=0.995
* replay buffer size: BUFFER_SIZE = int(1e5)
* batch size for training: BATCH_SIZE = 64
* future discount factor: GAMMA = 0.99
* soft update of target network: TAU = 1e-3
* network update: UPDATE_EVERY = 4

#### Epsilon Greedy Behavior

![epsilon behavior in training](https://github.com/MJPansa/DeepRL-Navigation/blob/master/epsilon_training.png)

#### NN-Parameters

* Input Layer : 37 neurons
* 1st Hidden Layer: 128 neurons with ReLU activation
* 2nd Hidden Layer: 64 neurons with ReLU activation
* Output Layer: 4 neurons, no activations
* Optimizer: Adam with default values and lr = 5e-4
* Loss: Mean Squared Error

#### Further Improvements

Implementation of common DQN tricks to boost performance could be implemented such as:
* Dueling DQN
* Double DQN
* Prioritized Experience Replay

Apart from that:
* Better search for more suitable parameters could also yield improved results.
* Using a visual state space instead the smaller 37-dim state space might also result in imrpoved results, although it is more demanding hardwarewise for both experience replay storage as well as compute (GPU).
