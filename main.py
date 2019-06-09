from Q_agent import Agent

import sys
from collections import deque

from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
import torch

if len(sys.argv)<2:
    print("ERROR: No mode specified. Please use either 'train' or 'eval' mode")
    sys.exit(1)
if sys.argv[1]=='train' or sys.argv[1]=='eval': 
    train = True if sys.argv[1]=='train' else False
else:
    print("ERROR: Unknown mode. Please use either 'train' or 'eval' mode")
    sys.exit(1)

if train:
    print("---------------------------------------------")
    print("----------RUNNING IN TRAIN-MODE--------------")
    print("---------------------------------------------")
else:
    print("---------------------------------------------")
    print("----------RUNNING IN EVAL-MODE---------------")
    print("---------------------------------------------")

env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64",no_graphics=train)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions and state space
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

    

#Initialize the agent
agent = Agent(state_size=state_size, action_size = action_size, seed=0,)

# Loading pretrained model for eval
if train==False:
    agent.qnetwork_local=torch.load('checkpoint_DQN.pth')
    agent.qnetwork_local.eval()

def dqn(n_episodes=2000 if train else 200, max_t=10000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    
    
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True if train else False)[brain_name] 
        state = env_info.vector_observations[0]  
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps if train else 0.0)
            env_info = env.step(action)[brain_name]        
            next_state = env_info.vector_observations[0]   
            reward = env_info.rewards[0]                   
            done = env_info.local_done[0]                                                 
            if train:
                agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state 
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            print('\n epsilon: '+str(eps))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local, 'checkpoint_DQN.pth')
            break
    env.close()
    return scores

scores = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
