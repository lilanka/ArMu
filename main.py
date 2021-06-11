#!/bin/python3

import gym
import numpy as np

from controller import Controller

env = gym.make('FetchPickAndPlace-v1')
obs = env.reset()

OBS_DIM = obs['observation'].shape[0] + \
          obs['achieved_goal'].shape[0] + \
          obs['desired_goal'].shape[0]
ACT_DIM = 4  
EPISODES = 120 
BUFFER_SIZE = 10000
BATCH_SIZE = 100
MEAN = 0.0
STD = (0.2, 0.2) # std and std_bar
TAU = 1
C = (-0.5, 0.5) # -c, c
lr = 1e-3
d = 1 # to set fixed number of updates

td3 = Controller(OBS_DIM, ACT_DIM, BUFFER_SIZE, BATCH_SIZE, MEAN, STD, TAU, C, lr, d)

for _ in range(EPISODES):
  env.render()
  action = td3.step(obs)
  new_obs, reward, done, info = env.step(action)
  td3.forward(obs, action, reward, new_obs, done)
env.close()
