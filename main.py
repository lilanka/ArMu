#!/bin/python3

import gym
import numpy as np

from controller import Controller

env = gym.make('FetchPickAndPlace-v1')
obs = env.reset()

OBS_DIM = obs['observation'].shape[0]
ACT_DIM = 4  
EPISODES = 1000
BUFFER_SIZE = 10000
BATCH_SIZE = 100

td3 = Controller(OBS_DIM, ACT_DIM, BUFFER_SIZE, BATCH_SIZE)

print(env.action_space.sample().shape)
for _ in range(EPISODES):
  env.render()
  action = td3.forward(obs)
  #obs = env.step(np.array(np.random.rand(4)))
env.close()
