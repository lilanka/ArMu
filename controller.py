import torch
import numpy as np

from utils.networks import Actor, Critic
from utils.memory import Memory
from utils.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Controller:
  def __init__(self, OBS_DIM, ACT_DIM, MEMORY_SIZE, BATCH_SIZE, mean, _std, tau, c):
    
    self.act_dim = ACT_DIM
    self.mean = mean
    self.std = _std[0]
    self.std_bar = _std[1]
    self.batch_size= BATCH_SIZE 
    self.tau = tau
    self.c = c 

    # initialize critic networks and actor networks
    self.actor = Actor(OBS_DIM, ACT_DIM)
    self.critic1 = Critic(OBS_DIM, ACT_DIM)
    self.critic2 = Critic(OBS_DIM, ACT_DIM)
    
    # initialize target networks
    self.t_actor = Actor(OBS_DIM, ACT_DIM)
    self.t_critic1 = Critic(OBS_DIM, ACT_DIM)
    self.t_critic2 = Critic(OBS_DIM, ACT_DIM)
   
    # initialize target parameters 
    deepcopy(self.actor.parameters(), self.t_actor.parameters())
    deepcopy(self.critic1.parameters(), self.critic1.parameters())
    deepcopy(self.critic2.parameters(), self.critic2.parameters())
   
    # initialize replay buffer
    self.buffer = Memory(MEMORY_SIZE, BATCH_SIZE)

  def step(self, obs):
    # choose actions and take new observations
    state = obs_concat(obs, device)

    action = self.actor.forward(state)
    # no -c, c for now
    action[:] += torch.normal(mean=self.mean, std=self.std, size=action.shape)
    return to_numpy(action)

  def forward(self, obs, action, reward, new_obs, done):
    self.buffer.push(obs, action, reward, new_obs, done)

    if len(self.buffer) <= self.batch_size:
      return

    mini_batch = self.buffer.pull(device)
    
    for i in range(len(mini_batch[0])):
      action = self.t_actor.forward(mini_batch[3][i])
      # ~a←πφ′(s′) + ∼clip(N(0,~σ),−c,c)
      action[:] = action + torch.clamp(torch.normal(mean=self.mean, std=self.std_bar, size=action.shape), min=self.c[0], max=self.c[1])
      
      rew1 = self.t_critic1(mini_batch[3][i], action)
      rew2 = self.t_critic2(mini_batch[3][i], action)
      exp_reward = min(rew1, rew2)
      target = mini_batch[2][i] + self.tau * exp_reward
       
