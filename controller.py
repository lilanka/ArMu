from utils.networks import Actor, Critic
from utils.memory import Memory
from utils.utils import *

import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Controller:
  def __init__(self, OBS_DIM, ACT_DIM, MEMORY_SIZE, BATCH_SIZE):
    
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

  def forward(self, obs):
    state = to_tensor(obs['observation'], device)
    achieved_goal = to_tensor(obs['achieved_goal'], device)
    desired_goal = to_tensor(obs['desired_goal'], device)

    action = self.actor.forward(state)
    return action 

  def learn(self, samples):
    pass
