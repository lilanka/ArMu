import torch
import numpy as np
from torch import optim

from utils.networks import Actor, Critic
from utils.memory import Memory
from utils.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Controller:
  def __init__(self, OBS_DIM, ACT_DIM, MEMORY_SIZE, BATCH_SIZE, mean, _std, tau, c, lr, d):
    
    self.act_dim = ACT_DIM
    self.mean = mean
    self.std = _std[0]
    self.std_bar = _std[1]      # ~σ
    self.batch_size= BATCH_SIZE 
    self.tau = tau              
    self.c = c                  # clip min, max values 
    self.d = d

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
  
    # optimizers
    self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
    self.critic_optim = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)

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
      # ~a←πφ′(s′) + ∼clip(N(0,~σ),−c,c)
      action = self.t_actor.forward(mini_batch[3][i]) + torch.clamp(torch.normal(mean=self.mean, std=self.std_bar, size=self.t_actor.out.bias.shape), min=self.c[0], max=self.c[1])
      
      t_Q1 = self.t_critic1.forward(mini_batch[3][i], action)
      t_Q2 = self.t_critic2.forward(mini_batch[3][i], action)

      exp_reward = torch.min(t_Q1, t_Q2)
      target = mini_batch[2][i] + self.tau * exp_reward

      # update Critic functions
      critic_loss = (self.critic1.forward(mini_batch[0][i], mini_batch[1][i]) - target).pow(2).mean() + \
             (self.critic2.forward(mini_batch[0][i], mini_batch[1][i]) - target).pow(2).mean()

      self.critic_optim.zero_grad()
      critic_loss.backward()
      self.critic_optim.step()

      if i % self.d == 0:
        # update the policy 
        actor_loss = -self.critic1(mini_batch[0][i], self.actor(mini_batch[0][i])).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # soft-update the networks
        softcopy(self.t_critic1.parameters(), self.critic1.parameters(), self.tau)
        softcopy(self.t_critic2.parameters(), self.critic2.parameters(), self.tau)
        softcopy(self.t_actor.parameters(), self.actor.parameters(), self.tau)

  def save(self):
    # save the model
    torch.save(self.critic1.state_dict(), "modeldata/_critic1")
    torch.save(self.critic2.state_dict(), "modeldata/_critic2")
    torch.save(self.critic_optim.state_dict(), "modeldata/_critic_optim")

    torch.save(self.actor.state_dict(), "modeldata/_actor")
    torch.save(self.actor_optim.state_dict(), "modeldata/_actor_optim")

  def load(self):
    # load the model
    self.critic1.load_state_dict(torch.load("modeldata/_critic1"))
    self.critic2.load_state_dict(torch.load("modeldata/_critic2"))
    self.critic_optim.load_state_dict(torch.load("modeldata/_critic_optim"))
    deepcopy(self.t_critic1.parameters(), self.critic1.parameters())
    deepcopy(self.t_critic2.parameters(), self.critic2.parameters())

    self.actor.load_state_dict(torch.load("modeldata/_actor"))
    self.actor_optim.load_state_dict(torch.load("modeldata/_actor_optim"))
    deepcopy(self.t_actor.parameters(), self.actor.parameters())
