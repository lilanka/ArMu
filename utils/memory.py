import random
import numpy as np
from collections import namedtuple, deque

import torch

from utils.utils import *

class Memory:
  """
  Args:
    buffer_size = size of the buffer
    batch_size = how many random samples should pull
    seed = random seed
  """
  def __init__(self, buffer_size, batch_size):
    self.buffer_size = buffer_size
    self.batch_size = batch_size
  
    self.memory = deque(maxlen=buffer_size)
    self.experience = namedtuple("Experience", \
        field_names=["state", "action", "reward", "new_state", "done"])

  def push(self, state, action, reward, new_state, done):
    sample = self.experience(state, action, reward, new_state, done)
    self.memory.append(sample)

  def pull(self, device):
    samples = random.sample(self.memory, k=self.batch_size)
   
    # TODO : use torch.cat. it is faster
    #            torch.stack is good. but not faster as cat

    states = [obs_concat(s.state, device) for s in samples]
    actions = [to_tensor(s.action, device) for s in samples]
    rewards = [torch.tensor([s.reward]) for s in samples]
    new_states = [obs_concat(s.new_state, device) for s in samples]
    dones = [torch.tensor([s.done]) for s in samples]

    return (states, actions, rewards, new_states, dones)

  def __len__(self):
    return len(self.memory)
