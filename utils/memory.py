import random
import numpy as np
from collections import namedtuple, deque

import torch

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
    self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "new_state"])

  def push(self, state, action, reward, new_state):
    sample = self.experience(state, action, reward, new_state)
    self.memory.append(sample)

  def pull(self):
    samples = random.sample(self.memory, k=self.batch_size)
 
    for s in samples:
      states = torch.from_numpy(s.state).float()
      actions = torch.from_numpy(s.action).float()
      rewards = torch.from_numpy(s.reward).float()
      new_states = torch.from_numpy(s.new_state).float()

    return (states, actions, rewards, new_states)
