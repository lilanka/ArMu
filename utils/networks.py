import torch
import torch.nn as nn
import torch.nn.functional as F

# Actor ----------------------------------------------------
class Actor(nn.Module):
  """
  Args:
    OBS_DIM - state space dimension
    ACT_DIM - action space dimension
  """
  def __init__(self, OBS_DIM, ACT_DIM, bias=True):
    super(Actor, self).__init__()

    self.l1 = nn.Linear(OBS_DIM, 400, bias=bias)
    self.l2 = nn.Linear(400, 300, bias=bias)
    self.out = nn.Linear(300, ACT_DIM, bias=bias)
    
  def forward(self, _input):
    x = self.l1(_input)
    x = F.relu(x)
    x = self.l2(x)
    x = F.relu(x)
    x = self.out(x)
    x = torch.tanh(x)
    return x

# Critic ---------------------------------------------------
class Critic(nn.Module):
  """
  Args:
    OBS_DIM - state space dimension
    ACT_DIM - action space dimension
  """
  def __init__(self, OBS_DIM, ACT_DIM, bias=True):
    super(Critic, self).__init__()

    self.l1 = nn.Linear(OBS_DIM + ACT_DIM, 400, bias=bias)
    self.l2 = nn.Linear(400, 300, bias=bias)
    self.out = nn.Linear(300, 1, bias=bias)

  def forward(self, state, action):
    """
    Args: 
      shape (x, y)
    """
    _input = torch.cat((state, action), 0)
    x = self.l1(_input)
    x = F.relu(x)
    x = self.l2(x)
    x = F.relu(x)
    x = self.out(x)
    x = torch.tanh(x)
    return x
