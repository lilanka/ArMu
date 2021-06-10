import copy

import torch

def deepcopy(target, source):
  for target_param, param in zip(target, source):
    target_param.data.copy_(param.data)

def to_tensor(np_array, device):
  return torch.from_numpy(np_array).float().to(device)
