import copy
import torch

def deepcopy(target, source):
  for target_param, param in zip(target, source):
    target_param.data.copy_(param.data)

def softcopy(target, source, tau):
  for target_param, param in zip(target, source):
    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def to_tensor(np_array, device):
  return torch.from_numpy(np_array).float().to(device)

def to_numpy(torch_tensor):
  return torch_tensor.cpu().detach().numpy()

def obs_concat(obs, device):
  # concat the observation given from the environment
  _obs = to_tensor(obs['observation'], device)
  ach_goal = to_tensor(obs['achieved_goal'], device)
  des_goal = to_tensor(obs['desired_goal'], device)
  
  state = torch.cat((_obs, ach_goal, des_goal), 0)
  return state
