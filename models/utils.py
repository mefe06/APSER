import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Soft update for target network
def soft_update(target: nn.Module, source: nn.Module, tau: float)-> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# Exactly match the behavioral policy parameters
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# Initialize network weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
