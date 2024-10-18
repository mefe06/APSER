import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Soft update for target network
def soft_update(target: nn.Module, source: nn.Module, tau: float)-> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)