import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(53510713690200)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.tanh(x)
        return x
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256+action_dim, 128)
        self.l3 = nn.Linear(128, 1)
        
    def forward(self, state, action):
        x = state
        x = self.l1(x)
        x = F.relu(x)
        x = torch.cat([x, action], dim=1)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)
        return x