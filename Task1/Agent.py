import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Model import Actor, Critic
from collections import deque
import random

torch.manual_seed(53510713690200)

class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        
    def __len__(self):
        return len(self.buffer)
        
    def push(self, state, next_state, action, reward, done):
        self.buffer.append((state, next_state, action, reward, done))
        
    def sample(self, batch_size, device):
        sample = random.sample(range(len(self.buffer)), batch_size)
        state = torch.from_numpy(np.vstack([self.buffer[i][0] for i in sample])).float().to(device)
        next_state = torch.from_numpy(np.vstack([self.buffer[i][1] for i in sample])).float().to(device)
        action = torch.from_numpy(np.vstack([self.buffer[i][2] for i in sample])).float().to(device)
        reward = torch.from_numpy(np.vstack([self.buffer[i][3] for i in sample])).float().to(device)
        return state, next_state, action, reward
    
class DDPGAgent():
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, buffer_size, batch_size, gamma, tau):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt =  torch.optim.Adam(self.actor.parameters(), actor_lr)
            
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device) 
        self.critic_target.load_state_dict(self.critic.state_dict())       
        self.critic_opt =  torch.optim.Adam(self.critic.parameters(), critic_lr)

        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        self.gamma = gamma
        self.tau = tau

    def get_action(self, state, if_train=1):
        state = torch.from_numpy(state).float().to(self.device)
        # start of your code
        # This method returns actions the agent output during the training process 
        action = self.actor(state).cpu().data.numpy()
    
        # end of your code
        return action
    
    def update(self, state, next_state, action, reward, done):
        self.buffer.push(state, next_state, action, reward, done)
        if len(self.buffer) < self.batch_size:
            return None, None
        state, next_state, action, reward = self.buffer.sample(self.batch_size, self.device)
        # start of your code
        # You need to compute actor_loss/critic_loss and implement the backpropagation in this block
        # function you may need: self.actor.zero_grad(), self.actor_opt.step(), actor_loss.backward()
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1-done) * self.gamma * target_Q
        current_Q = self.critic(state, action)
        critic_loss = torch.nn.functional.mse_loss(current_Q, target_Q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()




        
        # end of your code
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau*param + (1-self.tau)*target_param)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau*param + (1-self.tau)*target_param)
        return critic_loss, actor_loss 
    
    def save(self):
        torch.save(self.actor.state_dict(), 'checkpoints/actor.pth')
        torch.save(self.critic.state_dict(), 'checkpoints/critic.pth')
        torch.save(self.actor_target.state_dict(), 'checkpoints/actor_target.pth')
        torch.save(self.critic_target.state_dict(), 'checkpoints/critic_target.pth')

    def load(self, actor, critic, actor_target, critic_target):
        self.actor.load_state_dict(torch.load(actor,map_location="cpu"))
        self.critic.load_state_dict(torch.load(critic,map_location="cpu"))
        self.actor_target.load_state_dict(torch.load(actor_target,map_location="cpu"))
        self.critic_target.load_state_dict(torch.load(critic_target,map_location="cpu"))