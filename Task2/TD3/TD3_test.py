import gym
import torch
import torch.nn as nn
import numpy as np
import moviepy.editor as mp
import os

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = torch.relu(self.l1(torch.cat([state, action], 1)))
        q = torch.relu(self.l2(q))
        return self.l3(q)

class TD3:
    def __init__(self, state_dim, action_dim, max_action, device, params):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=params['lr'])

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer = torch.optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=params['lr'])

        self.max_action = max_action
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.policy_noise = params['policy_noise']
        self.noise_clip = params['noise_clip']
        self.policy_freq = params['policy_freq']
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

params = {
    'lr': 1e-4,
    'gamma': 0.99,
    'tau': 0.005,
    'policy_noise': 0.2,
    'noise_clip': 0.5,
    'policy_freq': 2,
}

env_name = "BipedalWalkerHardcore-v3"
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机种子以避免错误
env.seed(0)

td3 = TD3(state_dim, action_dim, max_action, device, params)

# 加载训练好的模型
checkpoint_dir = 'checkpoints'
checkpoint_actor = 'td3_actor_6290000.pth'
checkpoint_critic_1 = 'td3_critic1_6290000.pth'
checkpoint_critic_2 = 'td3_critic2_6290000.pth'

td3.actor.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint_actor)))
td3.critic_1.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint_critic_1)))
td3.critic_2.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint_critic_2)))

def test_agent(env, td3, episodes=1):
    video_path = 'test_td3.mp4'
    total_reward = 0
    frames = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            frame = env.render(mode='rgb_array')
            frames.append(frame)
            
            action = td3.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            state = next_state

            if done:
                print(f"Episode {episode + 1}: Reward: {episode_reward}")
                total_reward += episode_reward
                break

    avg_reward = total_reward / episodes
    print(f"Average Reward over {episodes} episodes: {avg_reward}")

    clip = mp.ImageSequenceClip(frames, fps=60)
    clip.write_videofile(video_path, codec="libx264")

    return avg_reward

average_reward = test_agent(env, td3)
print(f"Test complete. Average Reward: {average_reward}")

env.close()
