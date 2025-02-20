import os
import gym
import torch
import numpy as np
import random
from torch.optim import Adam
from tensorboardX import SummaryWriter
from tqdm import tqdm

class TD3:
    def __init__(self, state_dim, action_dim, max_action, device, params):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = Adam(self.actor.parameters(), lr=params['actor_lr'])

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = Adam(self.critic_1.parameters(), lr=params['critic_lr'])

        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = Adam(self.critic_2.parameters(), lr=params['critic_lr'])

        self.max_action = max_action
        self.discount = params['gamma']
        self.tau = params['tau']
        self.policy_noise = params['policy_noise']
        self.noise_clip = params['noise_clip']
        self.policy_freq = params['policy_freq']
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        return self.actor(state).cpu().data.numpy()

    def train(self, replay_buffer, batch_size):
        self.total_it += 1

        # Sample a batch of transitions from replay buffer
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)
        critic_loss = torch.nn.functional.mse_loss(current_Q1, target_Q) + torch.nn.functional.mse_loss(current_Q2, target_Q)

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        actor_loss = 0
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss

class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = torch.nn.Linear(state_dim, 400)
        self.l2 = torch.nn.Linear(400, 300)
        self.l3 = torch.nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.nn.functional.relu(self.l1(state))
        a = torch.nn.functional.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = torch.nn.Linear(state_dim + action_dim, 400)
        self.l2 = torch.nn.Linear(400, 300)
        self.l3 = torch.nn.Linear(300, 1)

    def forward(self, state, action):
        q = torch.nn.functional.relu(self.l1(torch.cat([state, action], 1)))
        q = torch.nn.functional.relu(self.l2(q))
        return self.l3(q)

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=10000000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.done[ind]).to(device)
        )


def make_env(env_name):
    env = gym.make(env_name)
    return env

torch.manual_seed(53510713690200)

env_name = "BipedalWalkerHardcore-v3"
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = {
    'actor_lr': 1e-4,
    'critic_lr': 1e-4,
    'gamma': 0.99,
    'tau': 0.005,
    'policy_noise': 0.2*max_action,
    'noise_clip': 0.5*max_action,
    'policy_freq': 2,
    'batch_size': 256,
}
td3 = TD3(state_dim, action_dim, max_action, device, params)
replay_buffer = ReplayBuffer(state_dim, action_dim)


writer = SummaryWriter()


total_steps = 1e7
max_timesteps = 1600
save_interval = 10000
update_interval = 50

state = env.reset()
episode_reward = 0
episode_num = 0

for step in tqdm(range(int(total_steps))):
    action = td3.select_action(state)
    action = (action + np.random.normal(0, max_action * 0.1, size=action_dim)).clip(-max_action, max_action)
    next_state, reward, done, _ = env.step(action)
    #tricks for bipedal walker
    if reward <=-100:
        reward = -1
        replay_buffer.add(state, action, next_state, reward, done=True)
    else:
        replay_buffer.add(state, action, next_state, reward, done=False)
    state = next_state
    episode_reward += reward

    if replay_buffer.size >= params['batch_size']:
        critic_loss, actor_loss = td3.train(replay_buffer, params['batch_size'])
        writer.add_scalar('Loss/Critic', critic_loss, step)
        writer.add_scalar('Loss/Actor', actor_loss, step)

    if done or step % max_timesteps == 0:
        writer.add_scalar('Reward/Episode', episode_reward, episode_num)
        print(f"Episode: {episode_num}, Reward: {episode_reward}")
        state = env.reset()
        episode_reward = 0
        episode_num += 1

    if step % save_interval == 0:
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(td3.actor.state_dict(), os.path.join('checkpoints', f"td3_actor_{step}.pth"))
        torch.save(td3.critic_1.state_dict(), os.path.join('checkpoints', f"td3_critic1_{step}.pth"))
        torch.save(td3.critic_2.state_dict(), os.path.join('checkpoints', f"td3_critic2_{step}.pth"))

writer.close()
env.close()