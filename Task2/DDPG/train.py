from Agent import DDPGAgent
import gym
import numpy as np
from tensorboardX import SummaryWriter
from tqdm.rich import tqdm
import torch

torch.manual_seed(53510713690200)

writer = SummaryWriter()
env = gym.make('BipedalWalkerHardcore-v3')
# Hyper-parameters to tune
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
BUFFER_SIZE = 1000000
BATCH_SIZE = 128
GAMMA = 0.99
ACTOR_LR = 0.0001
CRITIC_LR = 0.0001
TAU = 0.0001
MAX_EPSISODE = 10000
T = 700

agent = DDPGAgent(STATE_DIM, ACTION_DIM, ACTOR_LR, CRITIC_LR, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU)
count = 0
max_score = -np.inf
try:
    for i in tqdm(range(MAX_EPSISODE)):
        score = 0
        state, _ = env.reset(seed=0)
        for j in range(T):
            action = agent.get_action(state)
            next_state, reward, done, info, _ = env.step(action)
            l1, l2 = agent.update(state, next_state, action, reward, done)
            score += reward
            state = next_state
            if l1 != None and l2 != None:
                writer.add_scalar('loss/critic', l1, count)
                writer.add_scalar('loss/actor', l2, count)
                count += 1
            if done:
                break
        writer.add_scalar('score', score, i+1)
        if score > max_score:
            agent.save()
            max_score = score
        if score > 200:
            break
except KeyboardInterrupt:
    print(f"Keyboard Terimated. Trained for {i+1} epsiodes.")
finally:
    env.close()

