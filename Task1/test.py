from Agent import DDPGAgent
import gym
from matplotlib import pyplot as plt
from matplotlib import animation

env = gym.make('BipedalWalker-v3', render_mode='human')
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
BUFFER_SIZE = 1000000
BATCH_SIZE = 128
GAMMA = 0.99
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
TAU = 1e-4

agent = DDPGAgent(STATE_DIM, ACTION_DIM, ACTOR_LR, CRITIC_LR, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU)
agent.load('Checkpoints/actor.pth','Checkpoints/critic.pth','Checkpoints/actor_target.pth','Checkpoints/critic_target.pth')

score = 0
frames = []

state, _ = env.reset(seed=0)
while True:
    frame = env.render()
    action = agent.get_action(state,0)
    next_state, reward, done, info, _ = env.step(action)
    state = next_state
    score += reward
    frames.append(frame)
    if done:
        break
print(f'The score of this agent is {score:.2f}.')
env.close()
