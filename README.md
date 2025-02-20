# AU3323-BipedalWalker-v3
Final Project: Deep Reinforcement Learning. AU3323: Artifical Intelligence. May 7, 2024

# 环境介绍

BipedalWalker-v3 是 OpenAI Gym 提供的一个经典控制环境，用于训练和测试强化学习算法。该环境模拟一个双足机器人在不同地形上的行走任务，旨在挑战算法在复杂环境中的控制能力。

- 状态空间

  包含 24 个连续变量，描述机器人的关节角度、速度、脚的位置和地形等信息。
- 动作空间

  4 维连续空间，对应左右大腿和小腿的关节扭矩，范围在 [-1, 1] 之间。
- 奖励函数

   基于前进距离获得奖励。
  
   包含对能量消耗（扭矩）的惩罚和摔倒惩罚。
- 结束条件-

  机器人摔倒或步数超过 2000 步任务结束。
- 训练目标
  学习机器人平稳行走，最大化累计奖励。
  处理平衡控制、能量效率和地形变化等挑战
  
BipedalWalker-v3 是强化学习测试的经典平台，有助于探索和验证不同算法的性能和鲁棒性。

# 依赖安装

Pytorch安装： 请参照Pytorch官网的教程安装Pytorch。如果你正在使用Nvidia显卡，推荐安装CUDA版本的Pytorch。Gym安装: 我们将通过Pip安装gym。打开一个terminal/Anaconda Prompt/powershell。运行以下命令
```
pip install gym==0.26.2
pip install pygame
```

Others: 本次项目中，我们通过tensorboard以及tqdm监控指标的变化以及训练的进度。打开一个terminal/Anaconda Prompt/powershell。运行以下命令

```
pip install tensorboardx
pip install tqdm
pip install rich
```

# 任务 1：Deep Deterministic Policy Gradient

DDPG（Deep Deterministic Policy Gradient）算法是一种无模型的off-policy强化学习问题的算法。该算法非常适用于连续的动作空间。它结合了策略梯度算法和深度Q网络算法的优点，以解决连续动作空间中强化学习算法所面临的挑战。你可以参考论文Continuous control with deep reinforcement learning对算法进行详细学习。

![56ca7311-3dc9-4f4b-b37e-fe3b49cf70b3](https://github.com/user-attachments/assets/bcebb30c-4cb5-4ec2-aea5-2b7312c1c4e6)

# 任务 2：提升Agent性能

由DDPG算法训练的Agent在BipedalWalker这类简单环境中可能表现尚佳,因此，在任务 2中，你可以自由对Agent进行改进，以尝试通关‘BipedalWalkerHardcore’任务。Hardcore模式下的测试脚
本已经给出。

## 尝试使用DDPG算法通关Task 2

实验结果
<video src="https://github.com/YuyaoFan/AU3323-BipedalWalker-v3/blob/master/Videos/Task2_DDPG.mp4.mp4" controls="controls" width="500" height="300">您的浏览器不支持播放该视频！</video>


## 使用TD3算法通关Task 2

实验结果
<video src="https://github.com/YuyaoFan/AU3323-BipedalWalker-v3/blob/master/Videos/Task2_TD3.mp4.mp4" controls="controls" width="500" height="300">您的浏览器不支持播放该视频！</video>





