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

<img src="https://github.com/user-attachments/assets/bcebb30c-4cb5-4ec2-aea5-2b7312c1c4e6" width="50%" height="auto" style="display:inline-block;">

# 任务 2：提升Agent性能

由DDPG算法训练的Agent在BipedalWalker这类简单环境中可能表现尚佳,因此，在任务 2中，你可以自由对Agent进行改进，以尝试通关‘BipedalWalkerHardcore’任务。Hardcore模式下的测试脚
本已经给出。

## 尝试使用DDPG算法通关Task 2

实验结果

<img src="https://github.com/user-attachments/assets/4a9bd239-90ad-49b4-bd09-e7665e514460" width="50%" height="auto" style="display:inline-block;">
<img src="https://github.com/user-attachments/assets/a3d92bc5-05b6-40cd-8d7e-f563e5ba474f" width="50%" height="auto" style="display:inline-block;">

最终没有完成任务

<img src="https://github.com/user-attachments/assets/eff002b2-6d04-4f8e-a79c-6aa1a90ecdbf" width="60%" height="auto" style="display:inline-block;">

原因分析：DDPG（Deep Deterministic Policy Gradient）算法存在对Q值高估计的风险，主要原因包括以下几点：

- 函数逼近误差：
DDPG使用神经网络作为函数逼近器来估计Q值。这些神经网络在训练过程中不可避免地会出现误差。由于Critic网络要学习估计复杂的Q函数，训练过程中的误差和偏差可能导致Q值的高估计。

- 过度乐观估计：
由于Critic网络在更新时依赖于Actor网络生成的动作，Actor网络输出的动作可能会在特定状态下导致Q值过高。因为Critic网络在训练过程中试图最大化Q值，如果Actor网络输出的动作本身有高估的偏差，Critic网络会倾向于进一步放大这种高估。

- 目标网络延迟更新：
虽然目标网络通过软更新机制来缓解目标值的变化，但这种延迟更新并不能完全消除高估的风险。目标网络的参数在每次更新时只进行部分更新，而不是立即反映当前最新的网络参数，这可能导致高估值累积。

- 动作空间的探索与利用：
在训练过程中，探索策略可能会选择高估Q值的动作。如果这种高估没有得到及时校正，会进一步影响Critic网络的学习。

## 使用TD3算法通关Task 2

TD3（Twin Delayed Deep Deterministic Policy Gradient）是一种改进的连续动作空间强化学习算法，它是在 DDPG 的基础上提出的，以解决 DDPG 在高维、复杂环境中存在的稳定性和过估计问题。

TD3通过一下三个关键改进解决DDPG中存在的一些问题：

1.目标策略噪声

2.双Q网络

3.延迟策略更新

<img src="https://github.com/user-attachments/assets/eec53fd4-ea34-45d0-969e-a50e3bdb4e79" width="60%" height="auto" style="display:inline-block;">

训练结果

<img src="https://github.com/user-attachments/assets/6d975972-c05b-4134-ac4d-c4a23b8b65e8" width="60%" height="auto" style="display:inline-block;">

与DDPG训练结果相比，TD3训练结果有以下优势：

- 训练速度更快

- 收敛速度更快（从趋势上判断）

- 网络更稳定

但是在训练过程中仍出现以下问题：

- 训练不能快速收敛

- 训练有效性低

需要使用一些技巧改进训练过程

这里参考文章https://zhuanlan.zhihu.com/p/409553262?utm_medium=social&utm_psn=1779488358701985792 中的技巧对训练进行改进

<img src="https://github.com/user-attachments/assets/0d548678-74ba-4012-988d-f46d816267ed" width="60%" height="auto" style="display:inline-block;">

改进后的训练效果显著提升

<img src="https://github.com/user-attachments/assets/32cb6af0-7faa-476b-867c-a1ddffea8011" width="60%" height="auto" style="display:inline-block;">

最终得分311.11分通过测试

# 提交文件说明

Videos文件记录了Task1和Task2测试结果视频

Task1文件包含Checkpoints、run、即DDPG实现的agent、test、train、Model文件

Task2包含DDPG和TD3文件夹，文件夹中为相应算法的训练数据和代码实现












