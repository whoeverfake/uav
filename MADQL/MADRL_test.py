import numpy as np
import torch
import os
from MADQL_define import MADQL
from muav_env import MultiAgentEnv

def test_madql(env, madql, num_agents, num_tests=10):
    rewards = []
    phm = []

    for test in range(num_tests):
        state = env.reset()
        done = False
        total_reward = 0

        for i in range(100):
            # 获取每个 agent 的动作
            actions = madql.act(state, epsilon=0)  # epsilon=0，使用确定性策略
            next_state, reward, done, _ = env.step(actions)
            total_reward = sum(reward) + 0.9 * total_reward
            state = next_state

        rewards.append(total_reward)
        print(f"Test {test + 1}/{num_tests}, network utility: {np.sum(env._calculate_reward()).item():.2f}")

        phm.append(np.sum(env.calculate_reward_final()))

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    return rewards, avg_reward, std_reward


env = MultiAgentEnv()
num_agents = env.agent_num

# 加载训练好的模型
def load_model(madql, save_dir, episode, step_count):
    for i, (actor, critic) in enumerate(zip(madql.actors, madql.critics)):
        actor_path = os.path.join(save_dir, f'actor_agent_{i}_episode_{episode}_step_{step_count}.pth')
        critic_path = os.path.join(save_dir, f'critic_agent_{i}_episode_{episode}_step_{step_count}.pth')
        actor.load_state_dict(torch.load(actor_path))
        critic.load_state_dict(torch.load(critic_path))
    print(f'Loaded model for episode {episode}, step {step_count}')

# 测试训练好的模型
madql_agent = MADQL(num_agents, 4, 3)  # 初始化模型
load_model(madql_agent, 'checkpoints_uav8', 999, 1000)  # 使用保存的模型

results = test_madql(env, madql_agent, num_agents)
