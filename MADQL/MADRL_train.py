import os
import matplotlib.pyplot as plt
import pandas as pd
from muav_env import MultiAgentEnv
from MADQL_define import  MADQL
import torch
import numpy as np


def train_madql(env, num_agents, state_dim, action_dim, num_episodes=1200, epsilon_start=1.0, epsilon_end=0.1,
                epsilon_decay=0.995, target_update_freq=10, save_freq=20, save_dir='checkpoints_uav62'):
    os.makedirs(save_dir, exist_ok=True)
    madql = MADQL(num_agents, state_dim, action_dim)
    epsilon = epsilon_start

    # 初始化记录列表
    total_rewards = []
    agent_rewards = [[] for _ in range(num_agents)]
    step_count = 0
    reward_max = -1

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        episode_rewards = [0] * num_agents
        max_step = 100

        while not done:
            action = madql.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            madql.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward = sum(reward) + madql.gamma * total_reward
            for i in range(num_agents):
                episode_rewards[i] = reward[i] + episode_rewards[i] * madql.gamma
            madql.replay()
            max_step -= 1
            if max_step <= 0:
                done = True

        step_count += 1
        if step_count % save_freq == 0:
            save_model(madql, save_dir, episode, step_count)

        total_rewards.append(total_reward)
        for i in range(num_agents):
            agent_rewards[i].append(episode_rewards[i])

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if episode % target_update_freq == 0:
            madql.update_target_network()

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {np.array(total_reward).item():.2f}, Epsilon: {epsilon:.2f}")

        # 保存总奖励和每个 agent 的奖励图
        plot_rewards(np.array(total_rewards), np.array(agent_rewards), save_dir)

    # 保存最终的奖励数据到 CSV 文件
    save_rewards(total_rewards, agent_rewards, save_dir)

    return madql


def save_model(madql, save_dir, episode, step_count):
    # 保存mdoel
    for i, (actor, critic) in enumerate(zip(madql.actors, madql.critics)):
        actor_path = os.path.join(save_dir, f'actor_agent_{i}_episode_{episode}_step_{step_count}.pth')
        critic_path = os.path.join(save_dir, f'critic_agent_{i}_episode_{episode}_step_{step_count}.pth')
        torch.save(actor.state_dict(), actor_path)
        torch.save(critic.state_dict(), critic_path)
        print(f'Saved actor and critic for agent {i} at episode {episode}, step {step_count}')


def plot_rewards(total_rewards, agent_rewards, save_dir):
    # huatu
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(total_rewards, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward over Episodes')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for i, rewards in enumerate(agent_rewards):
        plt.plot(rewards, label=f'Agent {i} Reward')
    plt.xlabel('Episode')
    plt.ylabel('Agent Reward')
    plt.title('Agent Rewards over Episodes')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reward_plot.png'))
    plt.close()


def save_rewards(total_rewards, agent_rewards, save_dir):
    # 保存到 CSV 文件
    df = pd.DataFrame({
        'Episode': list(range(len(total_rewards))),
        'Total Reward': total_rewards
    })

    for i, rewards in enumerate(agent_rewards):
        df[f'Agent {i} Reward'] = rewards

    df.to_csv(os.path.join(save_dir, 'rewards.csv'), index=False)
    print('Saved rewards to rewards.csv')


env = MultiAgentEnv()
num_agents = env.agent_num
state_dim = 4 ### 每个agent的状态是四维： UAV 位置、phi、接入用户数
action_dim = 3  # 每个动作包含3个数值

madql_agent = train_madql(env, num_agents, state_dim, action_dim)
