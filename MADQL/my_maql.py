import numpy as np
import random
import matplotlib.pyplot as plt
from my_madql import *
from constant_muav import *


# Define state and action spaces
num_states = (rcell_max + 1) * (rcell_max + 1) * (N + 1) * 100  # Example state space size
num_actions = 4 + K + 5  # 4 directions, K channels, 5 power levels

# Initialize Q-tables
Q_tables = [np.zeros((num_states, num_actions)) for _ in range(M)]

# Initialize storage for convergence curves
episode_rewards = []
drone_episode_rewards = [[] for _ in range(M)]


def state_to_index(state):
    x, y, num_users, effectiveness = state
    x = int(x)
    y = int(y)
    num_users = int(num_users)
    effectiveness = int(effectiveness)

    return (x * (rcell_max + 1) * (N + 1) * 100 + y * (N + 1) * 100 + num_users * 100 + effectiveness)


def q_learning():

    global episode_rewards
    global drone_episode_rewards
    episode_rewards = []
    drone_episode_rewards = [[] for _ in range(M)]

    for episode in range(max_episodes):
        states, user_cord = initialize_states()
        episode_reward = np.zeros(M)  # Total rewards for each drone in this episode

        drones_tmp = np.zeros((M, 2))
        drones_power = np.zeros((M, 1))
        channel_matrix = np.zeros((M, K))
        act_tmp = []

        for t in range(max_steps):
            for m in range(M):
                state = states[m]
                drones_tmp[m] = np.array([state[0], state[1]], dtype=int)
                state_index = state_to_index(state)

                if random.random() < epsilon:
                    action = random.randint(0, num_actions - 1)
                else:
                    action = np.argmax(Q_tables[m][state_index, :])

                # new_state, reward = take_action(state, action)
                new_state, reward, new_act = take_action(state, action, channel_matrix[m].argmax(), drones_power[m])

                drones_tmp[m, :] = np.array([new_state[0], new_state[1]])
                drones_power[m] = new_act[3]
                channel_matrix[m, :] = 0
                channel_matrix[m, new_act[2]] = 1
                act_tmp.append(action)

            new_state, reward = take_action_rand(drones_tmp, drones_power, channel_matrix, user_cord)
            done = 0  # Assuming done = 1 for every step, adjust as needed

            # Store the experience in replay buffer
            for m in range(M):
                replay_buffers[m].append((states[m], act_tmp[m], reward[m], new_state[m], done))

                # Update Q-networks
                update_networks()
                # Update state
                states[m] = new_state[m]
                episode_reward[m] += reward[m]

                new_state_index = state_to_index(new_state[m])

                best_next_action = np.max(Q_tables[m][new_state_index, :])
                Q_tables[m][state_index, action] = Q_tables[m][state_index, action] + \
                                                   alpha * (reward[m] + gamma * best_next_action - Q_tables[m][
                    state_index, action])

                states[m] = new_state[m]

                # Record rewards
                episode_reward[m] += reward[m]

        episode_rewards.append(np.sum(episode_reward))  # Overall reward for the episode
        for m in range(M):
            drone_episode_rewards[m].append(episode_reward[m])

        if episode % 10 == 0:
            print(f"Episode {episode}/{max_episodes}, Total Reward: {np.sum(episode_reward)}")


def plot_convergence():
    # Plot overall convergence
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(episode_rewards)), episode_rewards, label='Overall Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Convergence Curve - Overall')
    plt.legend()
    plt.grid(True)

    # Plot individual drones' convergence
    plt.figure(figsize=(12, 6))
    for m in range(M):
        plt.plot(range(len(drone_episode_rewards[m])), drone_episode_rewards[m], label=f'Drone {m + 1}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Convergence Curve - Each Drone')
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    q_learning()
    plot_convergence()
