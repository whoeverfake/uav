import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # 将输出限制在-1到1之间


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class MADQL:
    def __init__(self, num_agents, state_dim, action_dim, gamma=0.9, lr=0.001, batch_size=64, memory_size=10000, tau=0.01):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.tau = tau

        self.actors = [Actor(state_dim, action_dim) for _ in range(num_agents)]
        self.target_actors = [Actor(state_dim, action_dim) for _ in range(num_agents)]
        self.critics = [Critic(state_dim, action_dim) for _ in range(num_agents)]
        self.target_critics = [Critic(state_dim, action_dim) for _ in range(num_agents)]
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=lr) for critic in self.critics]

        for target_actor, actor in zip(self.target_actors, self.actors):
            target_actor.load_state_dict(actor.state_dict())
        for target_critic, critic in zip(self.target_critics, self.critics):
            target_critic.load_state_dict(critic.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, epsilon=0.1):
        actions = []
        for i, actor in enumerate(self.actors):
            state_tensor = torch.FloatTensor(state[i])
            action = actor(state_tensor).cpu().detach().numpy()
            noise = np.random.normal(0, epsilon, size=self.action_dim)
            action = np.clip(action + noise, -1, 1)
            actions.append(action.tolist())
        return actions

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for i in range(self.num_agents):
            state, action, reward, next_state, done = zip(*minibatch)
            state = torch.FloatTensor([s[i] for s in state])
            action = torch.FloatTensor([a[i] for a in action])
            reward = torch.FloatTensor([r[i] for r in reward]).unsqueeze(1)
            next_state = torch.FloatTensor([ns[i] for ns in next_state])
            done = torch.FloatTensor(done).unsqueeze(1)

            # Critic update
            next_action = self.target_actors[i](next_state)
            next_q_value = self.target_critics[i](next_state, next_action)
            target_q_value = reward + (1 - done) * self.gamma * next_q_value
            q_value = self.critics[i](state, action)

            critic_loss = nn.MSELoss()(q_value, target_q_value.detach())
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            # Actor update
            predicted_action = self.actors[i](state)
            actor_loss = -self.critics[i](state, predicted_action).mean()
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            # Soft update target networks
            self.soft_update(self.target_critics[i], self.critics[i])
            self.soft_update(self.target_actors[i], self.actors[i])

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def update_target_network(self):
        for target_actor, actor in zip(self.target_actors, self.actors):
            target_actor.load_state_dict(actor.state_dict())
        for target_critic, critic in zip(self.target_critics, self.critics):
            target_critic.load_state_dict(critic.state_dict())
