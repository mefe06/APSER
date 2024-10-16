import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from models.TD3 import TD3
import gymnasium as gym
# Define Actor and Critic Networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return torch.tanh(self.fc2(x))  # Assuming continuous action space

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Soft update for target network
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# Replay Buffer with Prioritization
class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, alpha):
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.scores = deque(maxlen=buffer_size)  # Storing scores as (sum of rewards, Q-value)
        self.alpha = alpha

    def add(self, transition, score):
        self.buffer.append(transition)
        max_priority = max(self.priorities, default=1.0)
        self.priorities.append(max_priority)
        self.scores.append(score)

    def sample(self, batch_size, beta):
        priorities = np.array(self.priorities, dtype=np.float32) ** self.alpha
        priorities = priorities[:-1]
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer)-1, batch_size, p=probabilities) # Ignore the last transition to avoid error on s'
        transitions = [self.buffer[idx] for idx in indices]
        return transitions, indices, probabilities[indices]

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

# Hyperparameters
env = gym.make("HalfCheetah-v4")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
buffer_size = 10000
batch_size = 64
discount = 0.99
tau = 0.005  # Soft update parameter
ro = 0.9  # Decay factor for updating nearby transitions
alpha = 0.6  # Prioritization exponent
beta = 0.4  # Importance sampling exponent
learning_starts = 1000  # Start learning after 1000 timesteps
nb_neighbors_to_update = 5  # Number of neighbors to update when a transition is updated
policy_noise = 0.2  # Noise added to target policy during critic update
noise_clip = 0.5  # Range to clip target policy noise
policy_freq = 2  # Delayed policy updates
exploration_noise = 0.1  # Noise added to actions for exploration
kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_action": max_action,
    "discount": discount,
    "tau": tau,
    "policy_noise": policy_noise,
    "noise_clip": noise_clip,
    "policy_freq": policy_freq,
    "device": device
}

# Initialize networks
actor = Actor(state_dim, action_dim)
actor_target = Actor(state_dim, action_dim)
critic = Critic(state_dim, action_dim)
critic_target = Critic(state_dim, action_dim)

# Copy weights to target networks
actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

# Optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

# Initialize replay buffer and other variables
replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha)
previous_scores = deque(maxlen=buffer_size)

# Loss function for critic
mse_loss = nn.MSELoss()
agent = TD3(**kwargs)
start_time_steps = 100
# Simulated environment interaction
state = env.reset()  # Initial state
actor_losses = []
for t in range(1, 1200):
    if t < start_time_steps:
        action = env.action_space.sample()
    else:
        action = (agent.select_action(np.array(state)) + np.random.normal(0, max_action * exploration_noise, size=action_dim)).clip(-max_action, max_action)
    next_state, reward, done, _, _ = env.step(action)
    # Store transition in buffer
    transition = (state, action, next_state, reward, done)
    initial_score = [0]  # Initial score for new transitions
    replay_buffer.add(transition, initial_score)
    previous_scores.append(initial_score)
    state = next_state
    # Do not sample from buffer until learning starts
    if t > learning_starts and len(replay_buffer.buffer) > batch_size:
        # Sample from replay buffer
        transitions, indices, probabilities = replay_buffer.sample(batch_size, beta)

        states, actions, next_states, rewards, not_dones = zip(*transitions)
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        not_dones = torch.FloatTensor(np.array(not_dones)).unsqueeze(1)

        for i in range(batch_size):
            total_reward = 0  # Accumulate rewards for n steps
            predicted_action = agent.actor(states[i].unsqueeze(0)).detach()
            next_state = next_states[i] # get s' from buffer
            next_action = replay_buffer.buffer[indices[i]+1][1] # get a' from buffer
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            next_action = torch.FloatTensor(next_action).unsqueeze(0)
            # Calculate previous score with current critic
            previous_score_with_current_critic = rewards[i].item() + discount * agent.critic_target(next_state, next_action)[0].detach().item()

            # Calculate current score with current critic
            current_score_with_current_critic = agent.critic_target(states[i].unsqueeze(0), predicted_action)[0].detach().item()
            # Calculate improvement
            improvement = current_score_with_current_critic - previous_score_with_current_critic
            priority = np.exp(-improvement)
            replay_buffer.update_priorities([indices[i]], [priority])
            replay_buffer.scores[indices[i]].append(current_score_with_current_critic)
            # Optionally, update nearby transitions' priorities
            for n_step in range(1, nb_neighbors_to_update + 1):
                if indices[i] - n_step >= 0:
                    replay_buffer.priorities[indices[i] - n_step] *= priority * ro ** n_step
                if indices[i] + n_step < len(replay_buffer.priorities):
                    replay_buffer.priorities[indices[i] + n_step] *= priority * ro ** n_step
        ### Update networks
        agent.total_it += 1
        with torch.no_grad():
            # Select action according to the target policy and add target smoothing regularization
            noise = (torch.randn_like(actions) * agent.policy_noise).clamp(-agent.noise_clip, agent.noise_clip)
            next_actions = (agent.actor_target(next_states) + noise).clamp(-agent.max_action, agent.max_action)

            # Compute the target Q-value
            target_Q1, target_Q2 = agent.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + not_dones * agent.discount * target_Q

        # Get the current Q-value estimates
        current_Q1, current_Q2 = agent.critic(states, actions)

        # Compute the critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        # Delayed policy updates, update actor networks every update period
        if agent.total_it % agent.policy_freq == 0:

            # Compute the actor loss
            actor_loss = -agent.critic.Q1(states, agent.actor(states)).mean()
            actor_losses.append(actor_loss.item())
            # Optimize the actor
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            # Update target networks using soft update
            soft_update(actor_target, actor, tau)
            soft_update(critic_target, critic, tau)

print(actor_losses)
