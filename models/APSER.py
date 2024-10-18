import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import gymnasium as gym

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
        priorities = priorities[:-1] + 1e-4  # Avoid division by zero
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer)-1, batch_size, p=probabilities) # Ignore the last transition to avoid error on s'
        transitions = [self.buffer[idx] for idx in indices]
        return transitions, indices, probabilities[indices]

    def update_priorities(self, indices: list[int], priorities:list[float]):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

def APSER(replay_buffer: PrioritizedReplayBuffer, agent, batch_size, beta, discount, ro, max_steps_before_truncation: int, bootstrap_steps=1, env=None, update_neigbors= True):
    transitions, indices, probabilities = replay_buffer.sample(batch_size, beta)

    states, actions, next_states, rewards, not_dones = zip(*transitions)
    states = torch.FloatTensor(np.array(states))
    actions = torch.FloatTensor(np.array(actions))
    next_states = torch.FloatTensor(np.array(next_states))
    rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
    not_dones = torch.FloatTensor(np.array(not_dones)).unsqueeze(1)
    
    # Calculate predicted actions in batch
    predicted_actions = agent.actor(states).detach()

    # Extract next actions from the replay buffer
    next_actions = torch.FloatTensor(np.array([replay_buffer.buffer[indices[i]+1][1] for i in range(batch_size)]))

    # For bootstrap steps = 1
    if bootstrap_steps == 1:
        # Vectorized calculation of previous scores
        previous_scores_with_current_critic = rewards + discount * agent.critic.Q1(next_states, next_actions).detach()

        # Vectorized calculation of current scores
        current_scores_with_current_critic = agent.critic.Q1(states, predicted_actions).detach()
    else:
        # For bootstrap steps > 1, we need to account for multiple steps (truncating after n steps)
        discounted_rewards = torch.zeros(batch_size, 1)
        for step in range(bootstrap_steps):
            predicted_actions = agent.actor(next_states).detach()
            next_states, rewards, dones, _ = env.step(predicted_actions)
            discounted_rewards += discount ** step * rewards
            if dones.any():  # Break if any episodes are done
                break
        
        # Calculate current score with critic for multi-step bootstrapping
        current_scores_with_current_critic = discounted_rewards + agent.critic_target(next_states, agent.actor(next_states)).detach()

    # Calculate improvement and priority for batch
    improvements = -(current_scores_with_current_critic - previous_scores_with_current_critic)
    priorities = torch.sigmoid(improvements).T.detach().numpy()[0]

    # Update priorities in replay buffer in batch
    replay_buffer.update_priorities(indices, priorities)
    if update_neigbors:
        # Update scores and priorities of neighboring transitions
        root = 1 # 0.5
        nb_neighbors_to_update = (priorities * max_steps_before_truncation ** root).astype(int)
        for i, nb_neighbors in enumerate(nb_neighbors_to_update):
            neighbors_range = range(1, nb_neighbors // 2 + 1)
            for n_step in neighbors_range:
                if indices[i] - n_step >= 0:
                    replay_buffer.priorities[indices[i] - n_step] += priorities[i] * ro ** n_step
                if indices[i] + n_step < len(replay_buffer.priorities):
                    replay_buffer.priorities[indices[i] + n_step] += priorities[i] * ro ** n_step
    
    return states, actions, next_states, rewards, not_dones

class ExperienceReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device=None):
        self.device = device

        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )