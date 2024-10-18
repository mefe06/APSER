import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from models.TD3 import TD3
from models.APSER import APSER, PrioritizedReplayBuffer
from utils import soft_update, evaluate_policy
import gymnasium as gym

# Hyperparameters
env_name = "LunarLander-v3"
env = gym.make(env_name, continuous=True)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
max_steps_before_truncation = env.spec.max_episode_steps
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
buffer_size = int(1e5)
batch_size = 256
eval_freq = int(5e3)
max_steps = int(1.5e5)
discount = 0.99
tau = 0.005  # Soft update parameter
ro = 0.9  # Decay factor for updating nearby transitions
alpha = 0.6  # Prioritization exponent
beta = 0.4  # Importance sampling exponent
learning_starts = 2000  # Start learning after 1000 timesteps
start_time_steps = 1000
#nb_neighbors_to_update = 5  # Number of neighbors to update when a transition is updated
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

# Initialize replay buffer and other variables
replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha)
previous_scores = deque(maxlen=buffer_size)
evaluations = []
file_name = "LL_exp_1"
# Loss function for critic
mse_loss = nn.MSELoss()
agent = TD3(**kwargs)
# Simulated environment interaction
done = True
actor_losses = []
for t in range(1, max_steps):
    if done:
        state, _ = env.reset()
    if t < start_time_steps:
        action = env.action_space.sample()
    else:
        action = (agent.select_action(np.array(state)) + np.random.normal(0, max_action * exploration_noise, size=action_dim)).clip(-max_action, max_action)
    next_state, reward, done, _, _ = env.step(action)
    # Store transition in buffer
    transition = [state, action, next_state, reward, done]
    initial_score = [0]  # Initial score for new transitions
    replay_buffer.add(transition, initial_score)
    previous_scores.append(initial_score)
    state = next_state
    # Do not sample from buffer until learning starts
    if t > learning_starts and len(replay_buffer.buffer) > batch_size:
        # Sample from replay buffer
        states, actions, next_states, rewards, not_dones = APSER(replay_buffer, agent, batch_size, beta, discount, ro, max_steps_before_truncation)
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
            soft_update(agent.actor_target, agent.actor, tau)
            soft_update(agent.critic_target, agent.critic, tau)

        # Evaluate the agent over a number of episodes
        if (t + 1) % eval_freq == 0:
            evaluations.append(evaluate_policy(agent, env_name))
            np.save(f"results/{file_name}_{t}", evaluations)