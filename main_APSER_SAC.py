import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from models.SAC import SAC
from models.APSER import APSER, PrioritizedReplayBuffer, ExperienceReplayBuffer
from models.utils import soft_update
from utils import evaluate_policy
import gymnasium as gym

# Hyperparameters
env_name = "LunarLanderContinuous-v3"
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
max_steps_before_truncation = env.spec.max_episode_steps
device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
buffer_size = int(1e6)
batch_size = 256
eval_freq = int(5e3)
max_steps = int(1e6)
discount = 0.99
tau = 0.005  # Soft update parameter
ro = 0.9  # Decay factor for updating nearby transitions
alpha = 0.6  # Prioritization exponent
beta = 0.4  # Importance sampling exponent
learning_starts = 25000  # Start learning after 1000 timesteps
start_time_steps = 25000
policy_noise = 0.2  # Noise added to target policy during critic update
noise_clip = 0.5  # Range to clip target policy noise
policy_freq = 1  # Delayed policy updates
exploration_noise = 0.1  # Noise added to actions for exploration
alpha = 0.2
lr = 0.0003
gamma = 0.99
policy_type = "Gaussian"
target_update_interval = 1
automatic_entropy_tuning = True
hidden_size = 256
kwargs = {
"num_inputs": env.observation_space.shape[0],
"action_space": env.action_space,
"gamma": gamma, 
"tau":tau , 
"alpha":alpha, 
"policy_type": policy_type, 
"target_update_interval": target_update_interval,
"automatic_entropy_tuning": automatic_entropy_tuning, 
"hidden_size":hidden_size, 
"lr": lr, 
"device": device
}
use_APSER = False
# Initialize replay buffer and other variables
if use_APSER:
    replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha)
else:
    replay_buffer = ExperienceReplayBuffer(state_dim, action_dim, buffer_size, device)
previous_scores = deque(maxlen=buffer_size)
evaluations = []
file_name = "LL_exp_1"
# Loss function for critic
mse_loss = nn.MSELoss()
agent = SAC(**kwargs)
# Simulated environment interaction
done = True
actor_losses = []
total_it = 0
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
    if use_APSER:
        replay_buffer.add(transition, initial_score)
    else:
        replay_buffer.add(*transition) 
    previous_scores.append(initial_score)
    state = next_state
    # Do not sample from buffer until learning starts
    if t > learning_starts:# and len(replay_buffer.buffer) > batch_size:
        # Sample from replay buffer
        if use_APSER:
            states, actions, next_states, rewards, not_dones = APSER(replay_buffer, agent, batch_size, beta, discount, ro, max_steps_before_truncation, update_neigbors=True)
        else:
            states, actions, next_states, rewards, not_dones = replay_buffer.sample(batch_size)
        ### Update networks
        total_it += 1
        with torch.no_grad():
            # Select the target smoothing regularized action according to policy
            next_state_action, next_state_log_pi, _ = agent.actor.sample(next_states)

            # Compute the target Q-value
            qf1_next_target, qf2_next_target = agent.critic_target(next_states, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - agent.alpha * next_state_log_pi
            next_q_value = rewards + not_dones * agent.gamma * min_qf_next_target

        # Get the current Q-value estimates
        qf1, qf2 = agent.critic(states, actions)

        # Compute the critic loss
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # Optimize the critic
        agent.critic_optimizer.zero_grad()
        qf_loss.backward()
        agent.critic_optimizer.step()

        # Compute policy loss
        pi, log_pi, _ = agent.actor.sample(states)

        qf1_pi, qf2_pi = agent.critic(states, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((agent.alpha * log_pi) - min_qf_pi).mean()

        # Optimize the actor
        agent.actor_optimizer.zero_grad()
        policy_loss.backward()
        agent.actor_optimizer.step()

        # Tune the temperature coefficient
        if agent.automatic_entropy_tuning:
            alpha_loss = -(agent.log_alpha * (log_pi + agent.target_entropy).detach()).mean()

            agent.alpha_optim.zero_grad()
            alpha_loss.backward()
            agent.alpha_optim.step()

            agent.alpha = agent.log_alpha.exp()

        # Soft update the target critic network
        if total_it % agent.target_update_interval == 0:
            soft_update(agent.critic_target, agent.critic, agent.tau)

        # Evaluate the agent over a number of episodes
        if (t + 1) % eval_freq == 0:
            evaluations.append(evaluate_policy(agent, env_name))
            np.save(f"results/SAC_{file_name}_{t}", evaluations)