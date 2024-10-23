import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from models.TD3 import TD3
from models.APSER import APSER, PrioritizedReplayBuffer, ExperienceReplayBuffer
from utils import evaluate_policy, save_with_unique_filename
import gymnasium as gym

# Hyperparameters
env_name = "Ant-v5"
env = gym.make(env_name) 
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
max_steps_before_truncation = env.spec.max_episode_steps
device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
buffer_size = int(250000)
batch_size = 256
eval_freq = int(2500)
max_steps = int(250000)
discount = 0.99
tau = 0.005  # Soft update parameter
ro = 0.9  # Decay factor for updating nearby transitions
alpha = 0.6  # Prioritization exponent
beta = 0.4  # Importance sampling exponent
learning_starts = 25000  # Start learning after 1000 timesteps
start_time_steps = 25000
uniform_sampling_period = 0
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
use_APSER = True
use_importance_weights = True
PER = False
agent_type = "TD3"
# Initialize replay buffer and other variables
if use_APSER:
    replay_buffer = PrioritizedReplayBuffer(state_dim, action_dim, buffer_size, device)
else:
    if PER:
        replay_buffer = PrioritizedReplayBuffer(state_dim, action_dim, buffer_size, device)
    else:
        replay_buffer = ExperienceReplayBuffer(state_dim, action_dim, buffer_size, device)
previous_scores = deque(maxlen=buffer_size)
evaluations = []
td_errors = []
file_name = f"Normalize_Relu_APSER_TD3_{env_name}_exp"
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
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    # Store transition in buffer
    transition = [state, action, next_state, reward, done]
    td_error = agent.critic.Q1(torch.FloatTensor(np.array(state)).to(agent.device).unsqueeze(0), torch.FloatTensor(np.array(action)).to(agent.device).unsqueeze(0))  - \
    reward - discount * (1-done)*(agent.critic.Q1(torch.FloatTensor(np.array(next_state)).to(agent.device).unsqueeze(0), 
                                         torch.FloatTensor(agent.select_action(torch.FloatTensor(np.array(next_state)).to(agent.device).unsqueeze(0))).to(agent.device).unsqueeze(0)))
    td_errors.append(td_error.detach().numpy())
    initial_score = [0]  # Initial score for new transitions
    if use_APSER:
        replay_buffer.add(*transition)
    else:
        replay_buffer.add(*transition) 
    previous_scores.append(initial_score)
    state = next_state
    # Do not sample from buffer until learning starts
    if t > learning_starts:# and len(replay_buffer.buffer) > batch_size:
        # Sample from replay buffer
        if use_APSER:
            if t< learning_starts + uniform_sampling_period:
                states, actions, next_states, rewards, not_dones, weights = APSER(replay_buffer, agent, batch_size, beta, discount, ro, max_steps_before_truncation, update_neigbors=False, uniform_sampling=True)
            else:
                states, actions, next_states, rewards, not_dones, weights = APSER(replay_buffer, agent, batch_size, beta, discount, ro, max_steps_before_truncation, update_neigbors=False)
            weights = torch.FloatTensor(weights).to(agent.device)
        else:
            states, actions, next_states, rewards, not_dones = replay_buffer.sample(batch_size)
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
        if use_importance_weights&use_APSER:
            critic_loss = (weights*F.mse_loss(current_Q1, target_Q, reduction='none')).mean() + (weights*F.mse_loss(current_Q2, target_Q, reduction='none')).mean()
        else:
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        # Delayed policy updates, update actor networks every update period
        if agent.total_it % agent.policy_freq == 0:

            # Compute the actor loss
            if use_importance_weights&use_APSER:
                actor_loss = -(weights*agent.critic.Q1(states, agent.actor(states))).mean()
            else:
                actor_loss = -agent.critic.Q1(states, agent.actor(states)).mean()
            actor_losses.append(actor_loss.item())
            # Optimize the actor
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            # Soft update the target networks
            for param, target_param in zip(agent.critic.parameters(), agent.critic_target.parameters()):
                target_param.data.copy_(agent.tau * param.data + (1 - agent.tau) * target_param.data)

            for param, target_param in zip(agent.actor.parameters(), agent.actor_target.parameters()):
                target_param.data.copy_(agent.tau * param.data + (1 - agent.tau) * target_param.data)

        # Evaluate the agent over a number of episodes
        if (t + 1) % eval_freq == 0:
            if use_APSER:
                evaluations.append(evaluate_policy(agent, env_name))
                save_with_unique_filename(evaluations, f"results/{file_name}_{t}")
                save_with_unique_filename( np.array(td_errors), f"results/{file_name}_td_errors_{t}")
                save_with_unique_filename(actor_losses, f"results/{file_name}_actor_losses_{t}")
            else:
                evaluations.append(evaluate_policy(agent, env_name))
                save_with_unique_filename(evaluations, f"results/{file_name}_{t}")
                rewards = np.array(replay_buffer.reward)
                save_with_unique_filename(rewards, f"results/{file_name}_rewards_{t}")
                #td_errors = np.array(td_errors)
                save_with_unique_filename(np.array(td_errors), f"results/{file_name}_td_errors_{t}")

