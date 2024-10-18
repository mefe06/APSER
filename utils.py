import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
# Soft update for target network
def soft_update(target: nn.Module, source: nn.Module, tau: float)-> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# Exactly match the behavioral policy parameters
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# Initialize network weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def evaluate_policy(agent, env_name, eval_episodes=10, show_evals = False):
    eval_env = gym.make(env_name, continuous = True)
    #eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        done = False

        while not done:
            action = agent.select_action(np.array(state))
            if show_evals:
                eval_env.render()
            state, reward, done, _, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")

    return avg_reward