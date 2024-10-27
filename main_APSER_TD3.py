import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from models.TD3 import TD3
from models.APSER import APSER, PrioritizedReplayBuffer, ExperienceReplayBuffer
from utils import evaluate_policy, save_with_unique_filename
import gymnasium as gym
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='TD3 with APSER training script')
    
    # Environment
    parser.add_argument('--env_name', type=str, default="Ant-v5", help='Gymnasium environment name')
    parser.add_argument('--max_steps', type=int, default=250000, help='Maximum number of training steps')
    parser.add_argument('--eval_freq', type=int, default=2500, help='How often to evaluate the policy')
    parser.add_argument('--file_name', type=str, default="TD3", help='Name of the file to save results')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # Buffer and batch settings
    parser.add_argument('--buffer_size', type=int, default=250000, help='Size of the replay buffer')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--learning_starts', type=int, default=25000, help='Steps before starting learning')
    parser.add_argument('--start_time_steps', type=int, default=25000, help='Initial random action steps')
    
    # Algorithm parameters
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update parameter')
    parser.add_argument('--ro', type=float, default=0.9, help='Decay factor for updating nearby transitions')
    parser.add_argument('--alpha', type=float, default=0.6, help='Prioritization exponent')
    
    # TD3 specific
    parser.add_argument('--policy_noise', type=float, default=0.2, help='Noise added to target policy')
    parser.add_argument('--noise_clip', type=float, default=0.5, help='Range to clip target policy noise')
    parser.add_argument('--policy_freq', type=int, default=2, help='Delayed policy updates frequency')
    parser.add_argument('--exploration_noise', type=float, default=0.1, help='Exploration noise')
    
    # APSER specific
    parser.add_argument('--uniform_sampling_period', type=int, default=0, help='Period of uniform sampling')
    parser.add_argument('--beta', type=float, default=0.4, help='Importance sampling exponent')
    parser.add_argument('--PER', action='store_true', help='Use PER when not using APSER (default: False)')
    parser.add_argument('--non_linearity', default="sigmoid")
    # APSER specific - Boolean flags
    parser.add_argument('--use_apser', action='store_true', default=True,
                       help='Use APSER prioritization')
    parser.add_argument('--no_apser', action='store_false', dest='use_apser',
                       help='Disable APSER prioritization')
    
    parser.add_argument('--use_importance_weights', action='store_true', default=True,
                       help='Use importance sampling weights')
    parser.add_argument('--no_importance_weights', action='store_false', dest='use_importance_weights',
                       help='Disable importance sampling weights')
    
    parser.add_argument('--update_neighbors', action='store_true', default=True,
                       help='Update neighboring transitions')
    parser.add_argument('--no_update_neighbors', action='store_false', dest='update_neighbors',
                       help='Disable updating neighboring transitions')
    
    parser.add_argument('--per', action='store_true', default=False,
                       help='Use PER when not using APSER')

    # Rest of your arguments...
    args = parser.parse_args()
    return args    

def main():
    args = parse_args()
    # Extract all parameters from args
    env_name = args.env_name
    max_steps = args.max_steps
    eval_freq = args.eval_freq
    buffer_size = args.buffer_size
    batch_size = args.batch_size
    learning_starts = args.learning_starts
    start_time_steps = args.start_time_steps
    discount = args.discount
    tau = args.tau
    ro = args.ro
    policy_noise = args.policy_noise
    noise_clip = args.noise_clip
    policy_freq = args.policy_freq
    exploration_noise = args.exploration_noise
    use_APSER = args.use_apser
    use_importance_weights = args.use_importance_weights
    PER = args.PER
    uniform_sampling_period = args.uniform_sampling_period
    beta = args.beta
    file_name = args.file_name
    seed = args.seed
    update_neigbors = args.update_neighbors
    start_time_steps =learning_starts
    # Initialize environment
    env = gym.make(env_name)
    np.random.seed(seed)
    #env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_steps_before_truncation = env.spec.max_episode_steps
    device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
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
    agent_name = "TD3"
    file_suffix = "APSER" if use_APSER else ("PER" if PER else "vanilla")
    file_name = f"{file_suffix}_{agent_name}_{env_name}_{seed}"
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
    agent = TD3(**kwargs)
    done = True
    actor_losses = []
    critic_losses = []
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
        transition = [state, action, next_state, reward, terminated]
        td_error = agent.critic.Q1(torch.FloatTensor(np.array(state)).to(agent.device).unsqueeze(0), torch.FloatTensor(np.array(action)).to(agent.device).unsqueeze(0))  - \
        reward - discount * (1-done)*(agent.critic.Q1(torch.FloatTensor(np.array(next_state)).to(agent.device).unsqueeze(0), 
                                            torch.FloatTensor(agent.select_action(torch.FloatTensor(np.array(next_state)).to(agent.device).unsqueeze(0))).to(agent.device).unsqueeze(0)))
        td_errors.append(td_error.detach().cpu().numpy())
        initial_score = [0]  # Initial score for new transitions
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
                    states, actions, next_states, rewards, not_dones, weights = APSER(replay_buffer, agent, batch_size, beta, discount, ro, max_steps_before_truncation, update_neigbors = update_neigbors)
                weights = torch.as_tensor(weights, dtype=torch.float32).to(agent.device)
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
            critic_losses.append(critic_loss.item())
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
                evaluations.append(evaluate_policy(agent, env_name))
                save_with_unique_filename(evaluations, f"results/{file_name}_{t}")
                save_with_unique_filename(np.array(td_errors), f"results/{file_name}_td_errors_{t}")
                save_with_unique_filename(actor_losses, f"results/{file_name}_actor_losses_{t}")
                save_with_unique_filename(critic_losses, f"results/{file_name}_critic_losses_{t}")

if __name__ == "__main__":
    main()