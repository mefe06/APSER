import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from models.SAC import SAC
from models.APSER import separate_APSER, APSER, PER, PrioritizedReplayBuffer, ExperienceReplayBuffer
from models.utils import soft_update
from utils import evaluate_policy, save_with_unique_filename
import gymnasium as gym
import argparse
import pickle
def parse_args():
    parser = argparse.ArgumentParser(description='SAC with APSER training script')
    
    # Environment
    parser.add_argument('--env_name', type=str, default="Hopper-v5", help='Gymnasium environment name')
    parser.add_argument('--max_steps', type=int, default=250000, help='Maximum number of training steps')
    parser.add_argument('--eval_freq', type=int, default=2500, help='How often to evaluate the policy')
    parser.add_argument('--file_name', type=str, default="SAC", help='Name of the file to save results')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # Buffer and batch settings
    parser.add_argument('--buffer_size', type=int, default=250000, help='Size of the replay buffer')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--learning_starts', type=int, default=2500, help='Steps before starting learning')
    parser.add_argument('--start_time_steps', type=int, default=2500, help='Initial random action steps')
    
    # Algorithm parameters
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gamma', type=float, default=0.005, help='Soft update parameter')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update parameter')
    parser.add_argument('--ro', type=float, default=0.9, help='Decay factor for updating nearby transitions')
    parser.add_argument('--alpha', type=float, default=0.2, help='Temperature parameter')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size')
    
    # Noise parameters
    parser.add_argument('--policy_noise', type=float, default=0.2, help='Noise added to target policy')
    parser.add_argument('--noise_clip', type=float, default=0.5, help='Range to clip target policy noise')
    parser.add_argument('--exploration_noise', type=float, default=0.1, help='Exploration noise')
    
    # APSER specific
    parser.add_argument('--uniform_sampling_period', type=int, default=0, help='Period of uniform sampling')
    parser.add_argument('--beta', type=float, default=0.4, help='Importance sampling exponent')
    
    # SAC specific
    parser.add_argument('--policy_type', type=str, default="Gaussian", help='Policy type')
    parser.add_argument('--target_update_interval', type=int, default=1, help='Target network update interval')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, help='Automatic entropy tuning')

    parser.add_argument('--non_linearity', default="sigmoid")
    parser.add_argument('--use_apser', action='store_true', default=True,
                       help='Use APSER prioritization')
    parser.add_argument('--no_apser', action='store_false', dest='use_apser',
                       help='Disable APSER prioritization')
    parser.add_argument('--use_separate', action='store_true', default=False,
                       help='For APSER, use different batches from different buffers for actor and critic')    
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
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # Extract all parameters from args
    env_name = args.env_name
    max_steps = args.max_steps
    eval_freq = args.eval_freq
    file_name = args.file_name
    buffer_size = args.buffer_size
    batch_size = args.batch_size
    learning_starts = args.learning_starts
    start_time_steps = args.start_time_steps
    discount = args.discount
    tau = args.tau
    ro = args.ro
    alpha = args.alpha
    lr = args.lr
    hidden_size = args.hidden_size
    exploration_noise = args.exploration_noise
    use_APSER = args.use_apser
    use_importance_weights = args.use_importance_weights
    use_PER = args.per
    uniform_sampling_period = args.uniform_sampling_period
    beta = args.beta
    policy_type = args.policy_type
    target_update_interval = args.target_update_interval
    automatic_entropy_tuning = args.automatic_entropy_tuning
    gamma = discount
    seed = args.seed
    update_neigbors = args.update_neighbors
    separate_samples=args.use_separate
    start_time_steps =learning_starts

    # Hyperparameters
    env = gym.make(env_name)
    np.random.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_steps_before_truncation = env.spec.max_episode_steps
    device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
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
    sampled_indices = []
    agent_name = "SAC"
    file_suffix = "separate_APSER" if separate_samples else ("APSER" if use_APSER else ("PER" if use_PER else "vanilla"))
    file_name = f"{file_suffix}_{agent_name}_{env_name}_{seed}"
    # Initialize replay buffer and other variables
    if not(use_APSER or use_PER):
        replay_buffer = ExperienceReplayBuffer(state_dim, action_dim, buffer_size, device)
    else:
        if separate_samples:
            actor_replay_buffer = PrioritizedReplayBuffer(state_dim, action_dim, buffer_size, device)
            critic_replay_buffer = PrioritizedReplayBuffer(state_dim, action_dim, buffer_size, device)
        else:
            replay_buffer = PrioritizedReplayBuffer(state_dim, action_dim, buffer_size, device)
    evaluations = []
    agent = SAC(**kwargs)
    done = True
    actor_losses = []
    critic_losses = []
    total_it = 0
    for t in range(1, max_steps):
        if done:
            state, _ = env.reset()
        if t < start_time_steps:
            action = env.action_space.sample()
        else:
            action = (agent.select_action(np.array(state)) + np.random.normal(0, max_action * exploration_noise, size=action_dim)).clip(-max_action, max_action)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        transition = [state, action, next_state, reward, terminated]
        if separate_samples:
            critic_replay_buffer.add(*transition)
            actor_replay_buffer.add(*transition)
        else:
            replay_buffer.add(*transition)
        state = next_state
        # Do not sample from buffer until learning starts
        if t > learning_starts:
            # Sample from replay buffer
            if use_APSER:
                if separate_samples:
                    actor_states, _, _, _, _, _, actor_indices, critic_states, critic_actions, critic_next_states, critic_rewards, critic_not_dones, critic_weights, critic_indices = separate_APSER(critic_replay_buffer, actor_replay_buffer, agent, batch_size, beta, discount, ro, max_steps_before_truncation, update_neigbors)
                    sampled_indices.append(list(np.concatenate([actor_indices, critic_indices])))
                    weights = critic_weights
                else:
                    states, actions, next_states, rewards, not_dones, weights, indices = APSER(replay_buffer, agent, batch_size, beta, discount, ro, max_steps_before_truncation, update_neigbors = update_neigbors)
                    sampled_indices.append(list(indices)) 
                weights = torch.as_tensor(weights, dtype=torch.float32).to(agent.device)
            else:
                if use_PER:
                    states, actions, next_states, rewards, not_dones, weights, indices = PER(replay_buffer, agent, batch_size, discount)
                    sampled_indices.append(list(indices))
                    weights = torch.as_tensor(weights, dtype=torch.float32).to(agent.device) 
                else:
                    states, actions, next_states, rewards, not_dones = replay_buffer.sample(batch_size)
            if not(use_APSER&separate_samples):
                actor_states = critic_states = states
                _ = critic_actions = actions
                _ = critic_next_states = next_states
                _ = critic_rewards = rewards
                _ = critic_not_dones = not_dones
                _ = critic_weights = weights
                _ = _ = indices
            ### Update networks
            total_it += 1
            with torch.no_grad():
                # Select the target smoothing regularized action according to policy
                next_state_action, next_state_log_pi, _ = agent.actor.sample(critic_next_states)

                # Compute the target Q-value
                qf1_next_target, qf2_next_target = agent.critic_target(critic_next_states, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - agent.alpha * next_state_log_pi
                next_q_value = critic_rewards + critic_not_dones * agent.gamma * min_qf_next_target

            # Get the current Q-value estimates
            qf1, qf2 = agent.critic(critic_states, critic_actions)

            # Compute the critic loss
            if use_importance_weights&use_APSER:
                qf1_loss = torch.mean(critic_weights * F.mse_loss(qf1, next_q_value, reduction='none'))
                qf2_loss = torch.mean(critic_weights * F.mse_loss(qf2, next_q_value, reduction='none'))
            else: 
                qf1_loss = F.mse_loss(qf1, next_q_value)
                qf2_loss = F.mse_loss(qf2, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            critic_losses.append(qf_loss.item())
            # Optimize the critic
            agent.critic_optimizer.zero_grad()
            qf_loss.backward()
            agent.critic_optimizer.step()
            # Compute policy loss
            pi, log_pi, _ = agent.actor.sample(actor_states)

            qf1_pi, qf2_pi = agent.critic(actor_states, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((agent.alpha * log_pi) - min_qf_pi).mean()
            actor_losses.append(policy_loss.item())
            # Optimize the actor
            agent.actor_optimizer.zero_grad()
            policy_loss.backward()
            agent.actor_optimizer.step()

            # Tune the temperature coefficient
            if agent.automatic_entropy_tuning:
                if use_importance_weights&use_APSER:
                    alpha_loss = -(agent.log_alpha * (log_pi + agent.target_entropy).detach() * weights).mean()
                else:
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
                save_with_unique_filename(evaluations, f"results/{file_name}_{t}")
                save_with_unique_filename( np.array(sampled_indices), f"results/{file_name}_sampled_indices_{t}")
                save_with_unique_filename(actor_losses, f"results/{file_name}_actor_losses_{t}")
                save_with_unique_filename(critic_losses, f"results/{file_name}_critic_losses_{t}")
            if (t + 1) % (eval_freq*5) == 0:
                with open(f"results/sum_tree_debug_{t}.pkl", "wb") as f:
                    pickle.dump(actor_replay_buffer.tree, f)

if __name__ == "__main__":
    main()