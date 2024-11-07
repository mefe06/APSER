import numpy as np
import matplotlib.pyplot as plt
import argparse
import os 
import sys

def plot_averages(path: str, envs: list[str], agent_name: str ,seeds: list[int], steps: int):
    for env in envs:
        sum = None
        for seed in seeds:                
            file_path = f"{path}/uniform_critic_actor_apser_{agent_name}_{env}_results_{seed}/separate_APSER_{agent_name}_{env}_{seed}_{steps-1}.npy"
            if sum is None:
                sum =  np.load(file_path)
            else:
                sum += np.load(file_path)
        mean = sum/len(seeds)
        plt.plot([i for i in range(len(mean))], mean)
    env_names = [env.split("-")[0] for env in envs]
    plt.legend(env_names)
    plt.title(f"{agent_name} per for {steps} steps")
    joint_env_names = "_".join(env_names)
    plt.savefig(f"uniform_critic_apser_actor_{agent_name}_per_{steps//1000}k_{joint_env_names}")
    print("plot succesfully saved!")

def plot_indices_histogram(path: str, results_name, envs: list[str], agent_name: str, seeds: list[int], steps: int):
    for env in envs:
        actor_sum = None
        critic_sum = None
        
        for seed in seeds:
            # Load sampled indices data for each seed
            file_path = f"{path}/{results_name}_{agent_name}_{env}_results_{seed}/separate_APSER_{agent_name}_{env}_{seed}_sampled_indices_{steps-1}.npy"
            indices = np.load(file_path)
            
            # Split indices into actor and critic parts
            mid_point = len(indices[0]) // 2
            actor_indices = indices[:, :mid_point]
            critic_indices = indices[:, mid_point:]
            
            # Accumulate the histograms for averaging
            actor_hist, _ = np.histogram(actor_indices, bins=250, range=(0, steps))
            critic_hist, _ = np.histogram(critic_indices, bins=250, range=(0, steps))
            
            if actor_sum is None:
                actor_sum = actor_hist
                critic_sum = critic_hist
            else:
                actor_sum += actor_hist
                critic_sum += critic_hist
        
        # Compute the mean histogram across seeds
        actor_mean_hist = actor_sum / len(seeds)
        critic_mean_hist = critic_sum / len(seeds)
        
        # Plotting actor indices histogram
        plt.figure(figsize=(10, 5))
        plt.bar(range(250), actor_mean_hist, width=1.0, alpha=0.7, label="Actor Indices")
        plt.title(f"{agent_name} Actor Indices Histogram for {env} over {steps} steps")
        plt.xlabel("Index Bins")
        plt.ylabel("Frequency (Averaged)")
        plt.legend()
        plt.savefig(f"{results_name}_{agent_name}_actor_indices_{steps//1000}k_{env}.png")
        print(f"Actor indices histogram for {env} saved successfully.")
        
        # Plotting critic indices histogram
        plt.figure(figsize=(10, 5))
        plt.bar(range(250), critic_mean_hist, width=1.0, alpha=0.7, label="Critic Indices", color='orange')
        plt.title(f"{agent_name} Critic Indices Histogram for {env} over {steps} steps")
        plt.xlabel("Index Bins")
        plt.ylabel("Frequency (Averaged)")
        plt.legend()
        plt.savefig(f"{results_name}_{agent_name}_critic_indices_{steps//1000}k_{env}.png")
        print(f"Critic indices histogram for {env} saved successfully.")

import numpy as np
import matplotlib.pyplot as plt

def plot_sampling_frequency(path: str, result_name, envs: list[str], agent_name: str, seeds: list[int], steps: int):
    for env in envs:
        actor_count = None
        critic_count = None
        
        for seed in seeds:
            # Load sampled indices data for each seed
            file_path = f"{path}/{result_name}_{agent_name}_{env}_results_{seed}/APSER_{agent_name}_{env}_{seed}_sampled_indices_{steps-1}.npy"
            indices = np.load(file_path)
            
            # Split indices into actor and critic parts
            mid_point = len(indices[0]) // 2
            actor_indices = indices[:, :mid_point]
            critic_indices = indices[:, mid_point:]
            # Count sampling occurrences for each transition
            unique_actor_indices, actor_counts = np.unique(actor_indices, return_counts=True)
            unique_critic_indices, critic_counts = np.unique(critic_indices, return_counts=True)
            
            # Initialize the total counts for each transition if not done yet
            if actor_count is None:
                actor_count = np.zeros(steps)
                critic_count = np.zeros(steps)
            
            # Accumulate counts across seeds
            actor_count[unique_actor_indices] += actor_counts
            critic_count[unique_critic_indices] += critic_counts
        
        # Average the sampling frequency across seeds
        actor_mean_count = actor_count / len(seeds)
        critic_mean_count = critic_count / len(seeds)
        
        # Plotting actor sampling frequency
        plt.figure(figsize=(10, 5))
        plt.plot(range(steps), actor_mean_count, label="Actor Sampling Frequency", alpha=0.7)
        plt.title(f"{agent_name} Actor Sampling Frequency for {env} over {steps} steps")
        plt.xlabel("Transition Index")
        plt.ylabel("Sampling Count (Averaged)")
        plt.legend()
        plt.savefig(f"{result_name}_{agent_name}_actor_sampling_frequency_{steps//1000}k_{env}.png")
        print(f"Actor sampling frequency plot for {env} saved successfully.")
        
        # Plotting critic sampling frequency
        plt.figure(figsize=(10, 5))
        plt.plot(range(steps), critic_mean_count, label="Critic Sampling Frequency", alpha=0.7, color='orange')
        plt.title(f"{agent_name} Critic Sampling Frequency for {env} over {steps} steps")
        plt.xlabel("Transition Index")
        plt.ylabel("Sampling Count (Averaged)")
        plt.legend()
        plt.savefig(f"{result_name}_{agent_name}_critic_sampling_frequency_{steps//1000}k_{env}.png")
        print(f"Critic sampling frequency plot for {env} saved successfully.")


if __name__ == "__main__":
    # TODO: fill to parse from command line and run plot averages
    parser = argparse.ArgumentParser(description="Plot average performance over multiple environments and seeds.")
    dir_path = os.getcwd()
    # Define command-line arguments
    parser.add_argument("--path", type=str, required=False, default=dir_path, help="Path prefix for numpy files.")
    parser.add_argument("--result_name", type=str, required=True)
    parser.add_argument("--envs", type=str, nargs='+', required=True, help="List of environments.")
    parser.add_argument("--agent_name", type=str, required=True, help="Name of the agent.")
    parser.add_argument("--seeds", type=int, nargs='+', required=True, help="List of seed values.")
    parser.add_argument("--steps", type=int, required=True, help="Number of steps.")
    
    # Parse arguments
    args = parser.parse_args()
    plot_sampling_frequency(args.path, args.result_name, args.envs, args.agent_name, args.seeds, args.steps)
