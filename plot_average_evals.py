import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_averages(path: str, envs: list[str], agent_name: str ,seeds: list[int], steps: int):
    for env in envs:
        sum = None
        for seed in seeds:                
            file_path = f"{path}/per_{env}_results_{seed}/PER_{agent_name}_{env}_{seed}_{steps-1}.npy"
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
    plt.savefig(f"{agent_name}_per_{steps//1000}k_{joint_env_names}")
    print("plot succesfully saved!")

if __name__ == "__main__":
    # TODO: fill to parse from command line and run plot averages
    parser = argparse.ArgumentParser(description="Plot average performance over multiple environments and seeds.")
    
    # Define command-line arguments
    parser.add_argument("--path", type=str, required=True, help="Path prefix for numpy files.")
    parser.add_argument("--envs", type=str, nargs='+', required=True, help="List of environments.")
    parser.add_argument("--agent_name", type=str, required=True, help="Name of the agent.")
    parser.add_argument("--seeds", type=int, nargs='+', required=True, help="List of seed values.")
    parser.add_argument("--steps", type=int, required=True, help="Number of steps.")
    
    # Parse arguments
    args = parser.parse_args()

    # Run the function with parsed arguments
    plot_averages(args.path, args.envs, args.agent_name, args.seeds, args.steps)
