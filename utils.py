import gymnasium as gym
import numpy as np
import os 

def evaluate_policy(agent, env_name, eval_episodes=10, show_evals = False):
    eval_env = gym.make(env_name)
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

def save_with_unique_filename(array, base_filename):
    # Split the base filename and extension
    filename, extension = os.path.splitext(base_filename)
    
    # Initialize a counter
    counter = 1
    # Check if the file exists
    while os.path.exists(base_filename):
        # If it exists, modify the filename by appending the counter
        base_filename = f"{filename}{counter}{extension}"
        counter += 1
    
    # Save the array using np.save
    np.save(base_filename, array)
