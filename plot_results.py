import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sys
import os

def plot_results(env_name, use_apser, use_per, agent_name="TD3", number_of_steps=250000, seed = 0):
    base_path = os.getcwd() + "/results"
    
    # Construct file name based on whether APSER is used
    file_suffix = "APSER" if use_apser else ("per" if use_per else "vanilla")
    prefix = f"{file_suffix}_{agent_name}_{env_name}_{seed}_"
    file_steps = str(int(number_of_steps) -1)
    # Define paths for files to load
    eval_scores_path = os.path.join(base_path, prefix + file_steps +".npy")
    actor_losses_path = os.path.join(base_path, prefix + "actor_losses_" + file_steps + ".npy")
    #td_errors_path = os.path.join(base_path, prefix + "td_errors_" + file_steps + ".npy")
    critic_losses_path = os.path.join(base_path, prefix + "critic_losses_" + file_steps + ".npy")

    # Load the files
    eval_scores = np.load(eval_scores_path)
    actor_losses = np.load(actor_losses_path)
    #td_errors = np.load(td_errors_path)
    critic_losses = np.load(critic_losses_path)
    # Get absolute value of TD errors
    #td_errors = np.abs(td_errors.T[0][0])
    
    # Create plot
    fig = go.Figure()
    x = [i for i in range(len(actor_losses))]
    fig.add_trace(go.Scatter(x=x, y=actor_losses, mode='lines', name='Actor Losses'))
    fig.add_trace(go.Scatter(x=x, y=eval_scores, mode='lines', name='Eval Scores'))
    #fig.add_trace(go.Scatter(x=x, y=td_errors, mode='lines', name='TD Errors'))
    fig.add_trace(go.Scatter(x=x, y=critic_losses, mode='lines', name='Critic Losses'))

    # Add title and labels
    fig.update_layout(
        title=f'{env_name} - {"APSER" if use_apser else ("per" if use_per else "vanilla")}',
        xaxis_title='Steps',
        yaxis_title='Values',
        showlegend=True
    )

    # Save the plot as HTML
    output_file = f"{prefix}_experiments_{str(number_of_steps)}_steps.html"
    fig.write_html(output_file)
    print(f"Plot saved as {output_file}")

if __name__ == "__main__":
    env_name = sys.argv[1]
    use_apser = sys.argv[2].lower() == 'true'
    use_per = sys.argv[3].lower() == 'true'
    agent_name = sys.argv[4] 
    number_of_steps = sys.argv[5]
    seed = sys.argv[5]
    plot_results(env_name, use_apser, use_per, agent_name, number_of_steps, seed)
