#!/bin/bash

# run_td3_experiments.sh
#!/bin/bash
CONDA_ENV="RL"

# Activate the Conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Global parameters
MAX_STEPS=1000
BUFFER_SIZE=1000
BATCH_SIZE=256
EVAL_FREQ=250
AGENT_NAME="TD3"
learning_starts=250
# Environments to test
ENVIRONMENTS=(
    "Hopper-v5"
    "Walker2d-v5"
    "Ant-v5"
    "HalfCheetah-v5"
)
SEEDS=(1 2 3)

# Base command
BASE_CMD="python main_APSER_TD3.py --max_steps $MAX_STEPS --buffer_size $BUFFER_SIZE --batch_size $BATCH_SIZE --eval_freq $EVAL_FREQ --learning_starts $learning_starts"
# Function to run experiment
run_experiment() {
    local env=$1
    local use_apser=$2
    local seed=$3
    
    echo "Running experiment with:"
    echo "Environment: $env"
    echo "APSER: $use_apser"
    
    if [ "$use_apser" = true ]; then
        cmd="$BASE_CMD --env_name $env --use_APSER --seed $seed"
    else
        cmd="$BASE_CMD --env_name $env --no_APSER --seed $seed"
    fi
    
    echo "Command: $cmd"
    eval $cmd
    
    # Plot results using Python script
    python plot_results.py $env $use_apser $AGENT_NAME $MAX_STEPS

    # Wait a bit between experiments
    sleep 5
}

# Run TD3 experiments
echo "Starting TD3 experiments..."

for env in "${ENVIRONMENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # With APSER
        run_experiment $env true $seed
        #run_experiment $env false $seed
    done
done

echo "TD3 experiments completed."