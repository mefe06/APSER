#!/bin/bash

# run_td3_experiments.sh
#!/bin/bash
CONDA_ENV="RL_env"

# Activate the Conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Global parameters
MAX_STEPS=250000
BUFFER_SIZE=250000
BATCH_SIZE=256
EVAL_FREQ=2500
AGENT_NAME="TD3"
learning_starts=25000
non_linearity="relu"

# Environments to test
ENVIRONMENTS=(
    "Hopper-v5"
    #"Walker2d-v5"
    "Ant-v5"
    #"HalfCheetah-v5"
)
SEEDS=(0 1 2)

# Base command
BASE_CMD="python main_APSER_TD3.py --update_neighbors --max_steps $MAX_STEPS --buffer_size $BUFFER_SIZE --batch_size $BATCH_SIZE --eval_freq $EVAL_FREQ --learning_starts $learning_starts"
# Function to run experiment
run_experiment() {
    local env=$1
    local use_apser=$2
    local use_per=$3
    local seed=$4
    local separate_apser=$5
    local same_batch=$6
    echo "Running experiment with:"
    echo "Environment: $env"
    echo "APSER: $use_apser"
    
    if [ "$use_apser" = true ]; then
        if [ "$separate_apser" = true ]; then
            if [ "$same_batch" = true ]; then
                cmd="$BASE_CMD --use_separate --same_batch --env_name $env --use_apser --seed $seed"
            else
                cmd="$BASE_CMD --use_separate --env_name $env --use_apser --seed $seed"
            fi
        else
            cmd="$BASE_CMD --env_name $env --use_apser --seed $seed"
        fi
    else
        if [ "$use_per" = true ]; then
            cmd="$BASE_CMD --env_name $env --per --no_apser --seed $seed"
        else
            cmd="$BASE_CMD --env_name $env --no_apser --seed $seed"
        fi
    fi
    
    echo "Command: $cmd"
    eval $cmd

    sleep 5
}

# Run TD3 expersaciments
echo "Starting TD3 experiments..."

for env in "${ENVIRONMENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        mkdir -p results
        run_experiment "$env" true false "$seed" true true
        mv results "per_alpha_1_critic_actor_apser_corrected_no_neighbors_${AGENT_NAME}_${env}_results_${seed}"
    done
done

echo "TD3 experiments completed."