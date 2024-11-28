# run_sac_experiments.sh
#!/bin/bash
CONDA_ENV="RL_env"

# Activate the Conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
# Global parameters
MAX_STEPS=1000000
BUFFER_SIZE=1000000
BATCH_SIZE=256
EVAL_FREQ=5000
AGENT_NAME="SAC"
learning_starts=25000
non_linearity="relu"
zeta_initial=0.50
zeta_final=0.75
# Environments to test
ENVIRONMENTS=(
    # "Hopper-v5"
    # "Walker2d-v5"
    # "Ant-v5"
    # "LunarLanderContinuous-v3"
    # "HalfCheetah-v5"
    # "Humanoid-v5"
    # "Swimmer-v5"
    "BipedalWalker-v3"
)
SEEDS=(0 1 2 3 4)

# Base command
BASE_CMD="python main_APSER_SAC.py --zeta_initial $zeta_initial --zeta_final $zeta_final --no_update_neighbors --max_steps $MAX_STEPS --buffer_size $BUFFER_SIZE --batch_size $BATCH_SIZE --eval_freq $EVAL_FREQ --learning_starts $learning_starts"
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
    
    # Plot results using Python script
    #python plot_results.py $env $use_apser $use_per $AGENT_NAME $MAX_STEPS $seed

    # Wait a bit between experiments
    sleep 5
}
# Run TD3 expersaciments
echo "Starting SAC experiments..."

for env in "${ENVIRONMENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        mkdir -p results
        run_experiment "$env" true false "$seed" true true
        mv results "per_alpha_0_critic_actor_apser_same_batch_zeta_annealed_050_075_1m_steps_${AGENT_NAME}_${env}_results_${seed}"
    done
done

echo "SAC experiments completed."