#!/bin/bash

# Sweep script for training diffusion language model with different parameter counts and learning rates

# Define model configurations (different parameter counts)
# Format: "n_layers n_heads d_model d_ff"
model_configs=(
    "2 4 128 512"    # Small model
    "4 8 256 1024"   # Medium model
    "6 8 512 2048"   # Base model (default from config.py)
    "8 8 768 3072"   # Large model
    "12 12 1024 4096" # XL model
)

# Define learning rates to sweep
learning_rates=(1e-5 3e-5 1e-4 3e-4 1e-3)

# Base output directory
base_out_dir="runs/sweep"

# Create base output directory
mkdir -p "$base_out_dir"

# Counter for experiments
exp_id=0

# Loop through model configurations
for model_config in "${model_configs[@]}"; do
    # Parse model config
    IFS=' ' read -r n_layers n_heads d_model d_ff <<< "$model_config"
    
    # Loop through learning rates
    for lr in "${learning_rates[@]}"; do
        # Calculate experiment number
        exp_id=$((exp_id + 1))
        
        # Create output directory for this experiment
        out_dir="${base_out_dir}/exp_${exp_id}_L${n_layers}H${n_heads}D${d_model}_lr${lr}"
        mkdir -p "$out_dir"
        
        # Print experiment info
        echo "Starting experiment $exp_id:"
        echo "  Model: L=${n_layers} H=${n_heads} D=${d_model} FF=${d_ff}"
        echo "  Learning rate: $lr"
        echo "  Output dir: $out_dir"
        echo ""
        
        # Run training with these parameters
        python train.py \
            --model.n_layers $n_layers \
            --model.n_heads $n_heads \
            --model.d_model $d_model \
            --model.d_ff $d_ff \
            --train.lr $lr \
            --train.out_dir "$out_dir" \
            --train.max_steps 1000 \
            --train.eval_every 200 \
            --train.save_every 500 \
            --train.log_every 50
            
        # Check if training succeeded
        if [ $? -eq 0 ]; then
            echo "Experiment $exp_id completed successfully"
        else
            echo "Experiment $exp_id failed"
        fi
        
        echo "----------------------------------------"
    done
done

echo "Sweep completed! Total experiments: $exp_id"
echo "Results saved in: $base_out_dir"