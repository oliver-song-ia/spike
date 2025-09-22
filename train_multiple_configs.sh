#!/bin/bash

# Bash script to train models with different configurations
# Usage: ./train_multiple_configs.sh

set -e  # Exit on any error

echo "=================================="
echo "Starting multi-configuration training"
echo "=================================="

# Define configurations to train
CONFIGS=(
    "experiments/ITOP-SIDE/5"
    "experiments/ITOP-SIDE/6" 
    "experiments/ITOP-SIDE/7"
    "experiments/ITOP-SIDE/8"
)

# Function to check if config directory exists
check_config_exists() {
    local config_path="$1"
    if [ ! -d "$config_path" ]; then
        echo "❌ Config directory not found: $config_path"
        return 1
    fi
    if [ ! -f "$config_path/config.yaml" ]; then
        echo "❌ Config file not found: $config_path/config.yaml"
        return 1
    fi
    echo "✅ Config found: $config_path"
    return 0
}

# Function to train a single configuration
train_config() {
    local config_path="$1"
    local config_name=$(basename "$config_path")
    
    echo ""
    echo "========================================="
    echo "🚀 Starting training: $config_name"
    echo "========================================="
    
    # Record start time
    start_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Start time: $start_time"
    
    # Run training
    if python train_itop.py --config "$config_path"; then
        end_time=$(date '+%Y-%m-%d %H:%M:%S')
        echo "✅ Training completed successfully: $config_name"
        echo "End time: $end_time"
        
        # Log success
        echo "$(date '+%Y-%m-%d %H:%M:%S') - SUCCESS: $config_name" >> training_log.txt
    else
        end_time=$(date '+%Y-%m-%d %H:%M:%S')
        echo "❌ Training failed: $config_name"
        echo "End time: $end_time"
        
        # Log failure
        echo "$(date '+%Y-%m-%d %H:%M:%S') - FAILED: $config_name" >> training_log.txt
        
        # Ask user if they want to continue
        read -p "Training failed for $config_name. Continue with next config? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Training stopped by user."
            exit 1
        fi
    fi
}

# Main execution
echo "Checking all configurations..."

# Check if all configs exist first
for config in "${CONFIGS[@]}"; do
    if ! check_config_exists "$config"; then
        echo "❌ Stopping due to missing configuration"
        exit 1
    fi
done

echo ""
echo "All configurations found. Starting training sequence..."

# Initialize log file
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting multi-config training" > training_log.txt

# Train each configuration
for config in "${CONFIGS[@]}"; do
    train_config "$config"
    
    # Small delay between trainings
    echo "Waiting 10 seconds before next training..."
    sleep 10
done

echo ""
echo "========================================="
echo "🎉 All training sessions completed!"
echo "========================================="
echo "Check training_log.txt for summary"

# Display final summary
echo ""
echo "Training Summary:"
echo "=================="
cat training_log.txt

echo ""
echo "Script completed at: $(date '+%Y-%m-%d %H:%M:%S')"