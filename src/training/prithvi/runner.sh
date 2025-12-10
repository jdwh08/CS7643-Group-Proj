#!/bin/bash
#SBATCH -N 1     
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 2:00:00                  
#SBATCH --gres=gpu:1
# by default this is the first available nvidia gpu

#SBATCH -J prithvi-train    # jobs name
# #SBATCH --mail-type=ALL #uncomment this if you want to receive emails
# #SBATCH --mail-user=email@domain.com #uncomment this if you want to receive emails

# Setup
module purge
module load cuda/12.1.1
nvcc --version || exit 1

cd "$HOME/scratch/CS7643-Group-Proj" || exit 1
source .venv/bin/activate || exit 1

# Determine config file based on execution mode
SCRIPT_DIR="$HOME/scratch/CS7643-Group-Proj/src/training/transformers_transfer_prithvi"

# Check if running as part of a job array
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # Array mode: use config from configs folder
    echo "Running in array mode (Task ID: $SLURM_ARRAY_TASK_ID)"
    
    CONFIGS_DIR="$SCRIPT_DIR/configs"
    
    # Get all YAML config files, sorted for consistency
    CONFIG_FILES=($(find "$CONFIGS_DIR" -name "*.yaml" -o -name "*.yml" | sort))
    
    # Check if we have any config files
    if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
        echo "Error: No config files found in $CONFIGS_DIR" >&2
        exit 1
    fi
    
    # Validate task ID is within range
    if [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] || [ "$SLURM_ARRAY_TASK_ID" -ge ${#CONFIG_FILES[@]} ]; then
        echo "Error: Task ID $SLURM_ARRAY_TASK_ID is out of range (0-$((${#CONFIG_FILES[@]} - 1)))" >&2
        exit 1
    fi
    
    # Select the config file for this task ID
    CONFIG_FILE="${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}"
    echo "CONFIG_FILE: $CONFIG_FILE"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file not found: $CONFIG_FILE" >&2
        exit 1
    fi
else
    # Standalone mode: use default config.yaml
    echo "Running in standalone mode"
    CONFIG_FILE="$SCRIPT_DIR/config.yaml"
    echo "CONFIG_FILE: $CONFIG_FILE"

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Default config file not found: $CONFIG_FILE" >&2
        exit 1
    fi
fi

# Run training with the selected config
python -u -m src.training.transformers_transfer_prithvi.runner --config "$CONFIG_FILE"
exit 0