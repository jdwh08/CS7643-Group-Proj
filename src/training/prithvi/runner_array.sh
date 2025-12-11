#!/bin/bash
# Run a job for each config file in the configs directory

CONFIGS_DIR="$HOME/scratch/CS7643-Group-Proj/src/training/prithvi/configs"
RUNNER_SCRIPT="$HOME/scratch/CS7643-Group-Proj/src/training/prithvi/runner.sh"

# Find all config files
CONFIG_FILES=($(find "$CONFIGS_DIR" -name "*.yaml" -o -name "*.yml" | sort))

# Check if we have any config files
if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
    echo "Error: No config files found in $CONFIGS_DIR" >&2
    echo "Please add .yaml or .yml files to the configs directory." >&2
    exit 1
fi

NUM_CONFIGS=${#CONFIG_FILES[@]}
echo "Found $NUM_CONFIGS config file(s):"
for i in "${!CONFIG_FILES[@]}"; do
    echo "  [$i] $(basename "${CONFIG_FILES[$i]}")"
done
echo ""

# Calculate array range (0 to NUM_CONFIGS-1)
TOTAL_ARRAY_RANGE="0-$((NUM_CONFIGS - 1))"

echo "SLURM job array: $TOTAL_ARRAY_RANGE, $NUM_CONFIGS jobs."

MAX_CONCURRENT=3
ARRAY_RANGE="$TOTAL_ARRAY_RANGE%$MAX_CONCURRENT"
echo "Using $MAX_CONCURRENT concurrent job(s)."

# Submit the job array
sbatch --array="$ARRAY_RANGE" "$RUNNER_SCRIPT"

echo "Job array submitted!"
