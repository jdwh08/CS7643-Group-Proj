#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8                        
#SBATCH --ntasks-per-node=1
#SBATCH -t 05:00:00                   # walltime
#SBATCH --gres=gpu:A100:1             # 1 A100 GPU
#SBATCH --mem-per-gpu=32G
#SBATCH -J unet-s1                    # job name (shared for array)
#SBATCH -o slurm_outs/unet_%A_%a.out  # stdout/stderr per array task

# -------------------------------
# Select config based on array ID
# -------------------------------
# 0 -> train on S1Weak
# 1 -> train on S1Hand

CONFIG_WEAK="src/training/unet/config_unet_s1weak.yml"
CONFIG_HAND="src/training/unet/config_unet_s1hand.yml"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    CONFIG_PATH="${CONFIG_WEAK}"
    RUN_LABEL="s1weak"
    ;;
  1)
    CONFIG_PATH="${CONFIG_HAND}"
    RUN_LABEL="s1hand"
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

# -------------------------------
# Environment setup
# -------------------------------
module purge
module load cuda/12.1.1

cd "$HOME/scratch/CS7643-Group-Proj" || exit 1

source .venv/bin/activate

mkdir -p slurm_outs

echo "=================================================="
echo "UNet training run: ${RUN_LABEL}"
echo "Config path: ${CONFIG_PATH}"
echo "Job ID: ${SLURM_JOB_ID}, Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi || echo "nvidia-smi not available"
echo "=================================================="

# -------------------------------
# Run training
# -------------------------------
python -u -m src.training.unet.runner --config "${CONFIG_PATH}"

echo "End time: $(date)"