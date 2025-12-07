#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8                         # CPU cores
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00                  # walltime
#SBATCH --gres=gpu:V100:1            # 1 GPU
#SBATCH --mem-per-gpu=16G
#SBATCH -J unet-s1hand               # job name
#SBATCH -o slurm_outs/unet_s1hand_%j.out   # stdout/stderr to file

# ---- Load modules / env ----
module purge
module load cuda/12.1.1

# Move to project root
cd $HOME/scratch/CS7643-Group-Proj

# Activate venv
source .venv/bin/activate

# Make sure slurm_outs exists
mkdir -p slurm_outs

# echo some debug info
echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi || echo "nvidia-smi not available"

# If you ever need PYTHONPATH, uncomment:
# export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# ---- Run UNet training (config_unet.yml controls dataset=weak, loss, etc.) ----
python -u -m src.training.unet.runner

echo "End time: $(date)"