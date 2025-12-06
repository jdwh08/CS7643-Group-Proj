#!/bin/bash
#SBATCH -N 1     
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 2:00:00                  
#SBATCH --gres=gpu:V100:1          
#SBATCH --mem-per-gpu=32G
#SBATCH -o logs/unet_s1weak_%j.out
#SBATCH -e logs/unet_s1weak_%j.err

set -euo pipefail

echo "=== Job started on $(date) ==="
echo "Running on node: $HOSTNAME"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

# -------------------------
# 1. Load CUDA
# -------------------------
module purge
module load cuda/12.1.1

# -------------------------
# 2. Activate your UV venv
# -------------------------
PROJECT_ROOT=$HOME/scratch/CS7643-Group-Proj
cd $PROJECT_ROOT

source .venv/bin/activate

echo "Python used:"
which python
python --version

# -------------------------
# 3. Run UNet on S1Weak
# -------------------------
echo "Starting training..."
python -m training.unet.train_unet

echo "=== Job finished on $(date) ==="