#!/bin/bash
#SBATCH -N 1     
#SBATCH -c 64
#SBATCH --ntasks-per-node=1
#SBATCH -t 2:00:00                  
#SBATCH --gres=gpu:V100:1          
#SBATCH --mem-per-gpu=32G
#SBATCH -J maskformer-weak    # jobs name

# And then some code to run, like
python3 runner.py