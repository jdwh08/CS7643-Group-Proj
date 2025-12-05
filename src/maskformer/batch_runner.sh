#!/bin/bash
#SBATCH -N 1     
#SBATCH -c 64
#SBATCH --ntasks-per-node=1
#SBATCH -t 1:30:00                  
#SBATCH --gres=gpu:1          

#SBATCH -J maskformer-weak    # jobs name

# And then some code to run, like
source ../../.venv/bin/activate
srun python3 runner.py
exit 0