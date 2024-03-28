#!/bin/bash
#SBATCH --job-name=ds_bloom
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=4
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --qos=qos_gpu-dev
#SBATCH --cpus-per-task=8
#SBATCH --account=knb@a100
#SBATCH -C a100

module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.0.1

## launch script on every node
set -x

# hack to avoid issues of very small $HOME
export HOME=$WORK"/home/"

# code execution
srun python dsbloom.py

