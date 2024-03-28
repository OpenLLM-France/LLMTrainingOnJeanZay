#!/bin/bash
#SBATCH --job-name=megatrondetok
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:50:00
#SBATCH --qos=qos_gpu-dev
#SBATCH --cpus-per-task=8
#SBATCH --account=knb@a100
#SBATCH -C a100

# hack to avoid issues of very small $HOME
export HOME=$WORK"/home/"

module load anaconda-py3/2023.09
conda activate megatron

# to get idr_torch
module load cpuarch/amd

## launch script on every node
set -x

echo "DATEDEBUT"
date

srun python mdetok.py

