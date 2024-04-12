#!/bin/bash
#SBATCH --job-name=data
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --account=knb@a100
#SBATCH --partition=prepost

# hack to avoid issues of very small $HOME
export HOME=$WORK"/home/"

module load pytorch-gpu/py3/2.2.0

echo "DATEDEBUT"
date

srun python fullcorpus2.py

