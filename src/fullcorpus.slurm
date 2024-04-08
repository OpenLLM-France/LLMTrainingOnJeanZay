#!/bin/bash
#SBATCH --job-name=data
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --account=knb@a100
#SBATCH --partition=prepost

# hack to avoid issues of very small $HOME
export HOME=$WORK"/home/"

module load python/3.11.5

echo "DATEDEBUT"
date

srun python fullcorpus.py

