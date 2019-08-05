#!/bin/bash

#SBATCH --job-name=significance
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=45000M
#SBATCH --partition=gpu_shared_course

module purge

module load Miniconda3/4.3.27

export PYTHONIOENCODING=utf
source activate dl

cd ..
srun python3 significance.py
