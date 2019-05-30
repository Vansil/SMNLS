#!/bin/bash

#SBATCH --job-name=tsne
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00:00
#SBATCH --mem=45000M
#SBATCH --partition=gpu_shared_course

module purge

module load Miniconda3/4.3.27

export PYTHONIOENCODING=utf
source activate dl

cd ..
srun python tsne_prep.py
