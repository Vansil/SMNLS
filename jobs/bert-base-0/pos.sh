#!/bin/bash

#SBATCH --job-name=pos.train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=16384M
#SBACTH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

module purge

module load Miniconda3/4.3.27
module load CUDA/9.0.176
module load cuDNN/7.3.1-CUDA-9.0.176

export PYTHONIOENCODING=utf
source activate dl

srun python3 -u train_jmt.py --output output/bert-base-0/pos --tasks pos --embedding-model bert-base-cased
