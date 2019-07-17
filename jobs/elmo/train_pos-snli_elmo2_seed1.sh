#!/bin/bash

#SBATCH --job-name=p-s1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=32000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

module purge

module load Miniconda3/4.3.27
module load CUDA/9.0.176
module load cuDNN/7.3.1-CUDA-9.0.176

export PYTHONIOENCODING=utf
source activate dl

cd ../..
srun python3 -u train_jmt.py --output output/elmo-2/vpos-snli --tasks pos snli --embedding-model ELMo2+GloVe --seed 1