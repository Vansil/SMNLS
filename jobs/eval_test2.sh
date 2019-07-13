#!/bin/bash

#SBATCH --job-name=eval.test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=6000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

module purge

module load Miniconda3/4.3.27
module load CUDA/9.0.176
module load cuDNN/7.3.1-CUDA-9.0.176

export PYTHONIOENCODING=utf
source activate dl

cd ..
srun python3 eval.py -c=output/pos-vua-snli/checkpoints/snli_epoch20/ -m=wic -o=output/pos-vua-snli/evaluation/
