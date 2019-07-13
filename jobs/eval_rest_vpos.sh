#!/bin/bash

#SBATCH --job-name=eval.vpos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --mem=15000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

module purge

module load Miniconda3/4.3.27
module load CUDA/9.0.176
module load cuDNN/7.3.1-CUDA-9.0.176

export PYTHONIOENCODING=utf
source activate dl

cd ..
srun python eval.py -c"output/vpos/checkpoints/pos_epoch20/" -o"output/vpos/evaluation/"
srun python eval.py -c"output/vpos-snli/checkpoints/snli_epoch20/" -o"output/vpos-snli/evaluation/" 
srun python eval.py -c"output/vpos-vua-snli/checkpoints/snli_epoch20/" -o"output/vpos-vua-snli/evaluation/"
srun python eval.py -c"output/vua-vpos/checkpoints/vua_epoch20/" -o"output/vua-vpos/evaluation/"
