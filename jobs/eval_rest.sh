#!/bin/bash

#SBATCH --job-name=eval.rest
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
srun python eval.py -c"output0/pos/checkpoints/pos_epoch20.pt" -o"output0/pos/evaluation/"
srun python eval.py -c"output0/pos-snli/checkpoints/snli_epoch20.pt" -o"output0/pos-snli/evaluation/" 
srun python eval.py -c"output0/pos-vua-snli/checkpoints/snli_epoch15.pt" -o"output0/pos-vua-snli/evaluation/"
srun python eval.py -c"output0/snli/checkpoints/snli_epoch20.pt" -o"output0/snli/evaluation/"
srun python eval.py -c"output0/vua/checkpoints/vua_epoch20.pt" -o"output0/vua/evaluation/"
srun python eval.py -c"output0/vua-pos/checkpoints/vua_epoch20.pt" -o"output0/vua-pos/evaluation/"
srun python eval.py -c"output0/vua-snli/checkpoints/snli_epoch14.pt" -o"output0/vua-snli/evaluation/"
