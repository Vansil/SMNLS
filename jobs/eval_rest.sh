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
srun python eval.py -c"output/pos/checkpoints/pos_epoch20/" -o"output/pos/evaluation/"
srun python eval.py -c"output/pos-snli/checkpoints/snli_epoch20/" -o"output/pos-snli/evaluation/" 
srun python eval.py -c"output/pos-vua-snli/checkpoints/snli_epoch20/" -o"output/pos-vua-snli/evaluation/"
srun python eval.py -c"output/snli/checkpoints/snli_epoch20/" -o"output/snli/evaluation/"
srun python eval.py -c"output/vua/checkpoints/vua_epoch20/" -o"output/vua/evaluation/"
srun python eval.py -c"output/vua-pos/checkpoints/vua_epoch20/" -o"output/vua-pos/evaluation/"
srun python eval.py -c"output/vua-snli/checkpoints/snli_epoch20/" -o"output/vua-snli/evaluation/"
