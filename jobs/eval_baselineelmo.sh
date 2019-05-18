#!/bin/bash

#SBATCH --job-name=eval.bases
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
srun python eval.py -c"output/baseline_elmo0/checkpoints/000000000.pt" -o"output/baseline_elmo0/evaluation/" -m=wic
srun python eval.py -c"output/baseline_elmo1/checkpoints/000000000.pt" -o"output/baseline_elmo1/evaluation/" -m=wic
srun python eval.py -c"output/baseline_elmo2/checkpoints/000000000.pt" -o"output/baseline_elmo2/evaluation/" -m=wic
srun python eval.py -c"output/baseline_elmo012/checkpoints/000000000.pt" -o"output/baseline_elmo012/evaluation/" -m=wic
