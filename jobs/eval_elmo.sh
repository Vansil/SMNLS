#!/bin/bash

#SBATCH --job-name=eval.elmo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
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

# elmo-2
# srun python eval.py -c"output/elmo-2/snli/checkpoints/snli_epoch20" -o"output/elmo-2/snli/evaluation/"
# srun python eval.py -c"output/elmo-2/vpos/checkpoints/pos_epoch20" -o"output/elmo-2/vpos/evaluation/"
# srun python eval.py -c"output/elmo-2/vpos-snli/checkpoints/snli_epoch20" -o"output/elmo-2/vpos-snli/evaluation/"
# srun python eval.py -c"output/elmo-2/vpos-vua-snli/checkpoints/snli_epoch20" -o"output/elmo-2/vpos-vua-snli/evaluation/"
# srun python eval.py -c"output/elmo-2/vua/checkpoints/vua_epoch20" -o"output/elmo-2/vua/evaluation/"
# srun python eval.py -c"output/elmo-2/vua-snli/checkpoints/snli_epoch20" -o"output/elmo-2/vua-snli/evaluation/"
# srun python eval.py -c"output/elmo-2/vua-vpos/checkpoints/vua_epoch20" -o"output/elmo-2/vua-vpos/evaluation/"

# elmo-3
srun python eval.py -c"output/elmo-3/snli/checkpoints/snli_epoch20" -o"output/elmo-3/snli/evaluation/"
# srun python eval.py -c"output/elmo-3/vpos/checkpoints/pos_epoch20" -o"output/elmo-3/vpos/evaluation/"
# srun python eval.py -c"output/elmo-3/vpos-snli/checkpoints/snli_epoch20" -o"output/elmo-3/vpos-snli/evaluation/"
# srun python eval.py -c"output/elmo-3/vpos-vua-snli/checkpoints/snli_epoch20" -o"output/elmo-3/vpos-vua-snli/evaluation/"
srun python eval.py -c"output/elmo-3/vua/checkpoints/vua_epoch20" -o"output/elmo-3/vua/evaluation/"
srun python eval.py -c"output/elmo-3/vua-snli/checkpoints/snli_epoch20" -o"output/elmo-3/vua-snli/evaluation/"
# srun python eval.py -c"output/elmo-3/vua-vpos/checkpoints/vua_epoch20" -o"output/elmo-3/vua-vpos/evaluation/"

