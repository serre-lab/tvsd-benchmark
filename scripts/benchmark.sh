#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=120g
#SBATCH -o logs/benchmark_%A_%a.out
#SBATCH -e logs/benchmark_%A_%a.err
#SBATCH -t 24:00:00
#SBATCH --array=0-2

regions=("V1" "V4" "IT")
export PYTHONPATH="$PYTHONPATH:$(pwd)"
module load cuda cudnn
# Fit on train activations, score on the held-out test images. --skip_pca because
# activations were already reduced by the train-fit IPCA at generation time (the
# same basis is applied to the test images, so it is leak-free).
python -u benchmark.py  --model_config $1 \
                        --monkey monkeyF \
                        --region ${regions[$SLURM_ARRAY_TASK_ID]} \
                        --skip_pca \
                        --ceiling_normalize

