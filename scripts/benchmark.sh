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
# --skip_pca: activations were already reduced by IncrementalPCA at generation
#   time. NOTE: that IPCA basis was fit over the full image set (train+test), so it
#   leaks test data into the reduction. For a leak-free estimate, generate raw
#   activations and drop --skip_pca so PCA is fit per train fold instead.
python -u benchmark.py  --model_config $1 \
                        --monkey monkeyF \
                        --region ${regions[$SLURM_ARRAY_TASK_ID]} \
                        --n_splits 10 \
                        --skip_pca

