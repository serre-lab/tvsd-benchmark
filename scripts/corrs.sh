#!/bin/bash
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=80g
#SBATCH -t 00:40:00
#SBATCH -p gpu
#SBATCH -o logs/corrs_%j.out
#SBATCH -e logs/corrs_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH -J tvsd_corrs
#SBATCH --array=0-16

module load cuda cudnn
python -u correlations.py \
    --array $SLURM_ARRAY_TASK_ID \
    --output_file results/corrs_${SLURM_ARRAY_TASK_ID}.json
