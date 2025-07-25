#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=120g
#SBATCH -o logs/inference_%A_%a.out
#SBATCH -e logs/inference_%A_%a.err
#SBATCH -t 24:00:00
#SBATCH --array=0-15

export PYTHONPATH="$PYTHONPATH:$(pwd)"
module load cuda cudnn
python -u benchmark.py  --model_config $1 \
                        --monkey monkeyF \
                        --array $SLURM_ARRAY_TASK_ID \
                        --n_splits 4 \
                        --preprocessed
                        
