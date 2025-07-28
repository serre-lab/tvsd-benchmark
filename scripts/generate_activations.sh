#!/bin/bash
#SBATCH -p gpu
#SBATCH --constraint=a5000
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=192g
#SBATCH -o logs/generate_activations_%A_%a.out
#SBATCH -e logs/generate_activations_%A_%a.err
#SBATCH -t 24:00:00

export PYTHONPATH="$PYTHONPATH:$(pwd)"
module load cuda cudnn
python -u generate_activations.py   --model_config $1 \
                                    --monkey monkeyF \
                                    --batch_size 256 \
                                    --pca_components 100
