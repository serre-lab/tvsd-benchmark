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

# Train split first: fits the IPCA basis and saves reduced train activations.
python -u generate_activations.py   --model_config $1 \
                                    --split train \
                                    --monkey monkeyF \
                                    --batch_size 128 \
                                    --pca_components 100 \
                                    --max_pca_train_batches 4

# Test split: reuses the train-fit IPCA (no refitting -> leak-free shared basis).
python -u generate_activations.py   --model_config $1 \
                                    --split test \
                                    --monkey monkeyF \
                                    --batch_size 128 \
                                    --pca_components 100
