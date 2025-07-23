#!/bin/bash
#SBATCH -p gpu
#SBATCH --constraint=a5000
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=120g
#SBATCH -o logs/inference_%A_%a.out
#SBATCH -e logs/inference_%A_%a.err
#SBATCH -t 24:00:00
#SBATCH --array=0-4

export PYTHONPATH=$PYTHONPATH:/users/jamullik/scratch/TVSD-real
module load cuda cudnn
python -u generate_activations.py   --model_config configs/resnet.yaml \
                                    --monkey monkeyF \
                                    --array $SLURM_ARRAY_TASK_ID \
                                    --batch_size 64 \
                                    --save_every 4 \
                                    --hook_interval 8
