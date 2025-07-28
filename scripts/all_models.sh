#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o logs/all_models_%A_%a.out
#SBATCH -e logs/all_models_%A_%a.err
#SBATCH --array=0-15
