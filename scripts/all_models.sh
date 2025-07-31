#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 1:00:00
#SBATCH -o logs/all_models_%A_%a.out
#SBATCH -e logs/all_models_%A_%a.err
#SBATCH --array=0-3

# Run all models present in configs/models.csv
# Array size is set to match the number of models in the CSV (0-2 for 3 models)

# Read the config file path for the current array task
config_file=$(tail -n +2 configs/models.csv | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" | cut -d',' -f2)

# Run gen_bench.sh with the config file
echo "Running gen_bench.sh with config file: $config_file"
scripts/gen_bench.sh "$config_file"

