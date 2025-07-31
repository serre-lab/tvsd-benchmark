#!/bin/bash

job1_id=$(sbatch scripts/generate_activations.sh $1 | awk '{print $4}')
sbatch --dependency=afterok:$job1_id scripts/benchmark.sh $1
