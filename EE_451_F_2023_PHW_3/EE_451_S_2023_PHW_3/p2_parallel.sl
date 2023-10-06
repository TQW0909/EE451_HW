#!/bin/bash
#SBATCH --job-name=p2_parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=p2_parallel.out
#SBATCH --error=p2_parallel.err

module purge
module load gcc/11.3.0

ulimit -s unlimited

gcc -lrt -fopenmp -o p2_parallel p2_parallel.c

./p2_parallel

