#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu 
#SBATCH --output=p1.out
#SBATCH --gres=gpu:p100:1

module purge
module load nvidia-hpc-sdk

./p1 1
./p1 4
./p1 16