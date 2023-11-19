#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu 
#SBATCH --output=p2.out
#SBATCH --gres=gpu:p100:1

module purge
module load nvidia-hpc-sdk

./p2 1
./p2 4
./p2 16