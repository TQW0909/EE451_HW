#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu 
#SBATCH --output=p1_p2.out
#SBATCH --gres=gpu:v100:1

module purge
module load nvidia-hpc-sdk

nvcc p1.cu -o p1
nvcc p2.cu -o p2

./p1

./p2