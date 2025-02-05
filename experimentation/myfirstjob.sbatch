#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -J practice_run
#SBATCH --mail-user=frieldskatherine@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH -A m4334
#SBATCH -t 0:0:10
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --gpus-per-node 1
#SBATCH -n 1

cd /global/homes/k/kfrields

srun  shifter
bash -c python data_evaluation.py