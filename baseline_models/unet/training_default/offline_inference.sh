#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 20:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --output=offline_inference_%j.out
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH --mail-type=ALL

shifter python offline_inference.py