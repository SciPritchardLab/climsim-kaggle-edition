#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH -t 50:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --output=offline_inference_%j.out
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH --mail-type=ALL

shifter python offline_inference_v6.py