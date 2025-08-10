#!/bin/bash
#SBATCH -A m4334
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 30:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --output=create_offline_zonal_mean_tendency_conf_%j.out
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH --mail-type=ALL

shifter python create_offline_zonal_mean_tendency_conf.py