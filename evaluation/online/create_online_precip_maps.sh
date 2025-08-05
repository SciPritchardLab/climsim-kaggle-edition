#!/bin/bash
#SBATCH --job-name=create_online_precip_maps
#SBATCH -A m4334
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 1:10:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --output=create_online_precip_maps_%j.out
#SBATCH --error=create_online_precip_maps_%j.err
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH --mail-type=ALL

module load conda
conda activate plotting

python create_online_precip_maps.py