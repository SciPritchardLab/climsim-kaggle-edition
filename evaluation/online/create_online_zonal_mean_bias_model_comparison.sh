#!/bin/bash
#SBATCH --job-name=create_online_zonal_mean_bias_model_comparison
#SBATCH -A m4334
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 5:30:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --output=create_online_zonal_mean_bias_model_comparison_%j.out
#SBATCH --error=create_online_zonal_mean_bias_model_comparison_%j.err
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH --mail-type=ALL

module load conda
conda activate plotting

python create_online_zonal_mean_bias_model_comparison.py