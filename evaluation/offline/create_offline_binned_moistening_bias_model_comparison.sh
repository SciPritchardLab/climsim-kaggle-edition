#!/bin/bash
#SBATCH --job-name=create_offline_binned_moistening_bias_model_comparison
#SBATCH -A m4334
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --output=create_offline_binned_moistening_bias_model_comparison_%j.out
#SBATCH --error=create_offline_binned_moistening_bias_model_comparison_%j.err
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH --mail-type=ALL

module load conda
conda activate plotting

python create_offline_binned_moistening_bias_model_comparison.py