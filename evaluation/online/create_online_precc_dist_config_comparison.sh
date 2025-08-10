#!/bin/bash
#SBATCH --job-name=create_online_precc_dist_config_comparison
#SBATCH -A m4334
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --output=create_online_precc_dist_config_comparison_%j.out
#SBATCH --error=create_online_precc_dist_config_comparison_%j.err
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH --mail-type=ALL

module load conda
conda activate plotting

python create_online_precc_dist_config_comparison.py