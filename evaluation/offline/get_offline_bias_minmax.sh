#!/bin/bash
#SBATCH --job-name=get_offline_bias_minmax
#SBATCH -A m4334
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --output=get_offline_bias_minmax_%j.out
#SBATCH --error=get_offline_bias_minmax_%j.err
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH --mail-type=ALL

module load conda
conda activate plotting

python get_offline_bias_minmax.py