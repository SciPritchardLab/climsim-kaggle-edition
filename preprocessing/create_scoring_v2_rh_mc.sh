#!/bin/bash
#SBATCH -A m4334
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 50:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --output=create_scoring_%j.out
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH --mail-type=ALL

shifter python create_scoring_v2_rh_mc.py \
            'E3SM-MMF.ml2steploc.0008-0[23456789]-*-*.nc' \
            'E3SM-MMF.ml2steploc.0008-1[012]-*-*.nc' \
            'E3SM-MMF.ml2steploc.0009-01-*-*.nc' \
    --save_path '/global/homes/j/jerrylin/scratch/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc_full/scoring_set/'