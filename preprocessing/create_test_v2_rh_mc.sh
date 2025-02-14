#!/bin/bash
#SBATCH -A m4334
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 8:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH --mail-type=ALL

shifter python create_test_v2_rh_mc.py \
            'E3SM-MMF.ml2steploc.0009-0[3456789]-*-*.nc' \
            'E3SM-MMF.ml2steploc.0009-1[012]-*-*.nc' \
            'E3SM-MMF.ml2steploc.0010-*-*-*.nc' \
            'E3SM-MMF.ml2steploc.0011-0[12]-*-*.nc'
    --save_path '/global/homes/j/jerrylin/scratch/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc_full/test_set/'
