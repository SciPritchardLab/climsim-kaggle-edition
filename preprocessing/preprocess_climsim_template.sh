#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 3:00:00
#SBATCH -n 1
#SBATCH -c 32
##SBATCH --gpus-per-task=1
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH --mail-type=ALL
##SBATCH --output=out_%j.out
##SBATCH --error=eo_%j.err

cd /global/homes/z/zeyuanhu/nvidia_codes/Climsim_private/preprocessing
shifter python create_npy_data_split_v4.py \
    'E3SM-MMF.ml2steploc.0003-*-*-*.nc' \
    --save_path '/global/homes/z/zeyuanhu/scratch/hugging/E3SM-MMF_ne4/preprocessing/v4_full/32/' \
    --start_idx 1
