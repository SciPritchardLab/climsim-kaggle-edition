#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 40:00
#SBATCH -n 1
#SBATCH -c 32
##SBATCH --gpus-per-task=1
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH --mail-type=ALL
##SBATCH --output=out_%j.out
##SBATCH --error=eo_%j.err

cd ..
shifter python process_data_v2_rh_mc.py \
            'E3SM-MMF.ml2steploc.0009-0[3456789]-*-*.nc' \
            'E3SM-MMF.ml2steploc.0009-1[012]-*-*.nc' \
            'E3SM-MMF.ml2steploc.0010-*-*-*.nc' \
            'E3SM-MMF.ml2steploc.0011-0[12]-*-*.nc' \
    --data_split 'test' \
    --stride_sample 12 \
    --start_idx 1 \
    --save_h5 True \
    --save_path '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/'