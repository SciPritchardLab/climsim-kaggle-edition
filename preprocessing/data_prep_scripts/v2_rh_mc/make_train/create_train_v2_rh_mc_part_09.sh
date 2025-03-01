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

shifter python ../process_data_split_v2_rh_mc.py \
    'E3SM-MMF.ml2steploc.0005-*-*-*.nc' \
    --data_split 'train' \
    --stride_sample 2 \
    --start_idx 0 \
    --save_h5 True \
    --save_path '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/train_set/51/'