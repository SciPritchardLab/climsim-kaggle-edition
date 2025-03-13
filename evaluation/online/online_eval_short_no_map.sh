#!/bin/bash
#SBATCH -A m4334
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --output=online_eval_short_%j.out
#SBATCH --mail-user=frieldskatherine@gmail.com
#SBATCH --mail-type=ALL

shifter python online_eval_short.py \
        --mmf_path '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/mmf_runs/mmf_speedeval_gpu/run/mmf_speedeval_gpu.eam.h2.0003-01-*.nc' \
        --nn_path '/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/unet_adamW/online/unet_adamW/run/unet_adamW.eam.h2.0003-01-*.nc' \
        --save_path '/pscratch/sd/k/kfrields/hugging/scoring/unet_adamW_output' \
        --var 'Q' \
        --max_day 20 \