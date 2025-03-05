#!/bin/bash
#SBATCH -A m4334
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --output=online_eval_short_%j.out
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH --mail-type=ALL

shifter python online_eval_short.py \
        --mmf_path '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/mmf_runs/mmf_speedeval_gpu/run/mmf_speedeval_gpu.eam.h2.0003-01-*.nc' \
        --nn_path '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/online_runs/climsim3_allhands/pao_model_debug_ps/run/pao_model_debug_ps.eam.h2.0003-01-*.nc' \
        --save_path '/pscratch/sd/j/jerrylin/debug_folder' \
        --var 'T' \
        --max_day 20 \