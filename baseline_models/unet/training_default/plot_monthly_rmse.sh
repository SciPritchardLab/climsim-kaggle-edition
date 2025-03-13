#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 30:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --output=plot_monthly_rmse_%j.out
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH --mail-type=ALL

shifter python plot_monthly_rmse.py \
        --shared_path '/pscratch/sd/z/zeyuanhu/hu_etal2024_data_v2/data/' \
        --hybrid_path_h0 '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/online_runs/climsim3_ensembles/unet/unet_seed_43_long/run' \
        --save_path '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/evaluation_figures/unet/unet_seed_43' \