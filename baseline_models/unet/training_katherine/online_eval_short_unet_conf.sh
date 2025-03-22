#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --output=online_eval_short_%j.out
#SBATCH --mail-user=frieldskatherine@gmail.com
#SBATCH --mail-type=ALL

shifter python online_eval_short_no_map.py \
        --mmf_path '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/mmf_runs/mmf_speedeval_gpu/run/mmf_speedeval_gpu.eam.h2.0003-01-*.nc' \
        --nn_path '/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/unet_adamW_conf_2/online/unet_conf_1_month/run/unet_conf_1_month.eam.h2.0003-01-*.nc' \
        --save_path '/pscratch/sd/k/kfrields/hugging/scoring/unet_conf_output' \
        --var 'Q' \
        --max_day 20 \
        --grid_path '/global/cfs/cdirs/m4334/jerry/climsim3_dev/grid_info/ClimSim_low-res_grid-info.nc' \
        --input_mean_file 'input_mean_v6_pervar.nc' \
        --input_max_file 'input_max_v6_pervar.nc' \
        --input_min_file 'input_min_v6_pervar.nc' \
        --output_scale_file 'output_scale_std_lowerthred_v6.nc' \
        --lbd_qn_file 'qn_exp_lambda_large.txt'