#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH -n 4
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
##SBATCH --mail-user=jerryL9@uci.edu
##SBATCH --mail-type=ALL
##SBATCH --output=out_%j.out
##SBATCH --error=eo_%j.err

cmd="python train_unet_h5loader_processed.py --config-name=config_single \
        data_path='/global/homes/z/zeyuanhu/scratch/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc_full_processed/'\
        expname='v2rh_mc_fix_huber' \
	variable_subsets='v2_rh_mc' \
	batch_size=1024 \
        num_workers=32 \
	qinput_prune=True \
        output_prune=True \
	input_clip=True \
	strato_lev=15 \
	strato_lev_out=12 \
	strato_lev_qinput=22 \
	input_clip_rhonly=True \
	aggressive_pruning=False \
        epochs=16 \
	optimizer='adam' \
	early_stop_step=-1 \
	loss='huber' \
	dropout=0.0 \
        save_top_ckpts=15 \
        learning_rate=0.0001 \
	logger='wandb' \
        wandb.project='v2rh_mc_unet' \
        scheduler_name='step' \
        scheduler.step.step_size=3 \
	scheduler.step.gamma=0.5 \
        unet_num_blocks=2 \
        unet_attn_resolutions=[0] \
        unet_model_channels=128 "

cd /global/cfs/cdirs/m4334/jerry/climsim3_dev/baseline_models/pao_model/training_jerry
srun -n $SLURM_NTASKS shifter bash -c "source ddp_export.sh && $cmd"
