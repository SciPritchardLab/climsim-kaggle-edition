#!/bin/bash
#SBATCH -A m4331
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH -n 8
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
##SBATCH --mail-user=zeyuan_hu@fas.harvard.com
##SBATCH --mail-type=ALL
##SBATCH --output=out_%j.out
##SBATCH --error=eo_%j.err

cmd="python train_sqeezeformer_h5loader.py --config-name=config_single \
        data_path='/global/homes/z/zeyuanhu/scratch/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_full/'\
        expname='v2rh_sqeezeformer_confidence' \
        variable_subsets='v2_rh' \
        batch_size=512 \
        num_workers=32 \
        squeeze_dim=256 \
        squeeze_head_dim=1024 \
        epochs=4 \
        restart_path='/global/homes/z/zeyuanhu/scratch/hugging/E3SM-MMF_ne4/saved_models/v2rh_sqeezeformer_confidence_init/model.mdlus' \
        save_top_ckpts=15 \
        learning_rate=0.001 \
        logger='wandb' \
        wandb.project='v2rh_sqeezeformer' \
        scheduler_name='cosine_warmup' \
        scheduler.cosine_warmup.T_0=4 \
        scheduler.cosine_warmup.eta_min=1.5e-6 "

cd /global/homes/z/zeyuanhu/nvidia_codes/Climsim_private/Hu_models/sqeezeformer_v2rh/training
srun -n $SLURM_NTASKS shifter bash -c "source ddp_export.sh && $cmd"