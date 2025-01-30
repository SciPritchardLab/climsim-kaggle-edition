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

cmd="python train_squeezeformer_h5loader.py --config-name=config_single \
        data_path='/global/homes/z/zeyuanhu/scratch/hugging/E3SM-MMF_ne4/preprocessing/v5_full_expanded2/'\
        expname='v5_sqeezeformer_confidence_test2' \
        variable_subsets='v5' \
        batch_size=768 \
        num_workers=32 \
        squeeze_dim=256 \
        squeeze_head_dim=1024 \
        epochs=4 \
        restart_path='/global/homes/z/zeyuanhu/scratch/hugging/E3SM-MMF_ne4/saved_models/v5_sqeezeformer_confidence_init/model.mdlus' \
        save_top_ckpts=15 \
        learning_rate=0.001 \
        optimizer='adamW' \
        weight_decay=4e-4 \
        clip_grad=True \
        clip_grad_norm=0.6 \
        logger='wandb' \
        wandb.project='v5_sqeezeformer' \
        scheduler_name='cosine_warmup' \
        scheduler.cosine_warmup.T_0=4 \
        scheduler.cosine_warmup.eta_min=3.0e-6 "

cd /global/homes/z/zeyuanhu/nvidia_codes/Climsim_private/Hu_models/squeezeformer_v5/training
srun -n $SLURM_NTASKS shifter bash -c "source ddp_export.sh && $cmd"