#!/bin/bash
#SBATCH -A m4331
#SBATCH -C gpu
#SBATCH -t 0:10:00
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH -n 2
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
##SBATCH --mail-user=zeyuanh@nvidia.com
##SBATCH --mail-type=ALL
##SBATCH --output=out_%j.out
##SBATCH --error=eo_%j.err

cd /global/homes/z/zeyuanhu/nvidia_codes/Climsim_private/Hu_models/Unet/training
source ddp_export.sh
shifter source ddp_export.sh &&  mpirun -np $SLURM_NTASKS  python train_unet_ddp.py --config-name=config_single \
    qinput_log=True \
    expname='v3_unet_jit_test' \
    qinput_prune=True \
    qoutput_prune=True \
    epochs=1 \
    learning_rate=0.001 \
    early_stop_step=300 \
    logger='wandb' \
    wandb.project='v3_unet' \
    scheduler_name='step' \
    scheduler.step.step_size=3 \