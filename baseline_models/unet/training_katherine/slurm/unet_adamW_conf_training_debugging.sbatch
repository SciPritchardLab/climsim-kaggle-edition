#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH -n 1
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --mail-user=frieldskatherine@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=out_%j.out
#SBATCH --error=eo_%j.err

cmd="python train_unet_conf_v6_h5loader.py --config-name=config_adamW_v6_debugging.yaml"

cd /global/cfs/cdirs/m4334/kfrields/climsim3/baseline_models/unet/training_katherine
srun -n $SLURM_NTASKS shifter bash -c "source ddp_export.sh && $cmd"

