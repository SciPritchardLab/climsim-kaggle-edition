#!/bin/bash
#SBATCH -A m4334
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --mail-user=frieldskatherine@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=out_%j.out
#SBATCH --error=eo_%j.err

cmd="python unet_conf_offline_inference_debug.py"

cd /global/cfs/cdirs/m4334/kfrields/climsim3/baseline_models/unet/training_katherine/offline
srun -n $SLURM_NTASKS shifter bash -c "source ddp_export.sh && $cmd"
