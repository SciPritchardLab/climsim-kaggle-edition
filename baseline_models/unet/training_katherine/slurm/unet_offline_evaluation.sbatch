#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 04:00:00
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH -n 1
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --mail-user=frieldskatherine@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=out_%j.out
#SBATCH --error=eo_%j.err

cmd="python unet_offline_inference_debug.py"

cd /global/cfs/cdirs/m4334/kfrields/climsim3/baseline_models/unet/training_katherine
srun -n $SLURM_NTASKS shifter bash -c "source ddp_export.sh && $cmd"
