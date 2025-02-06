#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu
#SBATCH -q debug 
#SBATCH -t 30:00
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH -n 8
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH --mail-type=ALL

set -e
set -x

export HYDRA_FULL_ERROR=1

cmd="python train_resLSTM_h5loader.py --config-name=config_debug"

cd /global/cfs/cdirs/m4334/jerry/climsim3_dev/online_testing/baseline_models/resLSTM/training
srun -n $SLURM_NTASKS shifter --module=gpu,nccl-plugin bash -c "source ddp_export.sh && $cmd"