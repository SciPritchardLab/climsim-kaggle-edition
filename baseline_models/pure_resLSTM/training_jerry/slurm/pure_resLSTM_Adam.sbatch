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
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH --mail-type=ALL

cmd="python train_pure_resLSTM_h5loader.py --config-name=config_Adam"

cd /global/cfs/cdirs/m4334/jerry/climsim3_dev/baseline_models/pure_resLSTM/training_jerry
srun -n $SLURM_NTASKS shifter bash -c "source ddp_export.sh && $cmd"