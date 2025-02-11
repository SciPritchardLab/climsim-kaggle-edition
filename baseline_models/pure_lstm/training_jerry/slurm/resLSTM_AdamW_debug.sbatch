#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 10:00
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH -n 4
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH --mail-type=ALL
#SBATCH --output=out_%j.out
#SBATCH --error=eo_%j.err

cmd="python train_resLSTM_h5loader.py --config-name=config_AdamW_debug"

cd /global/cfs/cdirs/m4334/jerry/climsim3_dev/baseline_models/resLSTM/training_jerry
srun -n $SLURM_NTASKS shifter bash -c "source ddp_export.sh && $cmd"