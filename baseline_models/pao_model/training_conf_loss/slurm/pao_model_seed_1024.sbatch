#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 8:30:00
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH -n 4
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH --mail-type=ALL
##SBATCH --output=out_%j.out
##SBATCH --error=eo_%j.err

cmd="python train_pao_model_conf_loss.py --config-name=config_seed_1024"

cd ..
srun -n $SLURM_NTASKS shifter bash -c "source ddp_export.sh && $cmd"