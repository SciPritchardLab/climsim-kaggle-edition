#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu
<<<<<<< HEAD
<<<<<<< HEAD
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH -n 4
=======
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH -n 1
>>>>>>> 2ad914d (changes to config files and sbatches)
=======
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH -n 4
>>>>>>> 3fb9a4f (added non-debugging config files and batch scripts)
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --mail-user=frieldskatherine@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=out_%j.out
#SBATCH --error=eo_%j.err

<<<<<<< HEAD
<<<<<<< HEAD
cmd="python train_unet_h5loader_processed.py --config-name=config_adam"
=======
cmd="python train_unet_h5loader_processed.py --config-name=config_adam_debugging"
>>>>>>> 2ad914d (changes to config files and sbatches)
=======
cmd="python train_unet_h5loader_processed.py --config-name=config_adam"
>>>>>>> 3fb9a4f (added non-debugging config files and batch scripts)

cd /global/cfs/cdirs/m4334/kfrields/climsim3/baseline_models/unet/training_katherine
srun -n $SLURM_NTASKS shifter bash -c "source ddp_export.sh && $cmd"
