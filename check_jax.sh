#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --partition=debug
#SBATCH --account=sd31
#SBATCH --constraint=gpu
#SBATCH --time=00:29:00


conda activate /scratch/snx3000/sdirmeie/PROJECTS/wgan/workdir/envs/gan-dev

srun nvidia-smi
srun python check_jax.py
