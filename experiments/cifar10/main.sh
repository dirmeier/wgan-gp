#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=sd31
#SBATCH --time=23:59:00
#SBATCH --output=/scratch/snx3000/sdirmeie/PROJECTS/wgan/workdir/slurm/%j.out
#SBATCH --error=/scratch/snx3000/sdirmeie/PROJECTS/wgan/workdir/slurm/%j.out

module load daint-gpu
conda activate /scratch/snx3000/sdirmeie/PROJECTS/wgan/workdir/envs/gan-dev


srun python main.py \
 --workdir=/scratch/snx3000/sdirmeie/PROJECTS/wgan/workdir/ \
 --config=config.py \
 --usewand
