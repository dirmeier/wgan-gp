#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --account=sd31
#SBATCH --time=00:29:00
#SBATCH --output=/scratch/snx3000/sdirmeie/PROJECTS/wgan/workdir/slurm/%j.out
#SBATCH --error=/scratch/snx3000/sdirmeie/PROJECTS/wgan/workdir/slurm/%j.out

module load daint-gpu
conda activate /scratch/snx3000/sdirmeie/PROJECTS/wgan/workdir/envs/gan-dev


srun python main.py \
 --workdir=/scratch/snx3000/sdirmeie/PROJECTS/wgan/workdir/ \
 --config=config.py \
 --config.training.n_steps=2 \
 --config.training.n_eval_frequency=2 \
 --config.training.n_checkpoint_frequency=2 \
 --config.training.n_sampling_frequency=2 \
 --usewand
