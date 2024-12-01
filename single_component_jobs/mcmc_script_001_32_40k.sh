#!/bin/bash
#SBATCH --account=kmpardo_1230
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_outputs/%j.out
module purge
eval "$(conda shell.bash hook)"
conda activate idm
python mcmc_script_nsteps.py 0.001 32 40000
