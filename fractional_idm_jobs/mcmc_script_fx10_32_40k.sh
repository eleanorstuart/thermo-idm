#!/bin/bash
#SBATCH --account=kmpardo_1230
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_outputs/%j.out
module purge
eval "$(conda shell.bash hook)"
conda activate idm
python mcmc_script_equilibrium.py 0.001 32 40000 0.1

