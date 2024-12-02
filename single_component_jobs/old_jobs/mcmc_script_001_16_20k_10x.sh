#!/bin/bash
#SBATCH --account=kmpardo_1230
#SBATCH --partition=main
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
module purge
eval "$(conda shell.bash hook)"
conda activate idm
python mcmc_script_nsteps_10x.py 0.001 16 20000
