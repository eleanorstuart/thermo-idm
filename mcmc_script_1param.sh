#!/bin/bash
#SBATCH --account=kmpardo_1230
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=16:00:00
module purge
eval "$(conda shell.bash hook)"
conda activate idm
python mcmc_script_nsteps_1param.py 0.001 4 10000 0.01