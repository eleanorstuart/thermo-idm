#!/bin/bash
#SBATCH --account=kmpardo_1230
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
module purge
eval "$(conda shell.bash hook)"
conda activate idm
python mcmc_script.py