#!/bin/bash
#SBATCH --account=kmpardo_1230
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=16:00:00
module purge
eval "$(conda shell.bash hook)"
conda activate idm
python mcmc_script_10k.py 0.0001