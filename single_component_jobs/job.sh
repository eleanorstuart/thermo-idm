#!/bin/bash
#SBATCH --account=kmpardo_1230
#SBATCH --partition=main
#SBATCH --nodes=8
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
module purge
eval "$(conda shell.bash hook)"
conda activate idm
python mcmc_script_10k.py 0.001