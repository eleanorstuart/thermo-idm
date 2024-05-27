#!/bin/bash
#SBATCH --account=kmpardo_1230
#SBATCH --partition=main
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=8:00:00
module purge
eval "$(conda shell.bash hook)"
conda activate idm
python mcmc_script_10k.py