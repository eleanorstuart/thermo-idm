#!/bin/bash
#SBATCH --account=<project_id>
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=8:00:00
module purge
eval "$(conda shell.bash hook)"
conda activate idm
python mcmc_script.py