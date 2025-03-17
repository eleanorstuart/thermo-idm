#!/bin/bash
#SBATCH --account=kmpardo_1230
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_outputs/%j.out
module purge
eval "$(conda shell.bash hook)"
conda activate idm
python /home1/stuarte/krisp/thermo-idm/scripts/mcmc_script_choose_dataset.py 0.001 64 40000 "/home1/stuarte/krisp/thermo-idm/data/sp_equilibrium_clusters.pkl"