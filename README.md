# thermo-idm
##Notebooks: 
- constraint_from_total_energy_avg.ipynb: Shows the calculation of the constraint on DM-proton interactions derived without assuming anything about additional heating/cooling mechanisms
- cooling_comparison.ipynb: Calculates what fraction of the cooling in a cluster would come from DM cooling
- generate_dataset.ipynb: Combines REFLEX and Planck data and saves the low z clusters
- generate_equilibrium_plots.ipynb: Generates figure 1 and the set of 49 clusters used for analysis
- interaction_rate_integrated.ipynb: Generates prior based on the requirement that DM and baryons not thermalize over a cluster's lifetime
- load_mcmc_output-equil.ipynb: MCMC output from 49 cluster set
- load_mcmc_output.ipynb: MCMC output from 111 cluster set
- load_mcmc_output_fidm.ipynb: Generates figure 2, output for 2-component DM
- load_mcmc_output_fidm_unequal_masses.ipynb: Output for 2-component DM with a lower-mass noninteracting species
- reflex_subset_error_mcmc.ipynb: Error propagation for L500


##To use the environment.yml file: 

`conda env create -f environment.yml` 

`conda activate idm` 

(environment.yml updated 12/01/2024)

Solution to `RuntumeError: Failed to process string with tex because latex could not be found`:

Check to see if latex is installed in the anaconda environment: `latex --version`

If latex is not installed, what worked for me on my Mac was downloading MacTex from tug.org/mactex (not sure if this step was necessary)

Then in my environment I ran `conda install -c conda-forge texlive-core` and `latex --version` and I had information about my latex installation this time.

Then I restarted my anaconda environment and the error was resolved.
