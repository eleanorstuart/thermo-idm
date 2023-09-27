# thermo-idm

To use the environment.yml file: 

`conda env create -f environment.yml` 

`conda activate idm` 

(currently out of date as of 09/13/2023 because the MCMC packages have not been added)

Solution to `RuntumeError: Failed to process string with tex because latex could not be found`:

Check to see if latex is installed in the anaconda environment: `latex --version`

If latex is not installed, what worked for me on my Mac was downloading MacTex from tug.org/mactex (not sure if this step was necessary)

Then in my environment I ran `conda install -c conda-forge texlive-core` and `latex --version` and I had information about my latex installation this time.

Then I restarted my anaconda environment and the error was resolved.