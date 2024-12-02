import numpy as np
import emcee
import corner
from astropy import units as u
from astropy import constants as const
import datetime

import argparse
from multiprocessing import Pool

from statistics_functions import log_lik
from old_py_files.mcmc_script_functions import create_cluster_objects, create_initial_state, plot_and_save_results



def run_mcmc():
    parser = argparse.ArgumentParser()
    parser.add_argument('rmin', type=float, help='lower bound for integration when calculating heating and cooling rates')
    parser.add_argument('nwalkers', type=int, help='number of walkers')
    parser.add_argument('nsteps', type=int, help='number of steps chains should run')
    parser.add_argument('agn_model', type=str, help='iqbal or eddington')
    parser.add_argument('mult_factor', type=float, help='multiply factor for agn')
    args = parser.parse_args()

    rmin = args.rmin
    nwalkers = args.nwalkers
    nsteps = args.nsteps
    agn_model=args.agn_model
    mult_factor = args.mult_factor



    nfws_subset, L_obs, variances_obs = create_cluster_objects(rmin)

    # prepare MCMC
    ndim=2
    initial=create_initial_state(nwalkers, ndim)

    # burn in
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers,
                                ndim,
                                log_lik,
                                args=[L_obs, nfws_subset, variances_obs, mult_factor, agn_model])
        state = sampler.run_mcmc(initial, 1000, progress=True)
        sampler.reset()

    #production run
    with Pool() as pool:
        #start sampler after some steps to minimize dependence on initial conditions
        state_f = sampler.run_mcmc(state, nsteps, progress = True)

    plot_and_save_results(sampler, args)
    return




run_mcmc()