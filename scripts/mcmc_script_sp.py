#from pdb import run
import numpy as np
import emcee
import corner
#from astropy import units as u
#from astropy import constants as const
import datetime

import argparse
from multiprocessing import Pool

import sys
import os
sys.path.append('..')
from functions.statistics_functions import log_lik
#from mcmc_script_functions import create_cluster_objects, create_initial_state, plot_and_save_results
import pickle



def run_mcmc():
    parser = argparse.ArgumentParser()
    parser.add_argument('rmin', type=float, help='lower bound for integration when calculating heating and cooling rates')
    parser.add_argument('nwalkers', type=int, help='number of walkers')
    parser.add_argument('nsteps', type=int, help='number of steps chains should run')
    parser.add_argument('fx', type=float, help='fraction of idm')
    parser.add_argument('mpsi', type=float, help='mass of noninteracting particle')
    args = parser.parse_args()
    rmin = args.rmin
    nwalkers = args.nwalkers
    nsteps = args.nsteps
    fx = args.fx
    mpsi = args.mpsi if args.mpsi else None

    # Open the data file
    with open('data/virialized_nfws_sp.pkl', 'rb') as file:
        # Load the data into nfws_subset variable
        nfws_subset = pickle.load(file)

    L_obs = [n.L500 for n in nfws_subset]
    variances_obs = np.array([n.L_err for n in nfws_subset])


    # prepare MCMC

    ndim = 2
    initial=np.zeros((nwalkers, ndim ))

    # initialize walkers
    log_sigma0 = np.random.uniform(low=-40, high=-23, size=nwalkers).astype(np.float128)
    #log_mp = np.log10(const.m_p.to(u.GeV, equivalencies=u.mass_energy()).value)
    log_mx = np.random.uniform(low=-4, high=-1, size=nwalkers).astype(np.float128)

    initial[:,0]=log_sigma0
    initial[:,1]=log_mx


    # burn in
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers,
                                ndim,
                                log_lik,
                                args=[L_obs, nfws_subset, variances_obs, fx, mpsi])
        state = sampler.run_mcmc(initial, 1000, progress=True)
        sampler.reset()

    #production run
    with Pool() as pool:
        #start sampler after some steps to minimize dependence on initial conditions
        state_f = sampler.run_mcmc(state, nsteps, progress = True)


    # create corner plot of results
    flat_samples = sampler.get_chain(thin=15, flat=True)

    print(flat_samples.shape)

    labels = [r"$\mathrm{log} (\sigma_0 / \mathrm{cm^2})$", r"$\mathrm{log} (m_\chi / \mathrm{GeV})$"]
    fig = corner.corner(
        flat_samples, labels=labels
    )
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    fig.savefig(f'plots/mcmc_corner_plot_{rmin}_{nwalkers}_{nsteps}_fx{fx}_mpsi{mpsi}_{formatted_datetime}.pdf', format='pdf', dpi=450)

    with open(f'pickled/mcmc_{rmin}_{nwalkers}_{nsteps}_fx{fx}_mpsi{mpsi}_{formatted_datetime}.pkl', 'wb') as f:
        pickle.dump(flat_samples, f)

    return


run_mcmc()