import numpy as np
import emcee
import corner
from astropy import units as u
from astropy import constants as const
import datetime

import argparse
from multiprocessing import Pool

from statistics_functions import log_lik
from load_data_and_var import load_data_and_var, generate_nfws_from_combined_dataset_werr



def run_mcmc():
    parser = argparse.ArgumentParser()
    parser.add_argument('rmin', type=float, help='lower bound for integration when calculating heating and cooling rates')
    parser.add_argument('nwalkers', type=int, help='number of walkers')
    parser.add_argument('nsteps', type=int, help='number of steps chains should run')
    args = parser.parse_args()
    rmin = args.rmin
    nwalkers = args.nwalkers
    nsteps = args.nsteps


    # create objects
    dataset, variances = load_data_and_var()
    nfws_werr = generate_nfws_from_combined_dataset_werr(dataset, variances, rmin=rmin)
    print('created objects')

    # grab only low z clusters
    nfws_subset = [nfw for nfw in nfws_werr if nfw.z < 0.2]
    L_obs = np.array([nfw.L500.value for nfw in nfws_subset])
    variances_obs = np.array([nfw.L_err for nfw in nfws_subset])

    # prepare MCMC

    ndim = 2
    initial=np.zeros((nwalkers, ndim ))

    # initialize walkers
    log_sigma0 = np.random.uniform(low=-40, high=-20, size=nwalkers).astype(np.float128)
    #log_mp = np.log10(const.m_p.to(u.GeV, equivalencies=u.mass_energy()).value)
    log_mx = np.random.uniform(low=-4, high=-1, size=nwalkers).astype(np.float128)

    initial[:,0]=log_sigma0
    initial[:,1]=log_mx


    # burn in
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers,
                                ndim,
                                log_lik,
                                args=[L_obs, nfws_subset, variances_obs, 1])
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

    fig.savefig(f'plots/mcmc_corner_plot_{rmin}_{nwalkers}_{nsteps}_{formatted_datetime}.pdf', format='pdf', dpi=450)

    import pickle
    with open('pickled/mcmc_{rmin}_{nwalkers}_{nsteps}_{formatted_datetime}.pkl', 'wb') as f:
        pickle.dump(flat_samples, f)

    return




run_mcmc()