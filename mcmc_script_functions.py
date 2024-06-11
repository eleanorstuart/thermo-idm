import numpy as np
import corner
import datetime

from load_data_and_var import load_data_and_var, generate_nfws_from_combined_dataset_werr


def create_cluster_objects(rmin):
    # create objects
    dataset, variances = load_data_and_var()
    nfws_werr = generate_nfws_from_combined_dataset_werr(dataset, variances, rmin=rmin)
    print('created objects')

    # grab only low z clusters
    nfws_subset = [nfw for nfw in nfws_werr if nfw.z < 0.2]
    L_obs = np.array([nfw.L500.value for nfw in nfws_subset])
    variances_obs = np.array([nfw.L_err for nfw in nfws_subset])

    return nfws_subset, L_obs, variances_obs

def create_initial_state(nwalkers, ndim):
    initial=np.zeros((nwalkers, ndim))

    # initialize walkers
    log_sigma0 = np.random.uniform(low=-40, high=-20, size=nwalkers).astype(np.float128)
    log_mx = np.random.uniform(low=-4, high=-1, size=nwalkers).astype(np.float128)

    initial[:,0]=log_sigma0
    initial[:,1]=log_mx

    return initial

def plot_and_save_results(sampler, rmin, nwalkers, nsteps):

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
    with open(f'pickled/mcmc_{rmin}_{nwalkers}_{nsteps}_{formatted_datetime}.pkl', 'wb') as f:
        pickle.dump(flat_samples, f)