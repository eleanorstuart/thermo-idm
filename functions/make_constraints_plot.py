import numpy as np
from pyparsing import line
import emcee
import corner
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.ticker import MultipleLocator, FixedLocator
from astropy import units as u
from astropy import constants as const
import time
import pickle
import scipy
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple#, HandlerTupleVertical


with open('data/prior_boundary_points.pkl', 'rb') as f:
    data = pickle.load(f)

# Unpack the dictionary into variables
mchi_values = data['mchi']
s0_max = data['max_s0']


def plot_constraint(fig, mcmc_output, color, label, shaded=False, lines=True, fx=1):
    if lines:
        line_alpha=1
    else:
        line_alpha=0
    # plot equal mass case
    samples_coords = np.array([[x, y] for y, x in mcmc_output])

    # Reshape to get a 2D array of samples
    x_samples = samples_coords[:,0]
    y_samples = samples_coords[:,1]

    # Create a 2D histogram
    nbins = 50
    H, xedges, yedges = np.histogram2d(x_samples, y_samples, bins=nbins, density=True)

    # Apply Gaussian smoothing
    sigma = 3.5
    H_smooth = scipy.ndimage.gaussian_filter(H, sigma=sigma)

    # Compute the bin centers
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    # Create a meshgrid for the centers
    X, Y = np.meshgrid(xcenters, ycenters)

    # Flatten the smoothed histogram and sort the values
    H_smooth_flat = H_smooth.flatten()
    H_smooth_sorted = np.sort(H_smooth_flat)

    # Calculate the cumulative distribution function (CDF)
    cdf = np.cumsum(H_smooth_sorted)
    cdf /= cdf[-1]  # Normalize


    # Find the 95th percentile value
    percentile_value = np.percentile(H_smooth, 5)

    # Generate the contour plot
    #plt.figure(dpi=450)#(figsize=(8, 6))
    contour = fig.contour(X, Y, H_smooth.T, 
                          levels=[percentile_value],
                          colors=color, 
                          linewidths=1.5, alpha=line_alpha)


    path = contour.collections[0].get_paths()[0]
    v = path.vertices
    x = v[:,0]
    y = v[:,1]
    print(f'fx={fx} avg s0: ', 10**np.mean(y))

    #y2=-24
    fig.plot(x, y, label=label, color=color, alpha=line_alpha)
    if shaded:
        y2 = np.interp(x, np.log10(mchi_values), np.log10(s0_max/fx))
        fig.plot(np.log10(mchi_values), np.log10(s0_max/fx), color=color, linewidth=1.5, alpha=line_alpha)
        fig.fill_between(x, y, y2, where=(y2 >= y), interpolate=True, color=color, alpha=0.4,)
