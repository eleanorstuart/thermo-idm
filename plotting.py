"""
useful plotting functions
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import astropy.units as u
import scipy.optimize
from datetime import datetime


def savefig(fig, figpath, writepdf=False, dpi=450):
    fig.savefig(figpath, dpi=dpi, bbox_inches="tight")
    print("{}: made {}".format(datetime.now().isoformat(), figpath))

    if writepdf:
        pdffigpath = figpath.replace(".png", ".pdf")
        fig.savefig(pdffigpath, bbox_inches="tight", rasterized=True, dpi=dpi)
        print("{}: made {}".format(datetime.now().isoformat(), pdffigpath))

    plt.close("all")


def paper_plot():
    sns.set_context("paper")
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    plt.rc("font", family="serif", serif="cm10")
    figparams = {
        "text.latex.preamble": r"\usepackage{amsmath} \boldmath",
        "text.usetex": True,
        "axes.labelsize": 20.0,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.figsize": [6.0, 4.0],
        "font.family": "DejaVu Sans",
        "legend.fontsize": 12,
    }
    plt.rcParams.update(figparams)
    cs = plt.rcParams["axes.prop_cycle"].by_key()["color"]
