import pandas as pd
import numpy as np
from astropy.table import QTable
from astropy import units as u

from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties import unumpy as unp

from cluster import Cluster

def load_clusters(nrows=None):
    skipfooter=0 if nrows else 1
    mcxccls=pd.read_csv('data/mcxc|.txt', header=3, sep='|', skiprows=[4],on_bad_lines='warn', skipfooter=skipfooter, nrows=nrows)
    mcxccls.columns=mcxccls.columns.str.strip()
    cls_data={'M500':mcxccls['M500'],
          'L500':mcxccls['L500'],
          'R500':mcxccls['R500']
         }
    units={
        'M500': 1e14*u.Msun,
        'L500': 1e37*u.W,
        'R500':u.Mpc
    }

    cls_table=QTable(cls_data, units=units)

    return [ 
        Cluster(
            cls_table['R500'][i],
            cls_table['M500'][i],
            L500=cls_table['L500'][i],
            m500=cls_table['M500'][i],
        )
        for i in range(mcxccls.shape[0])
    ], [variance(l) for l in cls_table['L500']] 

def variance(luminosity): # returns the variance on calculated temperature
     # TODO: add variance on luminosity
    #logL=np.log10((luminosity.to(1e44*u.erg/u.s))/(1e44*u.erg/u.s))

    A=ufloat(2.88, 0.15)
    B=ufloat(45.06, 0.03)

    lum = luminosity.to(u.erg / u.s)
    log_T = (np.log10(lum.value) - B) / A + np.log10(6)
    #T = (10**log_T * u.keV).to(u.GeV)

    T = 10**log_T * 1e-6 #divide by 10^6 to convert keV to GeV

    return T.std_dev *u.GeV #variances on T in GeV

    #return 0.1206 * np.sqrt(
    #    10 ** (-0.6944 * (45.06 - logL))
    #    * (8721 - 387.0 * logL + 4.295 * logL**2)
    #) #returns variance in Kelvin (needs to be checked)