import pickle
import pandas as pd
import numpy as np
from astropy.table import QTable
from astropy import units as u
import astropy.cosmology.units as cu

from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties import unumpy as unp

from cluster import Cluster
from cluster_measurements import ClusterMeasurements

def load_clusters(nrows=None, dataset='REFLEX'):
    skipfooter=0 if nrows else 1
    mcxccls=pd.read_csv('data/mcxc|.txt', header=3, sep='|', skiprows=[4],on_bad_lines='warn', skipfooter=skipfooter)
    mcxccls.columns=mcxccls.columns.str.strip()

    selected_cls=mcxccls[mcxccls['Sub-Cat'].str.strip()==dataset]

    cls_data={'M500':selected_cls['M500'],
          'L500':selected_cls['L500'],
          'R500':selected_cls['R500'],
          'z':selected_cls['R500']
         }
    units={
        'M500': 1e14*u.Msun,
        'L500': 1e37*u.W,
        'R500':u.Mpc,
        'z': cu.redshift
    }

    cls_table=QTable(cls_data, units=units)
    
    with open ('data/'+dataset+'.p', 'rb') as fp:
        L_uncertainties=np.array(pickle.load(fp))
    L_uncertainties_conv=(L_uncertainties*1e37*u.W).to(u.erg / u.s).value
    
    cluster_measurements = [ClusterMeasurements(
        cls_table['R500'][i], 
        cls_table['M500'][i],
        cls_table['z'][i],
        cls_table['L500'][i], 
        T_var = variance(cls_table['L500'][i], L_uncertainties_conv[i])) for i in range(nrows)]

    #return [Cluster(cm)for cm in cluster_measurements], [variance(cls_table['L500'][i], L_uncertainties_conv[i]) for i in range(nrows)] 
    return [Cluster(cm)for cm in cluster_measurements]

def variance(luminosity, l_unc): # returns the variance on calculated temperature
     # TODO: TEST THIS NOW THAT I HAVE ADDED L_unc
    #logL=np.log10((luminosity.to(1e44*u.erg/u.s))/(1e44*u.erg/u.s))

    A=ufloat(2.88, 0.15)
    B=ufloat(45.06, 0.03)

    lum = luminosity.to(u.erg / u.s)
    
    lums_uf = ufloat(lum.value, l_unc)
    log_T = (unp.log10(lums_uf) - B) / A + np.log10(6)
    #T = (10**log_T * u.keV).to(u.GeV)

    T = 10**log_T * 1e-6 #divide by 10^6 to convert keV to GeV

    return T.std_dev *u.GeV #variances on T in GeV #variances on T in GeV

    #return 0.1206 * np.sqrt(
    #    10 ** (-0.6944 * (45.06 - logL))
    #    * (8721 - 387.0 * logL + 4.295 * logL**2)
    #) #returns variance in Kelvin (needs to be checked)