import numpy as np
from astropy import constants as const
from astropy import units as u

def chi_squared(model, data, variance): # take lists of model prediction, data, and variance of same length
    return sum((model[i]-data[i])**2/variance[i]**2 for i in range(len(data)))

def log_likelihood_1(p0, T_data, var, clusters, m_chi):
    #if p0[0]<0 or p0[1]<0:
    #   return -np.inf
    if p0[0]>0:
        return -np.inf
    T_model = [c.pred_T_b(p0, m_chi) for c in clusters]
    X2 = chi_squared(T_model, T_data, var)
    return (-X2/2)

def log_likelihood(p0, data, var, clusters, pred_func: str, m_chi, n=0): #pred_func is the name of one of the pred_ methods
    #commented out stuff was for temperature with 2-param
    #if p0[0]>0 or p0[0]<-30:
    #    return -np.inf
    #if p0[1]<-10 or 10**p0[1]>const.m_p.to(u.GeV, equivalencies=u.mass_energy()).value:
    #    return -np.inf
    #m_chi=-2
    #if p0>-10 or p0<-30:
    if p0>0 or p0<-30:
        return -np.inf
    model = [getattr(c, pred_func)(p0, m_chi, n) for c in clusters]
    X2 = chi_squared(model, data, var)
    #print(p0)
    print(p0, X2)

    return (-X2/2)

def log_likelihood_2(p0, data, var, clusters, pred_func: str): #pred_func is the name of one of the pred_ methods
    if p0[0]>-5 or p0[0]<-40:
        return -np.inf
    if p0[1]<-20 or 10**p0[1]>const.m_p.to(u.GeV, equivalencies=u.mass_energy()).value:
        return -np.inf

    #if p0>-20 or p0<-60:
    #    return -np.inf
    model = [getattr(c, pred_func)(p0) for c in clusters]
    X2 = chi_squared(model, data, var)
    print(p0)
    return (-X2/2)