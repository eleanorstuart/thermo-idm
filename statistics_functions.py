import numpy as np
from astropy import constants as const
from astropy import units as u

def chi_squared(T_model, T_data, variance): # take lists of model prediction, data, and variance of same length
    chi_squared_sum = 0
    for i in range(len(T_model)):
        chi_squared_sum+=(T_model[i]-T_data[i])**2/variance[i]**2
    return chi_squared_sum


# def log_likelihood(p0, m_chi, T_data, var, clusters):
#     #if p0[0]<0 or p0[1]<0:
#     #   return -np.inf
#     T_model = [c.pred_T_b_small_m(p0, m_chi) for c in clusters]
#     X2 = chi_squared(T_model, T_data, var)
#     return (-X2/2)

def log_likelihood_1(p0, T_data, var, clusters, m_chi):
    #if p0[0]<0 or p0[1]<0:
    #   return -np.inf
    if p0[0]>0:
        return -np.inf
    T_model = [c.pred_T_b(p0, m_chi) for c in clusters]
    X2 = chi_squared(T_model, T_data, var)
    return (-X2/2)

def log_likelihood(p0, T_data, var, clusters):
    #if p0[0]<0 or p0[1]<0:
    #   return -np.inf
    if p0[0]>0:
        return -np.inf
    if p0[1]<-10:
        return -np.inf
    if 10**p0[1]>const.m_p.to(u.GeV, equivalencies=u.mass_energy()).value:
        return -np.inf

    print(p0)
    T_model = [c.pred_T_b(p0) for c in clusters]
    X2 = chi_squared(T_model, T_data, var)
    return (-X2/2)