import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.optimize import brentq

from equilibrium_functions import equil

def chi_squared(model, data, variance, add_var=0): # take lists of model prediction, data, and variance of same length
    # TODO: add second source of variance
    return sum((model[i]-data[i].value)**2/(variance[i].value**2 + add_var[i]**2) for i in range(len(data)))


# log likelihood function for MCMC
def log_lik(p0, obs, variances, nfws, multagn=1):
    log_s0 = p0[0]
    log_mx = p0[1]

    # prior
    if log_s0>0 or log_s0<-40:
        return -np.inf
    # DM particle mass is limited to being less than a proton mass
    if log_mx<-4 or 10**log_mx>const.m_p.to(u.GeV, equivalencies=u.mass_energy()).value:
        return -np.inf

    # calculate model predicted L500
    predicted_werr = [nfw.L500_analytic(log_s0, log_mx, mult_agn=multagn) for nfw in nfws]
    # TODO: take the error bars off of predicted and add them to variances
    variances_pred = [max(p.plus, p.minus) for p in predicted_werr]

    predicted = [p.value for p in predicted_werr]
    X2 = chi_squared(predicted, obs, variances, add_var=variances_pred)
    return -X2/2


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


def log_lik_eq(p0, T_data, var, clusters):
    if p0[0]<-40 or p0[0]>-10:
        return -np.inf
    if p0[1]<-3 or 10**p0[1]>const.m_p.to(u.GeV, equivalencies=u.mass_energy()).value:
        return -np.inf
    #print(p0)
    T_model = [brentq(equil, -10, 0, args=(c, p0[0], p0[1])) for c in clusters]
    X2 = chi_squared(np.power(10,T_model)*u.GeV, T_data, var)
    return (-X2/2)