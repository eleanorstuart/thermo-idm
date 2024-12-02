import numpy as np
from astropy import units as u
from astropy import constants as const
from scipy.special import gamma
u.set_enabled_equivalencies(u.mass_energy())


def c(n):
    return 2 ** (5 + n / 2) / (3 * np.sqrt(np.pi)) * gamma(3 + n / 2)


def temp_from_vdisp(vel_disp):
    return (vel_disp**2 * const.m_p / const.k_B).to(
        u.GeV, equivalencies=u.temperature_energy()
    )


def temp_from_luminosity(luminosity):
    lum = luminosity.to(u.erg / u.s)
    log_T = (np.log10(lum.value) - 45.06) / 2.88 + np.log10(6)
    T = np.power(10, log_T) * u.keV
    return T.to(u.GeV)



def Linj_from_L500(logL500, bound=None):
    # calculate Linj for a given L500
    if bound:
        if bound=='min': # minimize Linj by taking lowest b, m values
            b = 10 - 0.11
            m = 0.38 - 0.03
        elif bound=='max': # maximize  Linj by taking highest b, m values
            b = 10 + 0.11
            m = 0.38 + 0.03
    else:
        b = 10
        m = 0.38

    logMbh = b+m*(logL500)#/(1e44*u.erg/u.s))
    Mbh = np.power(10, logMbh)#*u.Msun
    return 1e44 * (Mbh/(np.power(10, 9.5)))#*u.Msun))

def T500_from_L500(logL500, bound=None): # logL=log(L/1e44erg/s)
    if bound:
        if bound=='max': # maximize T by taking lowest b, m values
            b = -2.34 - 0.09
            m = 4.71 - 0.26
        elif bound=='min': # minimize T by taking highest b, m values
            b = -2.34 + 0.09
            m = 4.71 + 0.26
    else:
        b = -2.34
        m = 4.71
    logT = (logL500 - b)/m
    #return logT
    return np.power(10, logT) * 1e-6 # in GeV

def L500_from_T500(T500):
    logT = np.log10(T500*1e6)
    return -2.34 + 4.71*logT # returns in 1e44 erg/s



def E_tot(nfw, L=None): # remove ".value" once ive gotten units figured out
    G_conv = (const.G/const.c**2).to(u.Mpc/u.Msun)
    pe = G_conv * (nfw.M500.value)**2*u.Msun**2/(nfw.R500.value*u.Mpc)
    
    if not L:
        L = nfw.L500.value
    xmin = 0.001
    xmax = 1
    xs = np.logspace(np.log10(xmin), np.log10(xmax), num=50)
    rs = xs*nfw.R500
    
    
    Tbs = np.array([T.value for T in np.asarray(nfw.T_g(rs, match_gaspari=True))]) # in GeV
    Tbs_norm = Tbs/Tbs[-1] # unitless
    Tbs_Ldep = Tbs_norm * T500_from_L500(np.log10(L))
    nbs = [n.value for n in np.asarray(nfw.n_e(rs))] # in 1/Mpc^3
    ke_b = (np.trapz(4*np.pi * rs**2 * 3./2* np.multiply(Tbs_Ldep,nbs),rs)).value*u.GeV
    
    avg_rho_x = nfw.M500.value/(4./3*np.pi*nfw.R500.value**3) # Msun/Mpc3
    #Tx_by_mx_float = [tm.value for tm in nfw.Tx_by_mx]
    ke_x = (np.trapz(4*np.pi*rs**2 * 3./2*avg_rho_x*nfw.Tx_by_mx, rs).value)*u.Msun
    #ke_x = 0 * u.erg
    return (pe+ke_b+ke_x).to(u.erg, equivalencies=u.mass_energy()).value