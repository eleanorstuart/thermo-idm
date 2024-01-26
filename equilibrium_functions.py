import numpy as np
from astropy import units as u
from astropy import constants as const
#from scipy.optimize import brentq
#from statistics_functions import chi_squared
from cluster_functions import c

def radiative_cooling_rate(T_b, measurements, n_e):
    prefactors=6.8*1e-42 *u.erg*u.cm**3
    Z=1
    T=T_b.to(u.K, equivalencies=u.temperature_energy())#self.baryon_temp.to(u.K, equivalencies=u.temperature_energy())
    T8=T/(1e8*u.K)

    #C=(prefactors*Z**2*(measurements.n_e.to(u.cm**-3))**2)/(T8**(1/2))
    C=(prefactors*Z**2*(n_e.to(u.cm**-3))**2)/(T8**(1/2))
    Eff_int = (C*T*const.k_B/const.h).to(u.GeV/(u.s*u.cm**3))
    return (measurements.volume*Eff_int).to(u.erg/u.s)

def agn_heating_rate(T_b, cluster):
    with u.set_enabled_equivalencies(u.mass_energy()):
        return (cluster.epsilon * accretion_rate(T_b, cluster)).to(
            u.erg / u.s, equivalencies=u.temperature_energy())

def accretion_rate(T_b, cluster):
    with u.set_enabled_equivalencies(u.mass_energy()):
        leading_factors = cluster.norm * 4 * np.pi * const.c**-3
        gm2 = (const.G * cluster.bh_mass) ** 2
        frac = (cluster.mu * cluster.m_b) ** (5 / 2) / cluster.adiabatic_idx ** (3 / 2)
        return leading_factors * gm2 * frac * plasma_entropy(T_b, cluster) ** (-3 / 2)

def plasma_entropy(T_b, cluster):
    baryon_number_density = (2 * cluster.measurements.n_e).to(u.m ** (-3))
    return (const.k_B * T_b.to(u.K, equivalencies=u.temperature_energy())
        ).to(u.GeV) / baryon_number_density ** (cluster.adiabatic_idx - 1)

def dm_cooling_rate(T_b, cluster, s0, m_chi, n=0, f_chi=1, m_psi=0.1*u.GeV):
    dm_temp = cluster.virial_temperature(m_chi, f_chi=f_chi, m_psi=m_psi)
    uth = np.sqrt(T_b / cluster.m_b + dm_temp / m_chi)
    rho_chi = cluster.rho_dm * f_chi

    denominator = (m_chi + cluster.m_b) ** 2
    numerator = (
            3
            * (T_b - dm_temp)
            * rho_chi
            * cluster.rho_b
            * cluster.measurements.volume.to(u.cm**3)
            * c(n)
            * uth ** (n + 1)
            * (const.c.to(u.cm / u.s))
            *s0
        
        )
    #conversion_factor = 0.197*1e-15 * (u.GeV * u.m)
    #print("dm cooling rate", (numerator / denominator).to(u.erg/u.s))
    return (numerator / denominator).to(u.erg/u.s)

def equil(logT_b, cluster, sig_0, m_chi, heating='effervescent'):
    #divide agn_heating_rate by 1e5 to put it at the same oom as the cooling
    s0=10**sig_0 * u.cm**2
    mx=10**m_chi *u.GeV
    T_b=10**(logT_b)*u.GeV

    if heating=='effervescent':
        heating_rate=cluster.eff_agn_heating_rate 
    else:
        print('using old agn model')
        heating_rate=(agn_heating_rate(T_b, cluster)/1e5).to(u.erg/u.s) 
    return (heating_rate 
        - dm_cooling_rate(T_b, cluster, s0, mx) 
        - radiative_cooling_rate(T_b, cluster)).value

#def log_lik(p0, T_data, var, clusters, m_chi):
#    if p0<-50 or p0>0:
#        return -np.inf
#    T_model = [brentq(equil, -15, 0, args=(c, p0)) for c in clusters]
#    X2 = chi_squared(np.power(10,T_model)*u.GeV, T_data, var)
#    return (-X2/2)