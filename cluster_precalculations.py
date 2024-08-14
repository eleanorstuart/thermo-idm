import numpy as np

from astropy import units as u
from astropy import constants as const

from asymmetric_uncertainty_mod import a_u
from cluster_functions import c

# model consts
h=0.7

# conversions
GeV_to_erg = (u.GeV/u.erg).to(1)

# use this class to calculate things that I will need for MCMC
class PreCalculations:
    def __init__(self, cluster):
        self.cluster = cluster
        M500 = cluster.M500
        R500 = cluster.R500
        c500 = cluster.c500

        M500_min = cluster.M500.value - cluster.M500.minus
        M500_max = cluster.M500.value + cluster.M500.plus

        # calculate minimum heating and cooling factors for the cluster
        R500_min = ((M500_min/(4*np.pi/3 * 500. * (cluster.rho_c).to(u.Msun/u.Mpc**3).value))**(1./3.))
        r0_min = 0.001*R500_min
        rc_min = 0.1*R500_min

        self.cluster.M500 = M500_min
        self.cluster.R500 = a_u(R500_min)
        self.cluster.c500 = (5.71*(M500_min/(2*1e12 * h**-1))**(-0.084) * (1+cluster.z)**(-0.47))

        self.agn_factors_min = (cluster.total_heating_rate(a_u(r0_min), 
                                                           a_u(R500_min), 
                                                           rc_min, 
                                                           Linj = 1*u.erg/u.s, 
                                                           n=50)).to(u.erg/u.s).value.value

        xs = np.logspace(-3, 0, num=50)
        self.rs_min = xs*self.cluster.R500

        Tbs_min = np.asarray(self.cluster.T_g(self.rs_min, match_gaspari=False))
        self.Tbs_norm_min = np.array([i.value for i in Tbs_min/(Tbs_min[-1]*u.GeV.to(u.K, equivalencies=u.temperature_energy()))]) # units 1/K

        self.rad_factors_min = cluster.total_cooling_rate(a_u(r0_min), 
                                                          a_u(R500_min), 
                                                          vol_rate='gfe', 
                                                          Tb=self.Tbs_norm_min).value
        
        self.L_indep_factors_min = np.array([i.value for i in np.array(self.D(self.rs_min)*(u.Mpc/u.cm).to(1)**3)/GeV_to_erg])
        self.Tx_by_mx_min = np.asarray([t.value for t in cluster.Tx_over_mx(self.rs_min)])

        # calculate maximum heating and cooling factors for the cluster
        R500_max = ((M500_max/(4*np.pi/3 * 500. * (cluster.rho_c).to(u.Msun/u.Mpc**3).value))**(1./3.))
        r0_max = 0.001*R500_max
        rc_max = 0.1*R500_max

        self.cluster.M500 = M500_max
        self.cluster.R500 = a_u(R500_max)
        self.cluster.c500 = (5.71*(M500_max/(2*1e12 * h**-1))**(-0.084) * (1+cluster.z)**(-0.47))

        self.agn_factors_max = (cluster.total_heating_rate(a_u(r0_max), 
                                                           a_u(R500_max), 
                                                           rc_max, 
                                                           Linj = 1*u.erg/u.s, 
                                                           n=50)).to(u.erg/u.s).value.value

        self.rs_max = xs*self.cluster.R500 #np.array([r.value for r in rs])

        Tbs_max = np.asarray(self.cluster.T_g(self.rs_max, match_gaspari=False))
        self.Tbs_norm_max = np.array([i.value for i in Tbs_max/(Tbs_max[-1]*u.GeV.to(u.K, equivalencies=u.temperature_energy()))]) # units 1/K

        self.rad_factors_max = cluster.total_cooling_rate(a_u(r0_max), 
                                                          a_u(R500_max), 
                                                          vol_rate='gfe', 
                                                          Tb=self.Tbs_norm_max).value
        
        self.L_indep_factors_max = np.array([i.value for i in np.array(self.D(self.rs_max)*(u.Mpc/u.cm).to(1)**3)/GeV_to_erg])
        self.Tx_by_mx_max = np.asarray([t.value for t in cluster.Tx_over_mx(self.rs_max)])

        # restore original values
        self.cluster.M500 = M500 
        self.cluster.R500 = R500 
        self.cluster.c500 = c500

        



    def D(self,r): # calculates the part of DM cooling that isnt dependent on L500 or sigma0 or mx
            f_chi=1
            rho_chi = self.cluster.rho_tot(r) * f_chi
            conv_dens = (u.Msun/u.Mpc**3).to(u.erg/u.cm**3)
            
            num = (
                3 * rho_chi*conv_dens#.to(u.GeV/u.cm**3)
                * self.cluster.rho_g(r)*conv_dens#.to(u.GeV/u.cm**3, equivalencies=u.mass_energy())
                * c(0)
                * (const.c.to(u.cm / u.s)).value
                #* s0
                * 4 * np.pi * np.power(r, 2)
            )
            #denom= ((mx + const.m_p.to(u.GeV).value) ** 2)
            #return np.divide(num, denom)
            return num

