import numpy as np

from astropy import units as u
from astropy import constants as const

from scipy.optimize import brentq, root_scalar

from asymmetric_uncertainty_mod import a_u
from cluster_functions import c


# model consts
h=0.7

# conversions
GeV_to_erg = (u.GeV/u.erg).to(1)
mp_GeV = (const.m_p).to(u.GeV).value
GeV_to_K = (u.GeV).to(u.K, equivalencies=u.temperature_energy())

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


    def L500_numerical_sampled(self, M500, mx, s0, scaling_params):
        #measured_M500 = self.cluster.M500
        #measured_R500 = self.cluster.R500
        #measured_c500 = self.cluster.c500

        # set M500 to sampled value for calculations
        original_M500 = self.cluster.M500
        original_R500 = self.cluster.R500
        original_c500 = self.cluster.c500
        self.cluster.M500 = M500

        try:
            # calculate factors based on sampled properties
            # calculate minimum heating and cooling factors for the cluster
            R500 = ((M500/(4*np.pi/3 * 500. * (self.cluster.rho_c).to(u.Msun/u.Mpc**3).value))**(1./3.))
            r0 = 0.001*R500
            rc = 0.1*R500

            self.cluster.R500 = a_u(R500)
            self.cluster.c500 = (5.71*(M500/(2*1e12 * h**-1))**(-0.084) * (1+self.cluster.z)**(-0.47))

            self.agn_factors = (self.cluster.total_heating_rate(a_u(r0), 
                                       a_u(R500), 
                                       rc, 
                                       Linj = 1*u.erg/u.s, 
                                       n=50)).to(u.erg/u.s).value.value

            xs = np.logspace(-3, 0, num=50)
            self.rs = xs*self.cluster.R500

            Tbs = np.asarray(self.cluster.T_g(self.rs, match_gaspari=False))
            self.Tbs_norm = np.array([i.value for i in Tbs/(Tbs[-1]*u.GeV.to(u.K, equivalencies=u.temperature_energy()))]) # units 1/K

            self.rad_factors = self.cluster.total_cooling_rate(a_u(r0), 
                                      a_u(R500), 
                                      vol_rate='gfe', 
                                      Tb=self.Tbs_norm).value
            
            self.L_indep_factors = np.array([i.value for i in np.array(self.D(self.rs)*(u.Mpc/u.cm).to(1)**3)/GeV_to_erg])
            self.Tx_by_mx = np.asarray([t.value for t in self.cluster.Tx_over_mx(self.rs)])


            L_indep_factors_full = np.divide(self.L_indep_factors*s0,((mx + mp_GeV) ** 2))
            T_chi = np.array(self.Tx_by_mx*mx) # units GeV

        except Exception as e:
            print(f"Error occurred: {e}")
            raise

        finally:
            # restore original values
            self.cluster.M500 = original_M500
            self.cluster.R500 = original_R500
            self.cluster.c500 = original_c500


        # restore original values
        self.cluster.M500 = original_M500
        self.cluster.R500 = original_R500
        self.cluster.c500 = original_c500

        mLinj = scaling_params['mLinj']
        bLinj = scaling_params['bLinj']
        mT500 = scaling_params['mT500']
        bT500 = scaling_params['bT500']

        def equation(logL): # L in 1e44 erg/s
            #print("logL", logL)
            T500 = T500_from_L500_sampled(logL,mT500, bT500)
            #T500k = T500*GeV_to_K
            # Calculate T500 in logarithmic space to avoid overflow
            T500_log = np.log10(T500_from_L500_sampled(logL, mT500, bT500))
            T500k_log = T500_log + np.log10(GeV_to_K)
            T500k = 10**T500k_log
            Tbs = np.multiply(self.Tbs_norm, T500k)
            #print(Tbs)

            #cooling_rate=np.sqrt(T500k)*self.rad_factors
            # Calculate cooling rate in logarithmic space
            cooling_rate_log = 0.5 * T500k_log + np.log10(self.rad_factors)
            cooling_rate = 10**cooling_rate_log

            #uth = np.sqrt(Tbs/mp_GeV + self.Tx_by_mx)
            # Calculate uth in logarithmic space
            uth_log = 0.5 * np.log10(Tbs / mp_GeV + self.Tx_by_mx)
            uth = 10**uth_log
            integrand = (L_indep_factors_full*uth)*(Tbs-T_chi)
            dm_cooling_rate = np.trapz(integrand, [r.value for r in self.rs])

            Linj=Linj_from_L500_sampled(logL, mLinj, bLinj)
            heating_rate = Linj * self.agn_factors
            #print(heating_rate)
            print('L', logL, 'heating',heating_rate, 'cooling', cooling_rate, 'dm cooling', dm_cooling_rate)

            return heating_rate - cooling_rate - dm_cooling_rate

        if s0<=1e-26:
            lower = -2
            upper = 3
        if s0<=1e-25 and s0>1e-26:
            lower = -5
            upper = 10
        elif s0>1e-25 and s0<=1e-24:
            lower = -1
            upper = 40
        elif s0>1e-24:
            lower = 0
            upper = 50

        #print(lower, upper)
        try:
            log_L_solution = brentq(equation, lower, upper)
            #print(log_L_solution)
            #raise ValueError("brentq not working")
        except ValueError as e:
                print(f"Error finding log_L_solution: {e}")
                print(f"equation({lower}): {equation(lower)}, equation({upper}): {equation(upper)}")
                print(f"sigma0: {s0}, mx: {mx}, scaling_params: {scaling_params}, M500: {M500}")

                # try:
                #     result = root_scalar(equation, x0=upper, x1=upper+1, method='secant', bracket=[lower+5, upper+1])
                #     if result.converged:
                #         log_L_solution = result.root
                #     else:
                #         raise ValueError("root_scalar did not converge")
                # except Exception as e:
                #     print(f"Error finding log_L_solution with root_scalar: {e}")
                #     raise

        L_solution = 10**(log_L_solution)

        return L_solution


def Linj_from_L500_sampled(logL500, m, b):
    logMbh = b+m*(logL500)#/(1e44*u.erg/u.s))
    Mbh = np.power(10, logMbh)#*u.Msun
    return 1e44 * (Mbh/(np.power(10, 9.5)))#*u.Msun))

def T500_from_L500_sampled(logL500,m, b):
    logT = (logL500 - b)/m
    return np.power(10, logT) * 1e-6 # in GeV