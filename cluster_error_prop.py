import numpy as np

from astropy import units as u
from astropy import constants as const

from scipy.integrate import quad, trapezoid
from scipy.optimize import brentq


from asymmetric_uncertainty_mod import a_u, pos_errors, neg_errors

from cluster_functions import c
from cluster_energies import Energies
from cluster_iqbal_heating import IqbalHeating
from cluster_precalculations import PreCalculations
from cluster_thermalization import InteractionRate

from astropy.cosmology import FlatLambdaCDM
cosmo=FlatLambdaCDM(70, 0.3)

# model consts
h=0.7
h70=1
P0=6.85
c500_const=1.09
gamma=0.31
alpha=1.07
beta=5.46
gamma_b=4./3.
c0 = c(0) * (const.c.to(u.cm / u.s)).value
mp_GeV = (const.m_p).to(u.GeV).value
GeV_to_erg = (u.GeV).to(u.erg)
GeV_to_K = u.GeV.to(u.K, equivalencies=u.temperature_energy())
Mpc_to_cm = (u.Mpc).to(u.cm)

# nfw profile class
class Cluster_werr():
    def __init__(self, z, M500=None, L500=None, M500_err_lo=None, M500_err_hi=None, L_err=None, rmin=None):
        self.z = z
        self.rho_c = (3*cosmo.H(self.z)**2/(8*np.pi*const.G)).to(u.g/u.cm**3)
        self.overdensity_const = 18*np.pi**2 + 82*(cosmo.Om(self.z) - 1) - 39*(cosmo.Om(self.z) - 1)**2
        self.L500=L500
        
        if L_err!=None:
            self.L500 = a_u(self.L500.value, L_err.value, L_err.value)
            self.L_err = L_err.value
        if M500!=None: 
            self.M500 = a_u(M500.to(u.Msun).value, M500_err_hi.to(u.Msun).value,  M500_err_lo.to(u.Msun).value)
            self.c500 = (5.71*(self.M500/(2*1e12 * h**-1))**(-0.084) * (1+self.z)**(-0.47)) #this is an estimate from c200
            self.R500 = ((self.M500/(4*np.pi/3 * 500. * (self.rho_c).to(u.Msun/u.Mpc**3).value))**(1./3.))
            self.scale_radius = self.R500/self.c500
            #self.rho_s = self.get_rho_s(500., self.c500)
        else:
            raise ValueError("Provide M500")

        self.energies=Energies(self)
        self.heating=IqbalHeating(self)
        self.therm = InteractionRate(self)
       #self.precalc=PreCalculations(self)

        # calculations to make L500_analytic faster
        rmin = rmin*self.R500 if rmin else 0.001*self.R500
        rmax = self.R500
        rc = 0.1*self.R500

        #self.tot_rad = self.total_cooling_rate(rmin, rmax)
        self.agn_factors = (self.total_heating_rate(rmin, rmax, rc, Linj = 1*u.erg/u.s, n=50)).to(u.erg/u.s).value.value


        xmin = rmin.value/self.R500.value
        xmax = rmax.value/self.R500.value
        xs = np.logspace(np.log10(xmin), np.log10(xmax), num=50)
        rs = xs*self.R500
        self.Tbs = np.asarray(self.T_g(rs, match_gaspari=True))
        self.T500=self.Tbs[-1]
        self.Tbs_norm = np.array([i.value for i in self.Tbs/(self.T500*u.GeV.to(u.K, equivalencies=u.temperature_energy()))])

        self.rad_factors = self.total_cooling_rate(rmin, rmax, vol_rate='gfe', Tb=self.Tbs_norm).value# radiative cooling but needs to be multiplied by T500^1/2

        self.Tx_by_mx = np.asarray([t.value for t in self.Tx_over_mx(rs)])
        # with u.set_enabled_equivalencies(u.mass_energy()):
        #     uth = np.sqrt(self.Tbs / const.m_p.to(u.GeV).value + self.Tx_by_mx)
        #     conv = (u.Msun/u.Mpc**3).to(u.GeV/u.cm**3, equivalencies=u.mass_energy())
        #     self.dm_rdep_factors = np.asarray(self.rho_tot(rs) * self.rho_g(rs)*uth*conv**2)

        def D(r): # calculates the part of DM cooling that isnt dependent on L500 or sigma0 or mx
            f_chi=1
            rho_chi = self.rho_tot(r) * f_chi
            conv_dens = (u.Msun/u.Mpc**3).to(u.erg/u.cm**3)
            
            num = (
                3 * rho_chi*conv_dens#.to(u.GeV/u.cm**3)
                * self.rho_g(r)*conv_dens#.to(u.GeV/u.cm**3, equivalencies=u.mass_energy())
                * c(0)
                * (const.c.to(u.cm / u.s)).value
                #* s0
                * 4 * np.pi * np.power(r, 2)
            )
            #denom= ((mx + const.m_p.to(u.GeV).value) ** 2)
            #return np.divide(num, denom)
            return num
        conv_T = (u.GeV/u.erg).to(1)
        self.L_indep_factors = np.array([i.value for i in np.array(D(rs)*(u.Mpc/u.cm).to(1)**3)/conv_T])
        self.rs_Mpc = np.array([r.value for r in rs])
        
        #T_chi = self.virial_temperature(rs, mx)
        

        self.M500 = a_u(M500.to(u.Msun).value, M500_err_hi.to(u.Msun).value,  M500_err_lo.to(u.Msun).value)
        self.rvals = rs * Mpc_to_cm
        


    # profile properties
    def get_rho_s(self, overdensity, concentration):
        delta_c = overdensity/3 * concentration**3 / (np.log(1+concentration) - concentration/(1+concentration))
        rho_c = self.rho_c.to(u.Msun/u.Mpc**3).value
        return (delta_c*rho_c)

    def M_enc(self, r, r500=None, c500=None):
        if isinstance(r, float):
            r=r*u.Mpc
        r = np.array(r)

        r500=r500 or self.R500
        c500=c500 or self.c500
        ys = r/(r500/c500)
        y_term = (np.log(1+ys) - (ys/(1+ys)))
        #y_term = [(1+y).log() - (y/(1+y)) for y in ys] #gives same output as np.log
        rho_s = self.get_rho_s(500., c500)
        return np.multiply((4 * np.pi * (r500/c500)**3 * rho_s), y_term)


    def rho_tot(self, r, r500=None, c500=None):
        r500=r500 or self.R500
        c500=c500 or self.c500
        rho_s = self.get_rho_s(500., c500)
        y = r/(r500/c500)
        return np.divide(rho_s, (y * (1+y)**2)) # returns in Msun/Mpc**3
    
    # pressure
    def P500_planelles(self, m500=None): #uses old method to compute P500
        m500=m500 or self.M500
        omega_m =0.24
        omega_lambda = 1-omega_m
        h_p = 72./100.
        Ez = np.sqrt(omega_m*(1+self.z)**3 + omega_lambda)
        # returns in erg/cm3
        return (1.14/0.59 * 1.45*1e-11 * (m500/(1e15*h_p**(-1)))**(2./3.)*Ez**(8./3.))

    
    def Pg(self, x, m500=None): #x=r/r500
        P500 = self.P500_planelles(m500=m500)
        num = P0*P500
        denom = (np.power(np.multiply(c500_const,x),gamma) 
              * (1+ np.power(np.multiply(c500_const,x), alpha))**((beta-gamma)/alpha))
        return np.divide(num, denom) # returns in erg/cm3
    

    def Pg_r(self, r): 
        return self.Pg(r/self.R500) # returns in erg/cm3

    def dP_dr(self, r): # r has to be a list of rs in Mpc
        # Convert single value to list and append values if necessary
        single_value = False
        if isinstance(r, (int, float, a_u)) or (isinstance(r, list) and len(r) == 1):
            single_value = True
            if isinstance(r, list):
                r = r[0]
            r = np.array([r - 1e-6, r, r + 1e-6])  # Example adjustment, adjust as needed


        dPdr = np.gradient(self.Pg_r(r),r )
        grad2 = np.gradient(dPdr, r)
        dPdr_err_pos = np.sqrt((grad2*pos_errors(r))**2)
        dPdr_err_neg = np.sqrt((grad2*neg_errors(r))**2)

        results = [
        a_u(dPdr[i], dPdr_err_pos[i], dPdr_err_neg[i])
        for i in range(len(dPdr))
        ]

        # Remove appended values if input was a single value or list of length 1
        if single_value:
            return [results[1]]  # Return the middle value
        else:
            return results # returns in erg/cm3 /Mpc

    # gas profiles
    def rho_g(self, r): # density profile of the baryons in the ICM, r in Mpc
        dPdr=self.dP_dr(r)
        num = -1*np.power(r, 2)
        G = (const.G/const.c**2).to(u.Mpc / u.Msun).value
        conv = (u.erg/u.cm**3).to(u.Msun/u.Mpc**3)
        denom = (G*self.M_enc(r))
        return (num/denom)*np.array(dPdr)*conv # returns in Msun/Mpc3

    def T_g(self, r_value, match_gaspari=True, r500=None, m500=None, c500=None): #r is inputted as float
        r500=r500 or self.R500
        m500=m500 or self.M500

        r=r_value
        mu=0.59 # this comes from Tozzi, check this
        conv = (u.erg/u.cm**3).to(u.Msun/u.Mpc**3)
        if match_gaspari:
            anchor = self.energies.T500()
            T500 = (mu*(const.m_p.to(u.GeV).value)*self.Pg(r500/r500, m500=m500)/self.rho_g(r500))*conv
            mult_by = anchor/T500
        else:
            mult_by=1
        T_g = (mu*(const.m_p.to(u.GeV).value)*self.Pg(r/r500, m500=m500)/self.rho_g(r))*conv
        return T_g*mult_by # returns T_g in GeV



    # effervescent heating 
    def vol_heating_rate(self, rs, rc, Linj=None):
        r0=(0.015*self.R500)#.to(u.Mpc)
        q_factor = self.heating.q(r0, rc)
        L = Linj or self.heating.get_Linj_from_L500() #Linj or self.Linj(rc)
        hs = self.heating.h(L, rs, r0, rc, q_factor)
        Pgs = self.Pg(rs/self.R500)
        dPgdrs = self.dP_dr(rs)
        return(hs*Pgs**((gamma_b-1)/gamma_b))*(1./Pgs)*dPgdrs * (u.cm**3/u.Mpc**3).to(1)


    def total_heating_rate(self, rmin, rmax, rc, Linj = None,n=50): # rmin, rmax, rc given in Mpc
        xmin = rmin.value/self.R500.value
        xmax = rmax.value/self.R500.value
        xs = np.logspace(np.log10(xmin), np.log10(xmax), num=n)
        rs = xs*self.R500

        integrand = 4 * np.pi * np.multiply(
            self.vol_heating_rate(rs, rc, Linj=Linj), 
            np.power(rs, 2)) * (u.Mpc**3/u.cm**3).to(1)
        return trapezoid(integrand, rs)

    def dheating_dT(self, rmin, rmax, rc, T500):
        h = 1e-6 * T500
        Linj_plus = Linj_from_L500(L500_from_T500(T500+h))
        Linj_minus = Linj_from_L500(L500_from_T500(T500-h))
        heating_rate_plus = self.total_heating_rate(rmin, rmax, rc, Linj=Linj_plus)
        heating_rate_minus = self.total_heating_rate(rmin, rmax, rc, Linj=Linj_minus)
        dheating_dT = (heating_rate_plus - heating_rate_minus) / (2 * h)
        return dheating_dT

        


    # gaspari agn heating
    def get_Mbh(self, L=None):
        L500 = L or self.L500
        logMbh = 10+0.38*np.log10(L500)
        return np.power(10, logMbh)

    def plasma_entropy(self, r):
        adiabatic_idx = 5. / 3.
        baryon_number_density = (2 * self.n_e(r)).to(u.m ** (-3))
        return (
            const.k_B * self.T_g(r.to(u.Mpc).value).to(u.K, equivalencies=u.temperature_energy())
        ).to(u.GeV) / baryon_number_density ** (adiabatic_idx - 1)    

    # radiative cooling rate
    def vol_cooling_rate(self, r, match_gaspari=True):
        mu_h=1.26
        mu_e=1.14
        conv = (u.Mpc/u.cm).to(1)
        return (np.multiply((self.n_e(r)/conv**3)**2, self.cooling_function(r, match_gaspari=match_gaspari)) * mu_e/mu_h)#.to(u.erg/(u.s*u.cm**3))

    def total_cooling_rate(self, rmin, rmax, n=50, match_gaspari=True, vol_rate = 'iqbal', Tb=None):
        xmin = rmin.value/self.R500.value
        xmax = rmax.value/self.R500.value
        xs = np.logspace(np.log10(xmin), np.log10(xmax), num=n)
        rs = xs*self.R500

        if vol_rate=='iqbal':
            volume_cooling_rate = self.vol_cooling_rate(rs, match_gaspari=match_gaspari)
        elif vol_rate == 'gfe':
            volume_cooling_rate = self.vol_radiative_cooling_rate_gfe(rs, match_gaspari=match_gaspari, Tb=Tb)

        #print(rs[0],rs[-1])
        integrand = (4 * np.pi * np.multiply(
            volume_cooling_rate, 
            np.power(rs, 2))*(u.Mpc/u.cm).to(1)**3)
        return trapezoid(integrand, rs)
    
    def dcooling_dT(self, rmin, rmax, T500,  match_gaspari=True):
        h = 1e-6 * T500 *u.GeV.to(u.K, equivalencies=u.temperature_energy())
        cooling_rate_plus = self.total_cooling_rate(rmin, rmax, vol_rate='gfe', Tb=(self.Tbs_norm)*(T500+h))
        cooling_rate_minus = self.total_cooling_rate(rmin, rmax, vol_rate='gfe',  Tb=(self.Tbs_norm)*(T500-h))
        dcooling_dT = (cooling_rate_plus - cooling_rate_minus) / (2 * h)
        return dcooling_dT

    def cooling_function(self, r, match_gaspari=True):
        alpha=-1.7
        beta=0.5
        c1 = 8.6*1e-25# * u.erg*u.cm**3/u.s * u.keV**(-alpha)
        c2= 5.8*1e-24#*u.erg*u.cm**3/u.s * u.keV**(-beta)
        c3=6.3*1e-24#*u.erg*u.cm**3/u.s
        conv = (u.GeV/u.keV).to(1)
        return (c1*(self.T_g(r, match_gaspari=match_gaspari)*conv)**alpha + c2*(self.T_g(r, match_gaspari=match_gaspari)*conv)**beta + c3)#.to(u.erg*u.cm**3/u.s)

    def n_e(self, r):
        return np.sqrt(0.704)*(self.rho_g(r)/(const.m_p).to(u.Msun).value) # returns in 1/Mpc3
    
    def vol_radiative_cooling_rate_gfe(self, r, match_gaspari=True, Tb=None):
        prefactors=6.8*1e-38 #*u.erg*u.cm**3
        Tbs = Tb if Tb is not None else self.T_g(r, match_gaspari=match_gaspari)
        Tk=Tbs*u.GeV.to(u.K, equivalencies=u.temperature_energy()) # convert T from GeV to K
        ncm=self.n_e(r)/((u.Mpc/u.cm).to(1)**3)

        return prefactors * np.sqrt(Tk) *ncm**2 *(const.k_B/const.h).value # returns in erg/s /cm3

    # DM cooling
    def virial_temperature(self, r, m_chi, f_chi=1, m_psi=0.1 * u.GeV,):
        conv = (u.Msun/u.kg).to(1)
        M = self.M_enc(r)
        M_kg = self.M_enc(r) * conv
        frac = f_chi / m_chi+ (1 - f_chi) / m_psi.value
        G = (const.G/const.c**2).to(u.Mpc / u.Msun).value
        #M_kg = M.to(u.kg, equivalencies=u.mass_energy())
        #print(r)
        return (0.3 * G * M / (r * frac))#.to(u.GeV)

    def Tx_over_mx(self, r):
        #conv = (u.Msun/u.kg).to(1)
        M = self.M_enc(r)
        #M_kg = self.M_enc(r) * conv
        #frac = f_chi / m_chi.value + (1 - f_chi) / m_psi.value
        G = (const.G/const.c**2).to(u.Mpc / u.Msun).value
        #M_kg = M.to(u.kg, equivalencies=u.mass_energy())
        return (0.3 * G * M / (r))#.to(u.GeV)


    def vol_dm_cooling_rate(self, r, s0, m_chi, n=0, f_chi=1, m_psi=0.1*u.GeV, Tb = None):
        T_b = Tb if Tb is not None else self.Tbs
        T_chi = self.virial_temperature(r, m_chi)
        with u.set_enabled_equivalencies(u.mass_energy()):
            uth = np.sqrt(T_b / (const.m_p.to(u.GeV)).value + T_chi / m_chi)

        rho_chi = self.rho_tot(r) * f_chi
        conv_dens = (u.Msun/u.Mpc**3).to(u.erg/u.cm**3)
        conv_T = (u.GeV/u.erg).to(1)

        denominator = ((m_chi + const.m_p.to(u.GeV).value) ** 2)
        numerator = (
                3
                * (T_b - T_chi)/conv_T
                * rho_chi*conv_dens#.to(u.GeV/u.cm**3)
                * self.rho_g(r)*conv_dens#.to(u.GeV/u.cm**3, equivalencies=u.mass_energy())
                * c(n)
                * uth ** (n + 1)
                * (const.c.to(u.cm / u.s)).value
                * s0
        
            )
        return (numerator / denominator)#.to(u.erg/(u.s*u.cm**3))

    def integrated_dm_cooling_rate(self, rmin, rmax, s0, m_chi, num = 50, n=0, f_chi=1, m_psi=0.1*u.GeV, Tb=None):
        #log_rmin = np.log10(rmin)
        #log_rmax = np.log10(rmax)
        xmin = rmin.value/self.R500.value
        xmax = rmax.value/self.R500.value
        xs = np.logspace(np.log10(xmin), np.log10(xmax), num=num)
        rs = xs*self.R500
        #rs = np.logspace(log_rmin, log_rmax, num=num)*u.Mpc
        #print([r.value for r in rs])

        #rs_cm = rs*(u.Mpc/u.cm).to(1)


        integrand = (4 * np.pi * np.multiply(
            self.vol_dm_cooling_rate(rs, s0, m_chi, n=n, f_chi=f_chi, m_psi=m_psi, Tb=Tb)*(u.Mpc/u.cm).to(1)**3, np.power(rs, 2)))
        #print(integrand[0])
        return np.trapz(integrand, rs)
    
    def d_dmcooling_dT(self, r, s0, m_chi, n=0, f_chi=1, m_psi=0.1*u.GeV, T500=None):
        h = 1e-6 * T500 *u.GeV.to(u.K, equivalencies=u.temperature_energy())
        cooling_rate_plus = self.integrated_dm_cooling_rate(r, r, s0, m_chi, n=n, f_chi=f_chi, m_psi=m_psi, Tb=(self.Tbs_norm)*(T500+h))
        cooling_rate_minus = self.integrated_dm_cooling_rate(r, r, s0, m_chi, n=n, f_chi=f_chi, m_psi=m_psi, Tb=(self.Tbs_norm)*(T500-h))
        dcooling_dT = (cooling_rate_plus - cooling_rate_minus) / (2 * h)
        return dcooling_dT

    def integrated_dm_cooling_rate_fast(self, mx, s0_factors):
        # calculate the DM cooling rate for a given mchi and sigma0
        rs = self.rvals
        rel_temp = (self.Tbs - self.Tx_by_mx * mx)
        vol_dm_cooling_rates = (s0_factors * rel_temp * self.dm_rdep_factors)
        integrand = 4 * np.pi * vol_dm_cooling_rates * rs**2
        return np.trapz(integrand, rs)
    
    def calculate_eddington_luminosity(self):
        # calculate eddington luminosity for a galaxy cluster as a float in units of erg/s
        return (10**(47.5) * self.get_Mbh()/(10**(9.5)))


    def calculate_s0_integrand(self,r,  m_chi, n=0, f_chi=1, m_psi=0.1*u.GeV, Tb = None):
        rho_chi = self.rho_tot(r) #* f_chi


        Tbs = Tb if Tb is not None else self.Tbs
        with u.set_enabled_equivalencies(u.mass_energy()):
            uth = np.sqrt(Tbs / const.m_p.to(u.GeV).value +self.Tx_by_mx)#+ self.virial_temperature(r, m_chi) / m_chi).to(1)
        #conv = (u.Msun/u.Mpc**3).to(u.GeV/u.cm**3, equivalencies=u.mass_energy())
        #self.dm_rdep_factors = np.asarray(self.rho_tot(rs) * self.rho_g(rs)*uth*conv**2)
        return (
            3
            * (Tbs - self.Tx_by_mx * m_chi.value)#self.virial_temperature(r, m_chi)).to(u.erg)
            #* rho_chi*(u.Msun/u.Mpc**3).to(u.GeV / u.cm**3)
            #* self.rho_g(r).to(u.GeV / u.cm**3, equivalencies=u.mass_energy())
            *self.dm_rdep_factors
            * c(n)
            #* uth ** (n + 1)
            * (const.c.to(u.cm / u.s).value)
        ) / ((m_chi.value + const.m_p.to(u.GeV).value) ** 2)#.to(u.GeV**2)

    def sigma0_from_mchi(self, m_chi, n=0, f_chi=1, m_psi=0.1*u.GeV, rmin=0.002, rmax=0.1, agn_factor=1):
        # TODO: update to make these params customizable
        rmin = rmin*self.R500
        rmax = 1*self.R500
        rc = 0.1*self.R500 

        xmin = rmin.value/self.R500.value
        xmax = rmax.value/self.R500.value
        xs = np.logspace(np.log10(xmin), np.log10(xmax), num=n)
        rs = xs*self.R500

        #print(self.calculate_s0_integrand(rs, m_chi, n=n, f_chi=f_chi, m_psi=m_psi))
        #print(np.power(self.rvals, 2))

        integrand = (4 * np.pi * np.multiply(
            self.calculate_s0_integrand(rs, m_chi, n=n, f_chi=f_chi, m_psi=m_psi),#.to(u.erg/(u.s*u.Mpc**3*u.cm**2)), 
            np.power(self.rvals, 2)))#.to(u.erg/(u.s*u.Mpc*u.cm**2))
        integral = trapezoid(integrand, self.rvals)

        return (
            (
                agn_factor*self.total_heating_rate(rmin, rmax, rc).value
                - self.total_cooling_rate(rmin, rmax).value
            )
            / (integral)
        )#.to(u.cm**2)

    # for MCMC
    def L500_analytic(self, mx, s0_factors, mult_agn = 1):
        # calculates predicted L500 for a given mx and sigma0 

        net_cool = (self.integrated_dm_cooling_rate_fast(mx, s0_factors) + self.tot_rad)
        agn_heating_without_Linj = mult_agn*self.agn_factors
        # calculate predicted Linj
        Linj = net_cool/agn_heating_without_Linj
        # convert Linj into L500
        Mbh = Linj/10**(34.5) # in Msun
        logL500 =  (np.log10(Mbh) - a_u(10, 0.11, 0.11)) / a_u(0.38, 0.3, 0.3)
        return np.power(10, logL500) * 1e44 
    
    def L500_analytic_eddington(self, mx, s0_factors, frac_L = 1):
        # returns the predicted L500 for a given mx and sigma0 based on the eddington luminosity equation and L500-Mbh relation
        # calculate predicted eddington luminosity
        eddington_luminosity = (self.integrated_dm_cooling_rate_fast(mx, s0_factors) + self.tot_rad)/(frac_L)
        
        # calculate Mbh 
        Mbh = eddington_luminosity/1e38
        
        # calculate logL500 from Mbh
        logL500 = (np.log10(Mbh) - a_u(10, 0.11, 0.11)) / a_u(0.38, 0.03, 0.03)
        
        # return L500
        return np.power(10, logL500) * 1e44


    def L500_numerical(self, mx, s0):
        L_indep_factors_full = np.divide(self.L_indep_factors*s0,((mx + mp_GeV) ** 2))
        T_chi = np.array(self.Tx_by_mx*mx)

        L_indep_factors_full_min = np.divide(self.precalc.L_indep_factors_min*s0,((mx + mp_GeV) ** 2))
        T_chi_min = np.array(self.precalc.Tx_by_mx_min*mx)

        L_indep_factors_full_max = np.divide(self.precalc.L_indep_factors_max*s0,((mx + mp_GeV) ** 2))
        T_chi_max = np.array(self.precalc.Tx_by_mx_max*mx)

        def equation(logL): # L in 1e44 erg/s
            T500 = T500_from_L500(logL, bound=None)
            T500k = T500*GeV_to_K
            Tbs = np.multiply(self.Tbs_norm, T500k)

            cooling_rate=np.sqrt(T500k)*self.rad_factors

            uth = np.sqrt(Tbs/mp_GeV + self.Tx_by_mx)
            integrand = (L_indep_factors_full*uth)*(Tbs-T_chi)
            dm_cooling_rate = np.trapz(integrand, self.rs_Mpc)

            Linj=Linj_from_L500(logL)
            heating_rate = Linj * self.agn_factors
            #print(heating_rate)
            #print('heating',heating_rate.value, 'cooling', cooling_rate.value, 'dm cooling', dm_cooling_rate.value)

            return heating_rate - cooling_rate - dm_cooling_rate
        
        def equation_upper(logL): # yields L upper bound from highest mass, highest temperature, lowest AGN heating (lowest net heating for a given L)
            T500 = T500_from_L500(logL, bound='max')
            T500k = T500*GeV_to_K
            Tbs = np.multiply(self.precalc.Tbs_norm_max, T500k)
            #Tbs = np.multiply(self.Tbs_norm, T500k)

            cooling_rate=np.sqrt(T500k)*self.precalc.rad_factors_max
            #cooling_rate=np.sqrt(T500k)*self.rad_factors

            uth = np.sqrt(Tbs/mp_GeV + self.precalc.Tx_by_mx_max)
            #uth = np.sqrt(Tbs/mp_GeV + self.Tx_by_mx)
            integrand = (L_indep_factors_full_max*uth)*(Tbs-T_chi_max)
            #integrand = (L_indep_factors_full*uth)*(Tbs-T_chi)
            dm_cooling_rate = np.trapz(integrand, np.array([r.value for r in self.precalc.rs_max]))
            #dm_cooling_rate = np.trapz(integrand, self.rs_Mpc)

            Linj=Linj_from_L500(logL, bound='min')
            heating_rate = Linj * self.precalc.agn_factors_max
            #heating_rate = Linj * self.agn_factors

            return heating_rate - cooling_rate - dm_cooling_rate

        def equation_lower(logL): # yields L lower bound from lowest mass, lowest temperature, highest AGN heating (highest net heating for a given L)
            T500 = T500_from_L500(logL, bound='min')
            T500k = T500*GeV_to_K
            Tbs = np.multiply(self.precalc.Tbs_norm_min, T500k)
            #Tbs = np.multiply(self.Tbs_norm, T500k)

            cooling_rate=np.sqrt(T500k)*self.precalc.rad_factors_min
            #cooling_rate=np.sqrt(T500k)*self.rad_factors

            uth = np.sqrt(Tbs/mp_GeV + self.precalc.Tx_by_mx_min)
            #uth = np.sqrt(Tbs/mp_GeV + self.Tx_by_mx)
            integrand = (L_indep_factors_full_min*uth)*(Tbs-T_chi_min)
            #integrand = (L_indep_factors_full*uth)*(Tbs-T_chi)
            dm_cooling_rate = np.trapz(integrand, np.array([r.value for r in self.precalc.rs_min]))
            #dm_cooling_rate = np.trapz(integrand, self.rs_Mpc)

            Linj=Linj_from_L500(logL, bound='max')
            heating_rate = Linj * self.precalc.agn_factors_min
            #heating_rate = Linj * self.agn_factors

            return heating_rate - cooling_rate - dm_cooling_rate

        log_L_solution = (brentq(equation, -2, 180))
        #print(log_L_solution)
        L_upper=10**(brentq(equation_upper, log_L_solution-10, log_L_solution+180))
        L_lower=10**(brentq(equation_lower, log_L_solution - 20, log_L_solution+10))
        L_solution = 10**(log_L_solution)
        #print(L_solution, L_upper, L_lower)
        # could do L_upper - L_lower / 2 but I want to be generous with the error bars
        L_var = np.max([L_upper - L_solution, L_solution - L_lower])
        #return 10**(L_solution), 10**(L_upper), 10**(L_lower) # returns in units of 1e44 erg/s
        return L_solution, L_var

        # Ls = np.linspace(-5, 5)
        # net_heating_rate = [equation(L) for L in Ls]
        # net_heating_rate_min = [equation_lower(L) for L in Ls]
        # net_heating_rate_max = [equation_upper(L) for L in Ls]
        #L = 0

        #return net_heating_rate, net_heating_rate_min, net_heating_rate_max
        #return equation(L), equation_upper(L), equation_lower(L)#net_heating_rate
        

def Linj_from_L500(logL500, bound=None):
    # calculate Linj for a given L500
    if bound:
        if bound=='min': # minimize Linj by taking lowest b, m values
            b = 10 #- 0.11
            m = 0.38 #- 0.03
        elif bound=='max': # maximize  Linj by taking highest b, m values
            b = 10 #+ 0.11
            m = 0.38 #+ 0.03
    else:
        b = 10
        m = 0.38

    logMbh = b+m*(logL500)#/(1e44*u.erg/u.s))
    Mbh = np.power(10, logMbh)#*u.Msun
    return 1e44 * (Mbh/(np.power(10, 9.5)))#*u.Msun))

def T500_from_L500(logL500, bound=None):
    if bound:
        if bound=='max': # maximize T by taking lowest b, m values
            b = -2.34 #- 0.09
            m = 4.71 #- 0.26
        elif bound=='min': # minimize T by taking highest b, m values
            b = -2.34 #+ 0.09
            m = 4.71 #+ 0.26
    else:
        b = -2.34
        m = 4.71
    logT = (logL500 - b)/m
    return np.power(10, logT) * 1e-6 # in GeV

def L500_from_T500(T500):
    logT = np.log10(T500*1e6)
    return -2.34 + 4.71*logT

def L500_from_Linj(Linj):
    logMbh = np.log10(Linj/1e44) + 9.5
    return (logMbh - 10) / 0.38



# def temp_from_luminosity(luminosity):
#     lum = luminosity.to(u.erg / u.s)
#     log_T = (np.log10(lum.value) - 45.06) / 2.88 + np.log10(6)
#     T = np.power(10, log_T) * u.keV
#     return T.to(u.GeV)
