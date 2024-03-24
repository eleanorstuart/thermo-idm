import numpy as np

from astropy import units as u
from astropy import constants as const
from astropy.nddata import NDData, NDDataRef
from scipy.integrate import quad, trapezoid
from scipy.optimize import approx_fprime, brentq

from cluster_functions import c
from astropy_error_prop_functions import power_nddata, convert_nddata, log_nddata

from astropy.cosmology import FlatLambdaCDM
cosmo=FlatLambdaCDM(70, 0.3)

# model consts
h=0.7
h70=1
P0=6.85
c500=1.09
gamma=0.31
alpha=1.07
beta=5.46
gamma_b=4./3.

# nfw profile class
class NFWProfile():
    def __init__(self, z, Mvir=None, M500=None, R500=None, L500=None, M500_uncertainty=None):
        self.z = z
        self.rho_c = (3*cosmo.H(self.z)**2/(8*np.pi*const.G)).to(u.g/u.cm**3)
        self.overdensity_const = 18*np.pi**2 + 82*(cosmo.Om(self.z) - 1) - 39*(cosmo.Om(self.z) - 1)**2
        self.L500=L500 or None
        if Mvir:
            self.Mvir = Mvir
            self.cvir = (7.85*(self.Mvir/(2*1e12 * h**-1 * u.Msun))**(-0.081) * (1+self.z)**(-0.71)).to(1)
            self.Rvir = ((self.Mvir/(4*np.pi/3 * self.overdensity_const * self.rho_c))**(1/3)).to(u.Mpc)
            self.rs = (self.Rvir/self.cvir).to(u.Mpc)
            self.rho_s = self.get_rho_s(self.overdensity_const, self.cvir)
        elif M500: 
            #print('INSIDE M500')
            self.M500 = NDDataRef(data=M500, uncertainty=M500_uncertainty)
            #print(self.M500, self.M500.uncertainty)
            #calc_test = self.M500.multiply(2)
            #print(calc_test, calc_test.uncertainty)
            #self.M500.uncertainty = M500_uncertainty
            # with Duffy 2008 c200 params to be used as an approximation for c500 
            #self.c500 = (5.71*(self.M500.divide(2*1e12 * h**-1 * u.Msun))**(-0.084) * (1+self.z)**(-0.47)).to(1)
            self.c500 = (power_nddata(self.M500.divide(2*1e12 * h**-1 * u.Msun), -0.084).multiply(5.71).multiply( (1+self.z)**(-0.47)))#.to(1)
            old_c500 = (5.71*(self.M500.data *u.Msun/(2*1e12 * h**-1 * u.Msun))**(-0.084) * (1+self.z)**(-0.47)).to(1)
            #print('c500 compare:', self.c500, old_c500)

            self.R500 = R500 or self.get_R500()
            #print(self.R500)
            #print(self.M500.data)
            old_R500 = ((self.M500.data*1e14*u.Msun/(4*np.pi/3 * 500. * self.rho_c))**(1./3.)).to(u.Mpc)
            #print('R500 compare:', self.R500, old_R500)

            self.rs = convert_nddata(self.R500.divide(self.c500), u.Mpc)
            self.rho_s = self.get_rho_s(500., self.c500)

            #self.Rvir = self.get_R_from_overdensity(self.overdensity_const)
            #self.Mvir = self.M_enc(self.Rvir)
        else:
            raise ValueError("Provide Mvir or M500")

        if not M500:
            self.M500 = self.M_enc(self.R500)
        if not self.R500:
            self.R500 = self.get_R_from_overdensity(500.)
        

    # profile properties
    def get_rho_s(self, overdensity, concentration):
        #delta_c = overdensity/3 * concentration**3 / (np.log(1+concentration) - concentration/(1+concentration))
        delta_c =  power_nddata(concentration, 3).multiply(overdensity/3).divide(
            (log_nddata(concentration.add(1)).subtract(concentration.divide(concentration.add(1)))))
        #print(delta_c)
        return convert_nddata(delta_c.multiply(self.rho_c),(u.Msun/u.Mpc**3))

    def M_enc(self, r):
        if isinstance(r, float):
            r=r*u.Mpc
        y = r/self.rs
        return ((4 * np.pi * self.rs**3 * self.rho_s) * (np.log(1+y) - (y/(1+y)))).to(u.Msun)

    def get_R_from_overdensity(self, overdensity):
        rho_avg = lambda x: self.M_enc(x).value/(4./3.*np.pi * x**3) - overdensity*(self.rho_c).to(u.Msun/u.Mpc**3).value
        r = brentq(rho_avg, 0.1, 10) # search between 0.1 and 10 Mpc
        return r*u.Mpc

    def get_R500(self):
        #rho_avg = lambda x: self.M500.value/(4./3.*np.pi * x**3) - 500*(self.rho_c).to(u.Msun/u.Mpc**3).value
        #r = brentq(rho_avg, 0.1, 10) # search between 0.1 and 10 Mpc
        #return ((self.M500/(4*np.pi/3 * 500. * self.rho_c))**(1./3.)).to(u.Mpc)
        #print(self.M500.divide(4*np.pi/3 * 500. * self.rho_c))
        #print(power_nddata(self.M500.divide(4*np.pi/3 * 500. * self.rho_c),(1./3.)))
        return convert_nddata(power_nddata(self.M500.divide(4*np.pi/3 * 500. * self.rho_c),(1./3.)), u.Mpc)

    def rho_tot(self, r):
        y = r/self.rs
        return (self.rho_s/(y * (1+y)**2)).to(u.Msun/u.Mpc**3)
    

    # pressure
    def P500(self):
        Ez = cosmo.H(self.z)/cosmo.H0 
        return (1.14/0.59 * (1.65*1e-3*Ez**(8./3.)
            *(self.M500/(3*1e14*h70**(-1)*u.Msun))**2./3. 
            *h70**2 * u.keV * u.cm**-3)).to(u.erg/u.cm**3, equivalencies=u.mass_energy()) 

    def P500_planelles(self):
        omega_m =0.24
        omega_lambda = 1-omega_m
        h_p = 72./100.
        Ez = np.sqrt(omega_m*(1+self.z)**3 + omega_lambda)
        # equivalently: Ez = cosmo.H(self.z)/cosmo.H0 
        return (1.14/0.59 * 1.45*1e-11*u.erg/u.cm**3 * (self.M500/(1e15*h_p**(-1)*u.Msun))**(2./3.)*Ez**(8./3.))

    def Pg(self, x): #x=r/r500
        return (P0*self.P500_planelles() # switch out for P500()
           / (np.power(c500*x,gamma) 
              * (1+ np.power(c500*x, alpha))**((beta-gamma)/alpha))).to(u.erg/u.cm**3, equivalencies=u.mass_energy())

    def Pg_r(self, r): 
        if isinstance(r, float):
            r = r*u.Mpc
        return self.Pg(r/self.R500).value

    def dP_dr(self, rad): # rad has to be a list of rs in Mpc
        if isinstance(rad, u.Quantity):
            rad.to(u.Mpc)
            rad = rad.value
        return [approx_fprime(r, lambda x: self.Pg_r(x*u.Mpc), epsilon=r*1e-8)[0] for r in rad]*u.erg/(u.cm**3 * u.Mpc)

    # gas profiles
    def rho_g(self, r): # density profile of the baryons in the ICM
        dPdr=self.dP_dr(r)
        return (-1*(np.power(r, 2)/(const.G*self.M_enc(r)))*(dPdr)*const.c**2).to(u.Msun/u.Mpc**3, equivalencies=u.mass_energy())

    def T_g(self, r_value): #r is inputted as float
        r=r_value*u.Mpc
        mu=0.59 # this comes from Tozzi, check this
        return (mu*const.m_p*self.Pg(r/self.R500)/self.rho_g(r)).to(u.GeV)

    # effervescent heating 
    def vol_heating_rate(self, rs, rc, Linj=None):
        r0=(0.015*self.R500).to(u.Mpc)
        q_factor = self.q(r0, rc)
        L = Linj or self.get_Linj_from_L500() #Linj or self.Linj(rc)
        return np.array([(self.h(L, r, r0, rc, q_factor)
            *(self.Pg(r/self.R500))**((gamma_b-1)/gamma_b)
            *(1/r)
            *(r/self.Pg(r/self.R500))*self.dP_dr([r.to(u.Mpc).value])).to(u.erg/(u.s*u.cm**3)) for r in rs]).flatten() * u.erg/(u.s * u.cm**3)

    def total_heating_rate(self, rmin, rmax, rc, Linj = None,n=50): # rmin, rmax, rc given in Mpc
        log_rmin = np.log10(rmin.value)
        log_rmax = np.log10(rmax.value)
        rs = np.logspace(log_rmin, log_rmax, num=n)*u.Mpc
        integrand = (4 * np.pi * np.multiply(
            self.vol_heating_rate(rs, rc, Linj=Linj).to(u.erg/(u.s*u.Mpc**3)), 
            np.power(rs, 2))).to(u.erg/(u.s*u.Mpc))
        return trapezoid(integrand, rs)

    def q(self, r0, rc):
        rini=r0.value # IS THIS TRUE?
        #rmax = self.Rvir.to(u.Mpc).value
        rmax = 2*self.R500.to(u.Mpc).value # TODO: decide on rmax
        integral, _ = quad(lambda r: self.integrand(r, r0, rc), 
            rini, 
            rmax)
        #rs = np.logspace(np.log10(rini), np.log10(rmax))*u.Mpc
        #integrands = self.integrand(rs, r0, rc)
        #integral = trapezoid(integrands, rs) #TODO: work out units to be able to use trapezoid
        return integral*(u.erg**(1/4) * u.cm**(-3/4))

    def integrand(self, r, r0, rc):
        r=r*u.Mpc if isinstance(r, float) else r
        r0=r0*u.Mpc if isinstance(r0, float) else r0
        rc=rc*u.Mpc if isinstance(rc, float) else rc
        x=r/self.R500
        intgrd=(((self.Pg(x))**((gamma_b-1)/gamma_b)).to(u.erg**(1/4)*u.cm**(-3/4))
               *(1/self.Pg(x)).to(u.cm**3/u.erg) 
               *self.dP_dr([r.value]).to(u.erg/(u.cm**3 * u.Mpc))
               *(1-np.exp(-1*r/r0)).to(1)
               *(np.exp(-1*r/rc)).to(1))
        return intgrd.to(u.erg**(1/4) * u.cm**(-3/4) *u.Mpc**(-1)).value

    def h(self, Linj, r, r0, rc, q):
        return (Linj/(4*np.pi*r**2)
            *(1-np.exp(-1*r/r0))
            *np.exp(-1*r/rc)
            *(1/q))

    def Linj(self, rc):
        if rc == 0.3*self.R500:
            logLinj = -0.96 + 1.73*np.log10(self.Mvir/(1e14 * u.Msun))
        elif rc == 0.1*self.R500:
            logLinj = -1.58 + 1.53*np.log10(self.Mvir/(1e14 * u.Msun))
        return np.power(10, logLinj) * 1e45 * u.erg/u.s

    def get_Linj_from_L500(self, L=None):
        L500 = L or self.L500
        logMbh = 10+0.38*np.log10(L500/(1e44*u.erg/u.s))
        Mbh = np.power(10, logMbh)*u.Msun
        return 1e44*u.erg/u.s * (Mbh/(np.power(10, 9.5)*u.Msun))

    # gaspari agn heating
    def get_Mbh(self, L=None):
        L500 = L or self.L500
        logMbh = 10+0.38*np.log10(L500/(1e44*u.erg/u.s))
        return np.power(10, logMbh)*u.Msun

    def plasma_entropy(self, r):
        adiabatic_idx = 5. / 3.
        baryon_number_density = (2 * self.n_e(r)).to(u.m ** (-3))
        return (
            const.k_B * self.T_g(r.to(u.Mpc).value).to(u.K, equivalencies=u.temperature_energy())
        ).to(u.GeV) / baryon_number_density ** (adiabatic_idx - 1)    

    # radiative cooling rate
    def vol_cooling_rate(self, r):
        mu_h=1.26
        mu_e=1.14
        return (np.multiply(self.n_e(r)**2, self.cooling_function(r)) * mu_e/mu_h).to(u.erg/(u.s*u.cm**3))

    def total_cooling_rate(self, rmin, rmax, n=50):
        log_rmin = np.log10(rmin.value)
        log_rmax = np.log10(rmax.value)
        rs = np.logspace(log_rmin, log_rmax, num=n)*u.Mpc
        integrand = (4 * np.pi * np.multiply(
            self.vol_cooling_rate(rs).to(u.erg/(u.s*u.Mpc**3)), 
            np.power(rs, 2))).to(u.erg/(u.s*u.Mpc))
        return trapezoid(integrand, rs)

    def cooling_function(self, r):
        alpha=-1.7
        beta=0.5
        c1 = 8.6*1e-25 * u.erg*u.cm**3/u.s * u.keV**(-alpha)
        c2= 5.8*1e-24*u.erg*u.cm**3/u.s * u.keV**(-beta)
        c3=6.3*1e-24*u.erg*u.cm**3/u.s
        return (c1*(self.T_g(r.value))**alpha + c2*self.T_g(r.value)**beta + c3).to(u.erg*u.cm**3/u.s)

    def n_e(self, r):
        return np.sqrt(0.704)*(self.rho_g(r)/const.m_p).to(u.Mpc**-3)

    # DM cooling
    def virial_temperature(self, r, m_chi, f_chi=1, m_psi=0.1 * u.GeV,):
        M = self.M_enc(r)
        frac = f_chi / m_chi + (1 - f_chi) / m_psi
        M_kg = M.to(u.kg, equivalencies=u.mass_energy())
        return (0.3 * const.G * M_kg / (r * frac) * 1 / const.c**2).to(u.GeV)

    def vol_dm_cooling_rate(self, r, s0, m_chi, n=0, f_chi=1, m_psi=0.1*u.GeV):
        T_b = self.T_g(r.value)
        T_chi = self.virial_temperature(r, m_chi)
        with u.set_enabled_equivalencies(u.mass_energy()):
            uth = np.sqrt(T_b / const.m_p.to(u.GeV) + T_chi / m_chi).to(1)
        rho_chi = self.rho_tot(r) * f_chi

        denominator = ((m_chi + const.m_p) ** 2).to(u.GeV**2)
        numerator = (
                3
                * (T_b - T_chi).to(u.erg)
                * rho_chi.to(u.GeV/u.cm**3)
                * self.rho_g(r).to(u.GeV/u.cm**3, equivalencies=u.mass_energy())
                * c(n)
                * uth ** (n + 1)
                * (const.c.to(u.cm / u.s))
                * s0
        
            )
        return (numerator / denominator).to(u.erg/(u.s*u.cm**3))

    def integrated_dm_cooling_rate(self, rmin, rmax, s0, m_chi, num = 50, n=0, f_chi=1, m_psi=0.1*u.GeV):
        log_rmin = np.log10(rmin.value)
        log_rmax = np.log10(rmax.value)
        rs = np.logspace(log_rmin, log_rmax, num=num)*u.Mpc
    
        integrand = (4 * np.pi * np.multiply(
            self.vol_dm_cooling_rate(rs, s0, m_chi, n=n, f_chi=f_chi, m_psi=m_psi).to(u.erg/(u.s*u.Mpc**3)), 
            np.power(rs, 2))).to(u.erg/(u.s*u.Mpc))
        return trapezoid(integrand, rs)

    def calculate_s0_integrand(self,r,  m_chi, n=0, f_chi=1, m_psi=0.1*u.GeV):
        rho_chi = self.rho_tot(r) * f_chi
        with u.set_enabled_equivalencies(u.mass_energy()):
            uth = np.sqrt(self.T_g(r.value) / const.m_p.to(u.GeV) + self.virial_temperature(r, m_chi) / m_chi).to(1)
        return (
            3
            * (self.T_g(r.value) - self.virial_temperature(r, m_chi)).to(u.erg)
            * rho_chi.to(u.GeV / u.cm**3)
            * self.rho_g(r).to(u.GeV / u.cm**3, equivalencies=u.mass_energy())
            * c(n)
            * uth ** (n + 1)
            * (const.c.to(u.cm / u.s))
        ) / ((m_chi + const.m_p) ** 2).to(u.GeV**2)

    def sigma0_from_mchi(self, m_chi, n=0, f_chi=1, m_psi=0.1*u.GeV):
        # TODO: update to make these params customizable
        rmin = 0.002*self.R500
        rmax = 0.1*self.R500
        rc = 0.1*self.R500 

        log_rmin = np.log10(rmin.value)
        log_rmax = np.log10(rmax.value)
        rs = np.logspace(log_rmin, log_rmax)*u.Mpc

        integrand = (4 * np.pi * np.multiply(
            self.calculate_s0_integrand(rs, m_chi, n=n, f_chi=f_chi, m_psi=m_psi).to(u.erg/(u.s*u.Mpc**3*u.cm**2)), 
            np.power(rs, 2))).to(u.erg/(u.s*u.Mpc*u.cm**2))
        integral = trapezoid(integrand, rs)

        return (
            (
                self.total_heating_rate(rmin, rmax, rc)
                - self.total_cooling_rate(rmin, rmax)
            )
            / (integral)
        ).to(u.cm**2)

    # for MCMC
    def solve_for_L500(self, s0, mx):
        rmin = 0.001*self.R500
        rmax = self.R500
        net_cool = (self.integrated_dm_cooling_rate(rmin, rmax, s0, mx) + self.total_cooling_rate(rmin, rmax)).to(u.erg/u.s)
        rc = 0.1*self.R500
        # logL = log(L500/(erg/s))
        log_L_pred = brentq(f_L500, 30, 70, args=(self, net_cool, rmin, rmax, rc))
        return np.power(10, log_L_pred)*u.erg/u.s

    def L500_analytic(self, s0, mx):
        rmin = 0.001*self.R500
        rmax = self.R500
        net_cool = (self.integrated_dm_cooling_rate(rmin, rmax, s0, mx) + self.total_cooling_rate(rmin, rmax)).to(u.erg/u.s)
        rc = 0.1*self.R500
        agn_heating_without_Linj = self.total_heating_rate(rmin, rmax, rc, Linj = 1*u.erg/u.s, n=50)
        Linj = net_cool/agn_heating_without_Linj # in erg/s
        Mbh = Linj/10**(34.5) # in Msun
        logL500 = (np.log10(Mbh)-10)/0.38
        return np.power(10, logL500) * 1e44 *u.erg/u.s

    def h_no_L(self, r, r0, rc, q):
        return (1/(4*np.pi*r**2)
            *(1-np.exp(-1*r/r0))
            *np.exp(-1*r/rc)
            *(1/q))



def f_L500(logL, nfw, net_cool, rmin, rmax, rc):
    L = np.power(10, logL)*u.erg/u.s
    Linj = nfw.get_Linj_from_L500(L=L)
    heating = nfw.total_heating_rate(rmin, rmax, rc, Linj=Linj).to(u.erg/u.s)
    return (heating - net_cool).value

