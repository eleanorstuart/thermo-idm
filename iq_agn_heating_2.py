import numpy as np

from astropy import units as u
from astropy import constants as const
from scipy.integrate import quad, trapezoid
from scipy.optimize import approx_fprime, brentq

from cluster_functions import c

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
    def __init__(self, z, Mvir=None, M500=None):
        if Mvir:
            self.Mvir = Mvir
        elif M500: 
            self.M500 = M500
            # TODO: calculate Mvir from M500 and c500
        self.z = z

        self.Rvir = self.get_virial_radius()
        self.cvir = self.get_concentration_param()
        self.rs = (self.Rvir/self.cvir).to(u.Mpc)

        self.R500 = self.get_R500()
        if not M500:
            self.M500 = self.M_enc(self.R500)
        


    # cosmology functions (only depend on redshift)
    def overdensity_const(self):
        return 18*np.pi**2 + 82*(cosmo.Om(self.z) - 1) - 39*(cosmo.Om(self.z) - 1)**2

    def rho_c(self):
        return (3*cosmo.H(self.z)**2/(8*np.pi*const.G)).to(u.g/u.cm**3)

    # profile properties
    def get_virial_radius(self):
        return ((self.Mvir/(4*np.pi/3 * self.overdensity_const() * self.rho_c()))**(1/3)).to(u.Mpc)

    def get_concentration_param(self):
        return (7.85*(self.Mvir/(2*1e12 * h**-1 * u.Msun))**(-0.081) * (1+self.z)**(-0.71)).to(1)

    def get_c200(self): # to be used as an approximation for c500 TODO: put in correct constants
        return (7.85*(self.Mvir/(2*1e12 * h**-1 * u.Msun))**(-0.081) * (1+self.z)**(-0.71)).to(1)

    def rho_s(self):
        delta_cvir = self.overdensity_const()/3 * self.cvir**3 / (np.log(1+self.cvir) - self.cvir/(1+self.cvir))
        return (delta_cvir*self.rho_c()).to(u.Msun/u.Mpc**3)

    def M_enc(self, r):
        if isinstance(r, float):
            r=r*u.Mpc
        y = r/self.rs
        return ((4 * np.pi * self.rs**3 * self.rho_s()) * (np.log(1+y) - (y/(1+y)))).to(u.Msun)

    def get_R500(self):
        rho_avg = lambda x: self.M_enc(x).value/(4./3.*np.pi * x**3) - 500*(self.rho_c()).to(u.Msun/u.Mpc**3).value
        r500 = brentq(rho_avg, 0.1*self.Rvir.value, self.Rvir.value)
        return r500*u.Mpc

    def rho_tot(self, r):
        y = r/self.rs
        return (self.rho_s()/(y * (1+y)**2)).to(u.Msun/u.Mpc**3)

    # pressure
    def P500(self):
        Ez = cosmo.H(self.z)/cosmo.H0 
        return ((1.65*1e-3*Ez**(8./3.)
            *(self.M500/(3*1e14*h70**(-1)*u.Msun))**2./3. 
            *h70**2 * u.keV * u.cm**-3)).to(u.erg/u.cm**3, equivalencies=u.mass_energy()) 

    def P500_planelles(self):
        omega_m =0.24
        omega_lambda = 1-omega_m
        h_p = 72./100.
        Ez = np.sqrt(omega_m*(1+self.z)**3 + omega_lambda)
        # equivalently: Ez = cosmo.H(self.z)/cosmo.H0 
        return (1.45*1e-11*u.erg/u.cm**3 * (self.M500/(1e15*h_p**(-1)*u.Msun))**(2./3.)*Ez**(8./3.))

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
        return (-1*(r**2/(const.G*self.M_enc(r)))*(dPdr)*const.c**2).to(u.Msun/u.Mpc**3, equivalencies=u.mass_energy())

    def T_g(self, r_value): #r is inputted as float
        r=r_value*u.Mpc
        mu=0.59 # this comes from Tozzi, check this
        return (mu*const.m_p*self.Pg(r/self.R500)/self.rho_g(r)).to(u.GeV)

    # effervescent heating 
    def vol_heating_rate(self, rs, rc, Linj=None):
        r0=(0.015*self.R500).to(u.Mpc)
        q_factor = self.q(r0, rc)
        L = Linj or self.Linj(rc)
        return np.array([(self.h(L, r, r0, rc, q_factor)
            *(self.Pg(r/self.R500))**((gamma_b-1)/gamma_b)
            *(1/r)
            *(r/self.Pg(r/self.R500))*self.dP_dr([r.to(u.Mpc).value])).to(u.erg/(u.s*u.cm**3)) for r in rs]).flatten() * u.erg/(u.s * u.cm**3)

    def total_heating_rate(self, rmin, rmax, Linj, rc, n=50): # rmin, rmax, rc given in Mpc
        log_rmin = np.log10(rmin.value)
        log_rmax = np.log10(rmax.value)
        rs = np.logspace(log_rmin, log_rmax, num=n)*u.Mpc
        integrand = (4 * np.pi * np.multiply(
            self.vol_heating_rate(rs, Linj, rc).to(u.erg/(u.s*u.Mpc**3)), 
            np.power(rs, 2))).to(u.erg/(u.s*u.Mpc))
        return trapezoid(integrand, rs)

    def q(self, r0, rc):
        rini=r0.value # IS THIS TRUE?
        rmax = self.Rvir.to(u.Mpc).value
        integral, _ = quad(lambda r: self.integrand(r, r0, rc), 
            rini, 
            rmax)
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
                * self.rho_g.to(u.GeV/u.cm**3, equivalencies=u.mass_energy())
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