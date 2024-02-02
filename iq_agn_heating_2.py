import numpy as np

from astropy import units as u
from astropy import constants as const
from scipy.integrate import quad
from scipy.optimize import approx_fprime, brentq

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
    def __init__(self, Mvir, z):
        self.Mvir = Mvir
        self.z = z

        self.Rvir = self.get_virial_radius()
        self.cvir = self.get_concentration_param()
        self.rs = self.get_scale_radius()

        self.R500 = self.get_R500()
        self.M500 = self.M_enc(self.R500)


    # cosmology functions (only depend on redshift)
    def E(self):
        return cosmo.H(self.z)/cosmo.H0

    def overdensity_const(self):
        return 18*np.pi**2 + 82*(cosmo.Om(self.z) - 1) - 39*(cosmo.Om(self.z) - 1)**2

    def rho_c(self):
        return (3*cosmo.H(self.z)**2/(8*np.pi*const.G)).to(u.g/u.cm**3)

    # profile properties
    def get_virial_radius(self):
        return ((self.Mvir/(4*np.pi/3 * self.overdensity_const() * self.rho_c()))**(1/3)).to(u.Mpc)

    def get_concentration_param(self):
        return (7.85*(self.Mvir/(2*1e12 * h**-1 * u.Msun))**(-0.081) * (1+self.z)**(-0.71)).to(1)

    def get_scale_radius(self):
        return (self.Rvir/self.cvir).to(u.Mpc)

    def delta_cvir(self):
        return self.overdensity_const()/3 * self.cvir**3 / (np.log(1+self.cvir) - self.cvir/(1+self.cvir))

    def rho_s(self):
        return (self.delta_cvir()*self.rho_c()).to(u.Msun/u.Mpc**3)

    def M_enc(self, r):
        if isinstance(r, float):
            r=r*u.Mpc
        y = r/self.rs
        return ((4 * np.pi * self.rs**3 * self.rho_s()) * (np.log(1+y) - (y/(1+y)))).to(u.Msun)

    def get_R500(self):
        rho_avg = lambda x: self.M_enc(x).value/(4/3*np.pi * x**3) - 500*(self.rho_c()).to(u.Msun/u.Mpc**3).value
        r500 = brentq(rho_avg, 0.1*self.Rvir.value, self.Rvir.value)
        return r500*u.Mpc

    # pressure
    def P500(self):
        return ((1.65*1e-3*self.E()**(8./3.)
            *(self.M500/(3*1e14*h70**(-1)*u.Msun))**2./3. 
            *h70**2 * u.keV * u.cm**-3)).to(u.erg/u.cm**3, equivalencies=u.mass_energy()) 

    def Pg(self, x): #x=r/r500
        return (P0*self.P500() 
           / (np.power(c500*x,gamma) 
              * (1+ np.power(c500*x, alpha))**((beta-gamma)/alpha))).to(u.erg/u.cm**3, equivalencies=u.mass_energy())

    def Pg_r(self, r):
        if isinstance(r, float):
            r = r*u.Mpc
        return self.Pg(r/self.R500).value

    def dP_dr(self, rad):
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
    def vol_heating_rate(self, rs, Linj, rc):
        r0=(0.015*self.R500).to(u.Mpc)

        return np.array([(self.h(Linj, r, r0, rc)
            *(self.Pg(r/self.R500))**((gamma_b-1)/gamma_b)
            *(1/r)
            *(r/self.Pg(r/self.R500))*self.dP_dr([r.to(u.Mpc).value])).to(u.erg/(u.s*u.cm**3)) for r in rs]).flatten() * u.erg/(u.s * u.cm**3)

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

    def h(self, Linj, r, r0, rc):
        return (Linj/(4*np.pi*r**2)
            *(1-np.exp(-1*r/r0))
            *np.exp(-1*r/rc)
            *(1/self.q(r0, rc)))

    # radiative cooling rate
    def vol_cooling_rate(self, r):
        mu_h=1.26
        mu_e=1.14
        return (np.multiply(self.n_e(r)**2, self.cooling_function(r)) * mu_e/mu_h).to(u.erg/(u.s*u.cm**3))

    def cooling_function(self, r):
        alpha=-1.7
        beta=0.5
        c1 = 8.6*1e-25 * u.erg*u.cm**3/u.s * u.keV**(-alpha)
        c2= 5.8*1e-24*u.erg*u.cm**3/u.s * u.keV**(-beta)
        c3=6.3*1e-24*u.erg*u.cm**3/u.s
        return (c1*(self.T_g(r.value))**alpha + c2*self.T_g(r.value)**beta + c3).to(u.erg*u.cm**3/u.s)

    def n_e(self, r):
        return np.sqrt(0.704)*(self.rho_g(r)/const.m_p).to(u.Mpc**-3)