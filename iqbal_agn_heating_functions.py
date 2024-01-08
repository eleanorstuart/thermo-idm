import numpy as np
import sympy as sp

from astropy import units as u
from scipy.integrate import quad
from scipy.optimize import approx_fprime

from astropy.cosmology import FlatLambdaCDM
cosmo=FlatLambdaCDM(70, 0.3)

# model consts
h70=1
P0=6.85
c500=1.09
gamma=0.31
alpha=1.07
beta=5.46
gamma_b=4./3.

def E(z): #ratio of the Hubble constant at redshift z to its present value
    return cosmo.H(z)/cosmo.H(0)

def P500(measurements):
    M500=measurements.M500.to(u.Msun)
    return ((1.65*1e-3*E(measurements.z)**(8/3)
            *(measurements.M500/(3*1e14*h70**(-1)*u.Msun))**2/3 
            *h70**2 * u.keV * u.cm**-3)).to(u.erg/u.cm**3, equivalencies=u.mass_energy()) 

def Pg(x, measurements): #x=r/r500
    return (P0*P500(measurements) #units keVcm-3
           / ((c500*x)**gamma  #unitless
              * (1+ (c500*x)**alpha)**((beta-gamma)/alpha))).to(u.erg/u.cm**3, equivalencies=u.mass_energy())

def Pg_r(r, measurements):
    if isinstance(r, float):
        r = r*u.Mpc
    return Pg(r/measurements.R500, measurements).value

def dP_dr(rad, measurements):
    if isinstance(rad, u.Quantity):
        rad.to(u.Mpc)
        rad = rad.value
    #rs = np.array(rad)
    #return np.gradient(r, Pg_r(r*u.Mpc, measurements))*u.erg/(u.cm**3 * u.Mpc)
    return [approx_fprime(r, lambda x: Pg_r(x*u.Mpc, measurements)) for r in rad]*u.erg/(u.cm**3 * u.Mpc)
    
def integrand(r, measurements, r0, rc):
    r=r*u.Mpc if isinstance(r, float) else r
    r0=r0*u.Mpc if isinstance(r0, float) else r0
    rc=rc*u.Mpc if isinstance(rc, float) else rc
    #if isinstance(r, float) r=r*u.Mpc else r
    #    r=r*u.Mpc
    x=r/measurements.R500
    #print(r, r0, rc)
    intgrd=(((Pg(x, measurements))**((gamma_b-1)/gamma_b)).to(u.erg**(1/4)*u.cm**(-3/4))
               *(1/Pg(x, measurements)).to(u.cm**3/u.erg) 
               #*np.array([dP_dr([rad.value], R500, M500, z).to(u.erg*u.cm**(-3)*u.Mpc**(-1)) for rad in rs]) 
               *dP_dr([r.value], measurements).to(u.erg/(u.cm**3 * u.Mpc))
               *(1-np.exp(-1*r/r0)).to(1)
               *(np.exp(-1*r/rc)).to(1))
    #print(((Pg(x, measurements))**((gamma_b-1)/gamma_b)).to(u.erg**(1/4)*u.cm**(-3/4)))
    #print(intgrd)
    return intgrd.to(u.erg**(1/4) * u.cm**(-3/4) *u.Mpc**(-1)).value
    
def q(measurements, r0, rc):
    rini=0.015*measurements.R500.to(u.Mpc).value
    rmax=measurements.R500.to(u.Mpc).value
    integral, _ = quad(lambda r: integrand(r, measurements, r0, rc), 
        rini, 
        rmax, )
        #args=(measurements, r0, rc))
    return integral*(u.erg**(1/4) * u.cm**(-3/4))

def h(Linj, r, r0, rc, q):
    return (Linj/(4*np.pi*r**2)
            *(1-np.exp(-1*r/r0))
            *np.exp(-1*r/rc)
            *(1/q))

def vol_heating_rate(rs, measurements, Linj, rc):
    #x=r/measurements.R500
    r0=(0.015*measurements.R500).to(u.cm)
    return np.array([(h(Linj, r, r0, rc, q(measurements, r0, rc))
    *(Pg(r/measurements.R500,measurements))**((gamma_b-1)/gamma_b)
    *(1/r)*(r/Pg(r/measurements.R500, measurements))*dP_dr([r.to(u.Mpc).value], measurements)).to(u.erg/(u.s*u.cm**3)) for r in rs]).flatten() * u.erg/(u.s * u.cm**3)

def cooling_function(T):
    alpha=-1.7
    beta=0.5
    c1 = 8.6*1e-25 * u.erg*u.cm**3/u.s * u.keV**(-alpha)
    c2= 5.8*1e-24*u.erg*u.cm**3/u.s * u.keV**(-beta)
    c3=6.3*1e-24*u.erg*u.cm**3/u.s
    return (c1*(T)**alpha + c2*T**beta + c3).to(u.erg*u.cm**3/u.s)

def vol_cooling_rate(n_e, T):
    mu_h=1.26
    mu_e=1.14
    return (n_e**2 * cooling_function(T) * mu_e/mu_h).to(u.erg/(u.s*u.cm**3))

def overdensity(z):
    return 18*np.pi**2 + 82*(cosmo.Om(z) - 1) - 39*(cosmo.Om(z) - 1)**2

def virial_radius(Mvir, z):
    return(Mvir/(4*np.pi/3 * overdensity(z) * cosmo.critical_density(z)))**(1/3)

def c_vir(Mvir, z):
    h=0.7 # TODO: set this properly
    return (7.85*(Mvir/(2*1e12 * 0.7 * u.Msun))**(-0.081) * (1+z)**(-0.71)).to(1)

def scale_radius(Mvir, z):
    return (virial_radius(Mvir, z)/c_vir(Mvir, z)).to(u.Mpc)
