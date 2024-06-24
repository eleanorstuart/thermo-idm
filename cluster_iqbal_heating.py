import numpy as np
from astropy import constants as const
from astropy import units as u

from asymmetric_uncertainty_mod import a_u



gamma_b=4./3.

class IqbalHeating:
    def __init__(self, cluster):
        self.cluster = cluster

    
    def q(self, r0, rc, n=50):
        rini=r0#.value # IS THIS TRUE?
        #rmax = self.Rvir.to(u.Mpc).value
        rmax = 2*self.cluster.R500 #.to(u.Mpc).value # TODO: decide on rmax

        xmin = rini.value/self.cluster.R500.value
        xmax = rmax.value/self.cluster.R500.value
        xs = np.logspace(np.log10(xmin), np.log10(xmax), num=n)
        rs = xs*self.cluster.R500

        integral = np.trapz(self.integrand(rs, r0, rc), rs)

        #integral = trapezoid(integrands, rs) #TODO: work out units to be able to use trapezoid
        return integral#*(u.erg**(1/4) * u.cm**(-3/4))
    
    def integrand(self, r, r0, rc):
        x=r/self.R500
        intgrd=(((self.Pg(x))**((gamma_b-1)/gamma_b))#.to(u.erg**(1/4)*u.cm**(-3/4))
               *(1/self.Pg(x))#.to(u.cm**3/u.erg) 
               *self.dP_dr(r)#.to(u.erg/(u.cm**3 * u.Mpc))
               *(1-np.exp(-1*r/r0))#.to(1)
               *(np.exp(-1*r/rc)))#.to(1))
        return intgrd#.to(u.erg**(1/4) * u.cm**(-3/4) *u.Mpc**(-1)).value
    

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