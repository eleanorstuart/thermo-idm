import numpy as np
from astropy import constants as const
from astropy import units as u

class InteractionRate:
    def __init__(self, cluster):
        self.cluster = cluster

    
    def velocity(self, r): # input r as Mpc, M_enc should be in Msun
        G = const.G.to(u.Mpc/u.Msun*(u.cm/u.s)**2).value
        return np.sqrt(G*self.cluster.M_enc(r)/r) # in cm/s
    
    def dm_number_density(self, r, mchi, fdm=0.9): #input mchi in GeV
        mchi_Msun = mchi*(u.GeV).to(u.Msun, equivalencies=u.mass_energy())
        return fdm*self.cluster.rho_tot(r)/mchi_Msun # returns in 1/Mpc^3

    def calculate_interaction_rate(self, sigma0, mchi): # sigma0 in cm^2
        rmin = 0.001*self.cluster.R500
        rmax = self.cluster.R500
        avg_number_density = self.avg_over_volume(rmin, rmax, lambda r: self.dm_number_density(r, mchi))/(u.Mpc.to(u.cm))**3
        avg_velocity = self.avg_over_volume(rmin, rmax, self.velocity)


        interaction_rate = avg_number_density * avg_velocity * sigma0

        return interaction_rate
    
    def avg_over_volume(self, rmin, rmax, func):
        xmin = rmin.value/self.cluster.R500.value
        xmax = rmax.value/self.cluster.R500.value
        xs = np.logspace(np.log10(xmin), np.log10(xmax), num=50)
        rs = xs*self.cluster.R500
        return np.trapz([4*np.pi* r**2 * func(r) for r in rs],
                                       rs)/(4./3. * np.pi *(rmax**3-rmin**3))