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
        avg_number_density = self.avg_over_volume(rmin, rmax, lambda r: self.dm_number_density(r, mchi))/((u.Mpc.to(u.cm))**3)
        print(avg_number_density)
        avg_velocity = self.avg_over_volume(rmin, rmax, self.velocity)
        print(avg_velocity)


        interaction_rate = avg_number_density * avg_velocity * sigma0

        return interaction_rate # returns in 1/second
    
    def thermalization_timescale(self, sigma0, mchi, fdm=0.9):
        interaction_rate = self.calculate_interaction_rate(sigma0, mchi) * u.s.to(u.Gyr)
        return 1/interaction_rate
    
    def avg_over_volume(self, rmin, rmax, func):
        xmin = rmin.value/self.cluster.R500.value
        xmax = rmax.value/self.cluster.R500.value
        xs = np.logspace(np.log10(xmin), np.log10(xmax), num=50)
        rs = xs*self.cluster.R500
        return np.trapz([4*np.pi* r**2 * func(r) for r in rs],
                                       rs)/(4./3. * np.pi *(rmax**3-rmin**3))
    
    #def sigma0_vs_mchi_from_interaction_rate_limit(self, mchi):
    def sigma0_vs_mchi_from_interaction_rate_limit(self, mchi_list):

        if isinstance(mchi_list, list) or isinstance(mchi_list, np.ndarray): 
            sigma0_list = []
            for mchi in mchi_list:
                rmin = 0.001*self.cluster.R500
                rmax = self.cluster.R500
                avg_velocity = self.avg_over_volume(rmin, rmax, self.velocity)
                avg_number_density = self.avg_over_volume(rmin, rmax, lambda r: self.dm_number_density(r, mchi))/((u.Mpc.to(u.cm))**3)
                lifetime_s = 10*u.Gyr.to(u.s)
                sigma0 = 1/(lifetime_s*avg_number_density * avg_velocity)
                sigma0_list.append(sigma0)
            return sigma0_list
        else:
            rmin = 0.001*self.cluster.R500
            rmax = self.cluster.R500
            avg_velocity = self.avg_over_volume(rmin, rmax, self.velocity)
            avg_number_density = self.avg_over_volume(rmin, rmax, lambda r: self.dm_number_density(r, mchi_list))/((u.Mpc.to(u.cm))**3)
            lifetime_s = 10*u.Gyr.to(u.s)
            sigma0 = 1/(lifetime_s*avg_number_density * avg_velocity)
            return sigma0
        # rmin = 0.001*self.cluster.R500
        # rmax = self.cluster.R500
        # avg_velocity = self.avg_over_volume(rmin, rmax, self.velocity)
        # avg_number_density = self.avg_over_volume(rmin, rmax, lambda r: self.dm_number_density(r, mchi))/((u.Mpc.to(u.cm))**3)
        # lifetime_s = 10*u.Gyr.to(u.s)
        # return 1/(lifetime_s*avg_number_density * avg_velocity)