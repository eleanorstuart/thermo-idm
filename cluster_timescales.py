from dataclasses import dataclass
from astropy import units as u
from astropy import constants as const
from cluster_functions import *

@dataclass
class ClusterTimescales: # TODO: figure out what to do with attributes that are calculated here and in main class (volume and fb)
    #radius: float = 1 * u.Mpc
    #mass: float = 1.0e14 * u.Msun
    #luminosity: float = None
    measurements: None


    def __post_init__(self):
        #self.mass = self.mass.to(u.GeV)  # total mass
        self.baryon_temp = temp_from_luminosity(self.measurements.L500)
        self.volume = 4 / 3 * np.pi * self.measurements.R500**3
        
        

    def dynamical_time(self): # from GFE
        rho = (self.measurements.M500 / self.volume).to(u.kg/u.m**3)
        t_dyn = np.sqrt(3*np.pi/(16*const.G*rho))
        return t_dyn.to(u.Gyr, equivalencies=u.mass_energy())

    def free_fall_time(self): # from GFE
        return self.dynamical_time()/np.sqrt(2)

    def cooling_time(self): # from Binney & Tremaine
        t_cool = (3/2 * self.baryon_temp) / (self.radiative_cooling_rate()/self.measurements.N_b)
        return t_cool.to(u.Gyr)
    
    def radiative_cooling_rate(self):
        prefactors=6.8*1e-42 *u.erg*u.cm**3
        Z=1
        T=self.baryon_temp.to(u.K, equivalencies=u.temperature_energy())
        T8=T/(1e8*u.K)
        C=(prefactors*Z**2*(self.measurements.n_e.to(u.cm**-3))**2)/(T8**(1/2))
        Eff_int = (C*T*const.k_B/const.h).to(u.GeV/(u.s*u.cm**3))
        return (self.volume*Eff_int).to(u.erg/u.s)

    def relaxation_time(self):
        return