import numpy as np
from astropy import constants as const
from astropy import units as u

from asymmetric_uncertainty_mod import a_u

class Energies:
    def __init__(self, cluster):
        self.cluster = cluster

    @property
    def thermal_energy(self):
        # Implement your kinetic energy calculation here
        # You can access properties of the cluster with self.cluster.property_name
        rs = np.logspace(np.log10(0.15), 0)*self.cluster.R500
        Tgs = self.cluster.T_g(rs)
        avg_Tg = (np.trapz(4*np.pi * rs**2 * Tgs, rs))/(4./3 * np.pi * (self.cluster.R500**3 - (0.15*self.cluster.R500)**3) )
        return 3./2 * avg_Tg

    @property
    def abs_potential_energy(self):
        # Implement your potential energy calculation here
        # You can access properties of the cluster with self.cluster.property_name
        mu=0.59
        G = (const.G/const.c**2).to(u.Mpc / u.Msun).value
        mp_GeV = (const.m_p).to(u.GeV).value
        return  3./2*G*mu*mp_GeV*self.cluster.M500/self.cluster.R500
    
    @property
    def T500(self):
        b = a_u(-2.34, 0.09, 0.09)
        m = a_u(4.71, 0.26, 0.26)
        logL = np.log10(self.cluster.L500)
        logT = (logL - b)/m
        return np.power(10, logT) * 1e-6 # in GeV
    
    @property
    def thermal_energy_gaspari(self):
        return 3./2 * self.T500
    

    @property
    def virialization_fraction_gaspari(self):
        return 2*self.thermal_energy_gaspari/self.abs_potential_energy
    
    @property
    def virialization_fraction_iqbal(self):
        return 2*self.thermal_energy/self.abs_potential_energy
    
    @property
    def is_virialized_gaspari(self):
        if self.virialization_fraction_gaspari.value<=1:
            if self.virialization_fraction_gaspari.value + 2*self.virialization_fraction_gaspari.plus >=1:
                return True
        elif self.virialization_fraction_gaspari.value>=1:
            if self.virialization_fraction_gaspari.value - 2*self.virialization_fraction_gaspari.minus <=1:
                return True
        else:
            return False
        
    @property
    def is_virialized_iqbal(self):
        if self.virialization_fraction_iqbal.value<=1:
            if self.virialization_fraction_iqbal.value + 2*self.virialization_fraction_iqbal.plus >=1:
                return True
        elif self.virialization_fraction_iqbal.value>=1:
            if self.virialization_fraction_iqbal.value - 2*self.virialization_fraction_iqbal.minus <=1:
                return True
        else:
            return False