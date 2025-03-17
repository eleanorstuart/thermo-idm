import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.integrate import quad
from scipy.optimize import brentq

from asymmetric_uncertainty_mod import a_u



class Energies:
    def __init__(self, cluster):
        self.cluster = cluster
       # self.abs_potential_energy = self.abs_potential_energy
       # self.thermal_energy_norm = self.thermal_energy_norm() # in MCMC this needs to be multiplied by T500 to get the thermal energy


    def thermal_energy_norm(self):
        rs = np.logspace(np.log10(0.15), 0)*self.cluster.R500
        Tgs = self.cluster.T_g(rs)
        T500=Tgs[-1]
        Tgs_norm = np.array([i.value for i in Tgs/(T500*u.GeV.to(u.K, equivalencies=u.temperature_energy()))])

        avg_Tg_norm = (np.trapz(4*np.pi * rs**2 * Tgs_norm, rs))/(4./3 * np.pi * (self.cluster.R500**3 - (0.15*self.cluster.R500)**3))
        return 3./2 * avg_Tg_norm
    
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
    
    def T500(self, L=None):
        L500 = L if L is not None else self.cluster.L500
        b = a_u(-2.34, 0.09, 0.09)
        m = a_u(4.71, 0.26, 0.26)
        logL = np.log10(L500)
        logT = (logL - b)/m
        return np.power(10, logT) * 1e-6 # in GeV
    
    @property
    def thermal_energy_gaspari(self):
        return 3./2 * self.T500()
    

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
        

# functions for calculating surface pressure 
    def average_density(self, r):
        r500 = self.cluster.R500
        c500 = self.cluster.c500
        volume = (4 / 3) * np.pi * r**3
        integral, _ = quad(lambda x: 4 * np.pi * x**2 * self.cluster.rho_tot(x, r500, c500), 0, r)
        return integral / volume
    
    def vc2(self, r):
        return (const.G*self.cluster.M_enc(r)*u.Msun/(r*u.Mpc)).to((u.m/u.s)**2).value
    
    def integrand(self, r):
        return (4*np.pi*r**2*self.cluster.rho_tot(r).value*self.vc2(r))
    
    def Ps(self, Rvir, R80):
        V = 4*np.pi/3 * (Rvir**3 - (R80)**3)
        integral, _ = quad(lambda r: self.integrand(r), R80, Rvir)
        return integral/(3*V)
    
    def surface_pressure_energy(self):
        avg_rho_within_rvir = (self.cluster.overdensity_const*self.cluster.rho_c).to(u.Msun/u.Mpc**3).value
        
        Rvir= brentq(lambda r: self.average_density(r) - avg_rho_within_rvir, 1e-5, 10*self.cluster.R500.value)
        Mvir = (4*np.pi/3*Rvir**3 * avg_rho_within_rvir)
        
        R80 = brentq(lambda r: self.cluster.M_enc(r).value - 0.8*Mvir, 1e-5, Rvir)
        pressure = self.Ps(Rvir, R80)
        
        R90 = brentq(lambda r: self.cluster.M_enc(r).value - 0.9*Mvir, R80, Rvir)
        Es= ((4*np.pi*R90**3 * pressure)*(u.Msun*(u.m/u.s)**2))#.to(u.erg) 
        return Es
    
    def surface_pressure_energy_per_particle(self):
        Es = self.surface_pressure_energy()
        mu=0.59
        mp_GeV = (const.m_p).to(u.GeV)
        return (Es *mu* mp_GeV/(self.cluster.M500.value*u.Msun*const.c**2)).to(u.GeV, equivalencies=u.mass_energy()) 
        

    @property
    def virialization_fraction_surface_pressure(self):
        sp = self.surface_pressure_energy_per_particle().value
        surface_pressure_energy_err = a_u(sp, pos_err=2*self.cluster.M500.plus/self.cluster.M500.value * sp, neg_err=2*self.cluster.M500.minus/self.cluster.M500.value * sp)
        return (2*self.thermal_energy_gaspari - surface_pressure_energy_err)/self.abs_potential_energy
    

    @property
    def is_virialized_sp(self):
        if self.virialization_fraction_surface_pressure.value<=1:
            if self.virialization_fraction_surface_pressure.value + 2*self.virialization_fraction_surface_pressure.plus >=1:
                return True
        elif self.virialization_fraction_surface_pressure.value>=1:
            if self.virialization_fraction_surface_pressure.value - 2*self.virialization_fraction_surface_pressure.minus <=1:
                return True
        else:
            return False