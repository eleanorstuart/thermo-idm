from dataclasses import dataclass
from astropy import units as u
from astropy import constants as const
from cluster_functions import *


@dataclass
class ClusterMeasurements:
    R500: float = 1 * u.Mpc
    M500: float = 1.0e14 * u.Msun
    z: float = 0 # redshift
    L500: float = None
    T_var: float = None
    fb: float = 0.1

    def __post_init__(self):
        self.volume = 4 / 3 * np.pi * self.R500**3  # cluster volume
        self.N_b = self.fb*self.M500/const.m_p # number of baryons (might need to be 2x this?)
        self.n_e = self.N_b/self.volume # number density of electrons/baryons