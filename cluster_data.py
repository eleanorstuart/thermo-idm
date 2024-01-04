from dataclasses import dataclass
from astropy import units as u
from astropy import constants as const
from cluster_functions import *


@dataclass
class ClusterData:
    R500: float = 1 * u.Mpc
    M500: float = 1.0e14 * u.Msun
    z: float = 0. # redshift
    L500: float = None