import numpy as np
from astropy import units as u
from astropy import constants as const
from scipy.special import gamma
u.set_enabled_equivalencies(u.mass_energy())


def c(n):
    return 2 ** (5 + n / 2) / (3 * np.sqrt(np.pi)) * gamma(3 + n / 2)


def temp_from_vdisp(vel_disp):
    return (vel_disp**2 * const.m_p / const.k_B).to(
        u.GeV, equivalencies=u.temperature_energy()
    )


def temp_from_luminosity(luminosity):
    lum = luminosity.to(u.erg / u.s)
    log_T = (np.log10(lum.value) - 45.06) / 2.88 + np.log10(6)
    T = np.power(10, log_T) * u.keV
    return T.to(u.GeV)


def fun(T_b, cluster, p0, f_chi=1, n=0):
    T_b = T_b * u.GeV

    # sigma0=np.float_power(10, p0[0].astype(dtype=np.float128))*u.cm**2
    sigma0 = 10 ** p0[0] * u.cm**2
    # m_chi = np.float_power(10, p0[1].astype(dtype=np.float128))*u.GeV
    m_chi = 10 ** p0[1] * u.GeV

    norm = cluster.norm
    bh_mass = cluster.bh_mass
    mu = cluster.mu
    m_b = cluster.m_b
    nb = (2 * cluster.n_e).to(u.m ** (-3))
    gamma = cluster.adiabatic_idx
    V = cluster.volume
    efficiency = cluster.epsilon
    T_chi = cluster.virial_temperature(m_chi)

    accretion_factors = norm * 4 * np.pi * (const.G * bh_mass) ** 2
    plasma_entropy_factors = ((mu * m_b) ** (5 / 2) * nb) / gamma ** (
        3 / 2
    )  # no k_b because T_b in GeV
    cooling_factors = cluster.cooling_factors(n=n, f_chi=f_chi)

    B = (efficiency * accretion_factors * plasma_entropy_factors) / (cooling_factors)

    other_c = ((B * (m_chi + m_b) ** 2) / (sigma0) * (1 / const.c**3)).to(
        u.GeV ** (5 / 2)
    )
    return (T_b - T_chi) * (T_chi / m_chi + T_b / m_b) ** (1 / 2) * T_b ** (
        3 / 2
    ) - other_c
