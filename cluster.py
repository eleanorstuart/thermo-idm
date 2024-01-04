import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from astropy import constants as const
from scipy.optimize import root, brentq
from dataclasses import dataclass

from plotting import savefig, paper_plot
from cluster_functions import * 
from iqbal_agn_heating_functions import vol_heating_rate
from cluster_timescales import ClusterTimescales
from cluster_data import ClusterData



u.set_enabled_equivalencies(u.mass_energy())
with u.set_enabled_equivalencies(u.mass_energy()):
    adiabatic_idx = 5 / 3
    norm = (
        1 / 4
    )  # accretion rate normalization factor of order 1, norm(adiabatic_idx=5/3)=1/4
    mu = 1  # mean molecular weight of gas, 1 for proton gas (hydrogen)


@dataclass
class Cluster:
    radius: float = 1 * u.Mpc
    mass: float = 1.0e14 * u.Msun
    z: float = 0. # redshift
    vel_disp: float = None
    L500: float = None
    epsilon: float = 0.01
    fb: float = 0.1
    fdm: float = 0.9
    #m500: float = None
    v500: float = 0.0 * u.km / u.s
    m_chi: np.ndarray = np.logspace(-5, 3, num=100) * u.GeV
    m_b: float = const.m_p.to(u.GeV)  # baryon particle mass
    adiabatic_idx: float = 5.0 / 3.0
    norm: float = (
        1.0 / 4
    )  # accretion rate normalization factor of order 1, norm(adiabatic_idx=5/3)=1/4
    mu: float = 1.0  # mean molecular weight of gas, 1 for proton gas (hydrogen)
    bh_mass: float = None
    eff_agn_heating_rate: float = None


    def __post_init__(self):
        # General - read from data

        #self.data=ClusterData(self.radius, self.mass, self.z, self.L500)

        self.mass = self.mass.to(u.GeV)  # total mass
        self.volume = 4 / 3 * np.pi * self.radius**3  # cluster volume
        self.rho_tot = (self.mass / self.volume).to(u.GeV / u.cm**3)  # total density
        self.rho_b = self.rho_tot * self.fb  # baryon density
        self.rho_dm = self.rho_tot * self.fdm  # DM density
        if self.vel_disp is not None:
            if self.vel_disp.value:
                self.baryon_temp = temp_from_vdisp(self.vel_disp)  # baryon temperature
        elif self.L500.value:
            self.baryon_temp = temp_from_luminosity(self.L500)
        else:
            raise ValueError("Must provide a velocity dispersion or luminosity")

        # radiative cooling params
        #self.n_e = self.rho_b / self.m_b  # number density of electrons

        # setup AGN heating
        if self.bh_mass is None:
            self.bh_mass = self.get_bh_mass()

        if self.eff_agn_heating_rate is None:
            #TODO implement this better when I have an argument for Linj and rc
            #Linj=7*1e44*u.erg/u.s
            rc_factor=0.3
            self.eff_agn_heating_rate=self.get_effervescent_agn_heating_rate(rc_factor)

        self.timescales=ClusterTimescales(self.data)

        

    def agn_heating_rate(self):
        with u.set_enabled_equivalencies(u.mass_energy()):
            return (self.epsilon * self.accretion_rate()).to(
                u.GeV / u.s, equivalencies=u.temperature_energy()
            )

    def accretion_rate(self):
        with u.set_enabled_equivalencies(u.mass_energy()):
            leading_factors = self.norm * 4 * np.pi * const.c**-3
            gm2 = (const.G * self.bh_mass) ** 2
            frac = (self.mu * self.m_b) ** (5 / 2) / self.adiabatic_idx ** (3 / 2)
            return leading_factors * gm2 * frac * self.plasma_entropy() ** (-3 / 2)

    def get_bh_mass(self):
        slope = 1.39
        intercept = -9.56 #* u.Msun
        rhs=slope * np.log10(self.mass/u.Msun) + intercept
        return np.power(10, rhs) * u.Msun


    def plasma_entropy(self):
        baryon_number_density = (2 * self.n_e).to(u.m ** (-3))
        return (
            const.k_B * self.baryon_temp.to(u.K, equivalencies=u.temperature_energy())
        ).to(u.GeV) / baryon_number_density ** (self.adiabatic_idx - 1)

    def injected_energy(self, rc_factor): #rc_factor such that rcutoff=rc_factor*R500
        Mvir=(1.25*self.mass).to(u.Msun) # based on Iqbal assumption 
        if rc_factor==0.3:
            log_Linj = -0.96 + 1.73 * np.log10(Mvir/(1e14*u.Msun))
            Linj = np.power(10, log_Linj) * 1e45 * u.erg/u.s
        if rc_factor==0.1:
            log_Linj = -1.58 + 1.52 * np.log10(Mvir/(1e14*u.Msun))
            Linj = np.power(10, log_Linj) * 1e45 * u.erg/u.s
        return Linj
    
    def get_effervescent_agn_heating_rate(self, rc_factor):
        # TODO: implement method, make heating rate function take a class
        Linj=self.injected_energy(rc_factor)
        return (vol_heating_rate(self.radius, self.radius, self.mass, self.z, Linj, rc_factor*self.radius) * self.volume).to(u.erg/u.s)


    # def radiative_cooling_rate(self):
    #     prefactors=6.8*1e-42 *u.erg*u.cm**3
    #     Z=1
    #     T=self.baryon_temp.to(u.K, equivalencies=u.temperature_energy())
    #     T8=T/(1e8*u.K)
    #     C=(prefactors*Z**2*(self.n_e.to(u.cm**-3))**2)/(T8**(1/2))
    #     Eff_int = (C*T*const.k_B/const.h).to(u.GeV/(u.s*u.cm**3))
    #     return (self.volume*Eff_int).to(u.erg/u.s)

    def luminosity(self):
        T = temp_from_vdisp(self.v500).to(u.K, equivalencies=u.temperature_energy())
        b = -2.34 * 1e44 * u.erg / u.s
        m = (4.71 * 1e44 * u.erg / u.s) / u.K
        L = m * T + b
        return L.to(u.GeV / u.s)

    def virial_temperature(self, m_chi, f_chi=1, m_psi=0.1 * u.GeV):
        frac = f_chi / m_chi + (1 - f_chi) / m_psi
        M_kg = self.mass.to(u.kg, equivalencies=u.mass_energy())
        return (0.3 * const.G * M_kg / (self.radius * frac) * 1 / const.c**2).to(
            u.GeV
        )

    def sigma0(self, f_chi=1, m_psi=0.1 * u.GeV, n=0, Qh_dot = None):
        #total_heating_rate = Qh_dot or self.radiative_cooling_rate()
        #total_heating_rate=self.eff_agn_heating_rate - self.radiative_cooling_rate()
        total_heating_rate=self.eff_agn_heating_rate - self.timescales.radiative_cooling_rate()

        valid_m_chis = self.m_chi[
            np.where(
                self.virial_temperature(self.m_chi, f_chi=f_chi, m_psi=m_psi)
                < self.baryon_temp
            )
        ]

        dm_temp = self.virial_temperature(valid_m_chis, f_chi=f_chi, m_psi=m_psi)
        uth = np.sqrt(self.baryon_temp / self.m_b + dm_temp / valid_m_chis)
        rho_chi = self.rho_dm * f_chi

        numerator = total_heating_rate * (valid_m_chis + self.m_b) ** 2
        denominator = (
            3
            * (self.baryon_temp - dm_temp)
            * rho_chi
            * self.rho_b
            * self.volume
            * c(n)
            * uth ** (n + 1)
            * const.c.to(u.cm / u.s)
        )
        return (numerator / denominator).to(u.cm**2)

    # Plotting functions

    def plot_T_chi_vs_m_chi(
        self, f_chi=1, m_psi=0.1 * u.GeV
    ):  # produces T_chi vs m_chi plot given an f_chi and m_psi
        plt.loglog(
            self.m_chi,
            self.virial_temperature(self.m_chi, f_chi=f_chi, m_psi=m_psi),
            label=f"DM temp = virial temp, fx={f_chi}",
        )
        plt.xlabel(r"$m_{\chi} (GeV)$")
        plt.ylabel(r"$T_{\chi} (GeV)$")
        plt.legend(loc="upper left")

    def plot_sigma0_vs_m_chi(self, f_chi=None, m_psi=None, n=None, Qh_dot = None, region=False, save=False, minimal=False, **kwargs):
        if f_chi is None:
            f_chi = [1]
        if m_psi is None:
            m_psi = [0.1 * u.GeV]
        if n is None:
            n = [0]
        # plots sigma0 vs m_chi for all combinations of f_chi, m_psi, and n
        paper_plot()
        fig = plt.figure()
        params = [(f, m, i) for f in f_chi for m in m_psi for i in n]
        for f, m, i in params:
            sigma0 = self.sigma0(f_chi=f, m_psi=m, n=i, Qh_dot=Qh_dot)
            label = f"$f_{{\chi}}={f}$"
            if not minimal:
                    label = (
                    f"{label}, $m_{{\psi}}={m.to(u.GeV).value}$"+r"$\mathrm{(GeV)}$"
                    if f < 1
                    else label
                )
            plt.loglog(self.m_chi[: sigma0.shape[0]], sigma0, label=label)
            if region:
                plt.fill_between(
                    self.m_chi[: sigma0.shape[0]].value,
                    sigma0.value,
                    y2=1e-10,
                    alpha=0.3,
                )

        plt.xticks(**kwargs)
        plt.yticks(**kwargs)
        plt.ylim([10 ** (np.log10(np.nanmin(sigma0.value)) - 2 ), 10 ** (np.log10(np.nanmax(sigma0.value)))])

        plt.xlabel(r"$m_{\chi} \mathrm{(GeV)}$", **kwargs)
        plt.ylabel(r"$\sigma_0 (\mathrm{cm}^2)$", **kwargs)
        plt.legend(loc="upper left", **kwargs)
        if save:
            savefig(fig, "plots/sigma0_mchi.pdf")

