import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from astropy import constants as const
from scipy.optimize import root, brentq
from dataclasses import dataclass

from plotting import savefig, paper_plot
from cluster_functions import * 


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
    vel_disp: float = None
    L500: float = None
    epsilon: float = 0.01
    fb: float = 0.1
    fdm: float = 0.9
    m500: float = None
    v500: float = 0.0 * u.km / u.s
    m_chi: np.ndarray = np.logspace(-5, 3, num=100) * u.GeV
    m_b: float = const.m_p.to(u.GeV)  # baryon particle mass
    adiabatic_idx: float = 5.0 / 3.0
    norm: float = (
        1.0 / 4
    )  # accretion rate normalization factor of order 1, norm(adiabatic_idx=5/3)=1/4
    mu: float = 1.0  # mean molecular weight of gas, 1 for proton gas (hydrogen)
    bh_mass: float = None

    def __post_init__(self):
        # General - read from data
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
        self.n_e = self.rho_b / self.m_b  # number density of electrons

        # setup AGN heating
        if self.bh_mass is None:
            self.bh_mass = self.get_bh_mass()

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
        intercept = -9.56 * u.Msun
        return (slope * self.m500 + intercept).to(u.kg)

    def plasma_entropy(self):
        baryon_number_density = (2 * self.n_e).to(u.m ** (-3))
        return (
            const.k_B * self.baryon_temp.to(u.K, equivalencies=u.temperature_energy())
        ).to(u.GeV) / baryon_number_density ** (self.adiabatic_idx - 1)


    def radiative_cooling_rate(self):
        prefactors=6.8*1e-42 *u.erg*u.cm**3
        Z=1
        T=self.baryon_temp.to(u.K, equivalencies=u.temperature_energy())
        T8=T/(1e8*u.K)
        C=(prefactors*Z**2*(self.n_e.to(u.cm**-3))**2)/(T8**(1/2))
        Eff_int = (C*T*const.k_B/const.h).to(u.GeV/(u.s*u.cm**3))
        return (self.volume*Eff_int).to(u.erg/u.s)

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
        total_heating_rate = Qh_dot or self.radiative_cooling_rate()

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


    # model testing methods - Predicted value functions for MCMC
    def cooling_factors(self, n=0, f_chi=1):
        rho_chi=self.rho_dm*f_chi
        return 3 * rho_chi * self.rho_b * self.volume * c(n) * const.c

    def pred_T_b_small_m(self, p0, m_chi, n=0):
        # approximates T_b for small m_chi -> T_chi~0
        sigma0 = 10**p0 * u.cm**2
        V = self.volume.to(u.cm**3)
        x = (
            3
            * const.c
            * c(n)
            * V
            * self.rho_dm
            * self.rho_b
            * sigma0
            / (self.m_b + m_chi) ** 2
        ).to(1 / u.s)
        leading_factors = (self.norm * 4 * np.pi * const.c**-3).to(
            u.s**3 / u.cm**3
        )
        gm2 = ((const.G * self.bh_mass()) ** 2).to(u.cm**6 / u.s**4)
        frac = ((self.mu * self.m_b) ** (5 / 2) / self.adiabatic_idx ** (3 / 2)).to(
            u.GeV ** (5 / 2)
        )
        nb = (2 * self.n_e).to(u.cm ** (-3))  # baryon number density
        D = (
            self.epsilon
            * leading_factors
            * gm2
            * frac
            * (1 / nb ** (2 / 3)) ** (-3 / 2)
        )
        return (((D * np.sqrt(self.m_b)) / x) ** (1 / 3)).to(
            u.GeV, equivalencies=u.temperature_energy()
        )

    def pred_T_b(
        self, p0
    ):  # p0 is a vector with p0[0] = log(sigma0) and p0[1]=log(m_chi)
        x0 = 1e-5 * u.GeV  # starting estimate (could even do this using T_b_small)
       #solution = root(funr, x0, args=(self, p0)).x
       # return solution[0] * u.GeV



    def pred_T_b_1(
        self, s0, m_chi, n
    ):  # p0 is a vector with p0[0] = log(sigma0), m_chi is log(m_chi)
        #x0 = 1e-5 * u.GeV  
        p0 = [s0, m_chi]
        #solution = root(funr, x0, args=(self, p0), method='df-sane').x
        solution = brentq(funr_new, -500, 300, args=(self, p0, n))
        #print(solution)
        return 10**(solution)* u.GeV

    def pred_pref(self, p0):
        # predicts the radiation prefactors given vector p0=[log(sigma0), log(m_chi)]
        with u.set_enabled_equivalencies(u.temperature_energy()):
            sigma0 = 10 ** p0[0] * u.cm**2
            m_chi = 10 ** p0[1] * u.GeV
            T_chi = self.virial_temperature(m_chi)
            T_b = self.baryon_temp.to(u.K)
            Z=1

            u_th = (T_chi/m_chi + self.baryon_temp/self.m_b)**(1/2)
            dm_cooling_rate = -(self.cooling_factors() * (T_chi-self.baryon_temp) * sigma0*u_th/(m_chi+self.m_b)**2).to(u.erg/u.s)

            rad_factors = ((const.h*(T_b/(1e8*u.K))**(1/2))/(Z**2 * const.k_B * T_b * self.volume * self.n_e**2))
            return (rad_factors*dm_cooling_rate).to(u.erg*u.cm**3)

    def pred_Tb_pref(self, p0):
        #p0 is a choice for radiation prefactors
        x0 = 1e-5 * u.GeV  
        solution = root(funp, x0, args=(self, p0), method='df-sane').x
        print(p0,solution)
        return solution * u.GeV #solution[0] * u.GeV



