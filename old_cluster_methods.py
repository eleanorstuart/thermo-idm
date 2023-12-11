#paste in old cluster methods here to clean up cluster.py but save old code


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



