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

# old Cluster_werr methods:

def L500_analytic_eddington(self, mx, s0_factors, frac_L = 1):
    # returns the predicted L500 for a given mx and sigma0 based on the eddington luminosity equation and L500-Mbh relation
    # calculate predicted eddington luminosity
    eddington_luminosity = (self.integrated_dm_cooling_rate_fast(mx, s0_factors) + self.tot_rad)/(frac_L)
    
    # calculate Mbh 
    Mbh = eddington_luminosity/1e38
    
    # calculate logL500 from Mbh
    logL500 = (np.log10(Mbh) - a_u(10, 0.11, 0.11)) / a_u(0.38, 0.03, 0.03)
    
    # return L500
    return np.power(10, logL500) * 1e44



def L500_numerical_conservative(self, mx, s0, cluster_age=10): # cluster age in Gyr
    L_indep_factors_full = np.divide(self.L_indep_factors*s0,((mx + mp_GeV) ** 2))
    T_chi = np.array(self.Tx_by_mx*mx)

    cluster_age_s = cluster_age*u.Gyr.to(u.s) # convert Gyr to s 
    G_conv = (const.G/const.c**2).to(u.Mpc/u.Msun)

    def equation(logL): # L in 1e44 erg/s
        T500 = T500_from_L500(logL, bound=None)
        T500k = T500*GeV_to_K
        Tbs = np.multiply(self.Tbs_norm, T500k)
        #print(Tbs)

        cooling_rate=np.sqrt(T500k)*self.rad_factors


        pe = G_conv * (self.M500.value)**2*u.Msun**2/(self.R500.value*u.Mpc)

        #L = 10**logL
        xmin = 0.001
        xmax = 1
        xs = np.logspace(np.log10(xmin), np.log10(xmax), num=50)
        rs = xs*self.R500
        
        
        Tbs = np.array([T.value for T in np.asarray(self.T_g(rs, match_gaspari=True))]) # in GeV
        Tbs_norm = Tbs/Tbs[-1] # unitless
        Tbs_Ldep = Tbs_norm * T500_from_L500(logL)
        nbs = [n.value for n in np.asarray(self.n_e(rs))] # in 1/Mpc^3
        ke_b = (np.trapz(4*np.pi * rs**2 * 3./2* np.multiply(Tbs_Ldep,nbs),rs)).value*u.GeV
        
        avg_rho_x = self.M500.value/(4./3*np.pi*self.R500.value**3) # Msun/Mpc3
        #Tx_by_mx_float = [tm.value for tm in nfw.Tx_by_mx]
        ke_x = (np.trapz(4*np.pi*rs**2 * 3./2*avg_rho_x*self.Tx_by_mx, rs).value)*u.Msun
        #ke_x = 0 * u.erg
        E_tot =  (pe+ke_b+ke_x).to(u.erg, equivalencies=u.mass_energy()).value





        uth = np.sqrt(Tbs/mp_GeV + self.Tx_by_mx)
        integrand = (L_indep_factors_full*uth)*(Tbs-T_chi)
        dm_cooling_rate = np.trapz(integrand, self.rs_Mpc)

        Linj=Linj_from_L500(logL)
        heating_rate = Linj * self.agn_factors
        #print(heating_rate)
        #print('heating',heating_rate.value, 'cooling', cooling_rate.value, 'dm cooling', dm_cooling_rate.value)

        #total_energy = self.energies.abs_potential_energy + self.energies.thermal_energy_norm*T500 # check this 
        cluster_age_s = 10*u.Gyr.to(u.s) # convert Gyr to s
        return np.abs(E_tot/cluster_age_s) - np.abs(heating_rate - cooling_rate - dm_cooling_rate)

    logLs = np.linspace(-20, 20, 100)
    eqs = [equation(logL) for logL in logLs]
    return logLs, eqs





def L500_numerical_total_energy(self, mx, s0, cluster_age=10): # cluster age in Gyr
    L_indep_factors_full = np.divide(self.L_indep_factors*s0,((mx + mp_GeV) ** 2))
    T_chi = np.array(self.Tx_by_mx*mx)

    cluster_age_s = cluster_age*u.Gyr.to(u.s) # convert Gyr to s 

    G_conv = (const.G/const.c**2).to(u.Mpc/u.Msun)

    def equation(logL): # L in 1e44 erg/s
        T500 = T500_from_L500(logL, bound=None)
        T500k = T500*GeV_to_K
        Tbs = np.multiply(self.Tbs_norm, T500k)


        pe = G_conv * (self.M500.value)**2*u.Msun**2/(self.R500.value*u.Mpc)

        #L = 10**logL
        xmin = 0.001
        xmax = 1
        xs = np.logspace(np.log10(xmin), np.log10(xmax), num=50)
        rs = xs*self.R500
        
        
        Tbs = np.array([T.value for T in np.asarray(self.T_g(rs, match_gaspari=True))]) # in GeV
        Tbs_norm = Tbs/Tbs[-1] # unitless
        Tbs_Ldep = Tbs_norm * T500_from_L500(logL)
        nbs = [n.value for n in np.asarray(self.n_e(rs))] # in 1/Mpc^3
        ke_b = (np.trapz(4*np.pi * rs**2 * 3./2* np.multiply(Tbs_Ldep,nbs),rs)).value*u.GeV
        
        avg_rho_x = self.M500.value/(4./3*np.pi*self.R500.value**3) # Msun/Mpc3
        #Tx_by_mx_float = [tm.value for tm in nfw.Tx_by_mx]
        ke_x = (np.trapz(4*np.pi*rs**2 * 3./2*avg_rho_x*self.Tx_by_mx, rs).value)*u.Msun
        #ke_x = 0 * u.erg
        E_tot =  (pe+ke_b+ke_x).to(u.erg, equivalencies=u.mass_energy()).value


        uth = np.sqrt(Tbs/mp_GeV + self.Tx_by_mx)
        integrand = (L_indep_factors_full*uth)*(Tbs-T_chi)
        dm_cooling_rate = np.trapz(integrand, self.rs_Mpc)

        cluster_age_s = 10*u.Gyr.to(u.s) # convert Gyr to s
        return np.abs(E_tot/cluster_age_s) - np.abs(dm_cooling_rate)

    logLs = np.linspace(-20, 20, 100)
    eqs = [equation(logL) for logL in logLs]
    return logLs, eqs

def L500_analytic(self, mx, s0_factors, mult_agn = 1):
    # calculates predicted L500 for a given mx and sigma0 

    net_cool = (self.integrated_dm_cooling_rate_fast(mx, s0_factors) + self.tot_rad)
    agn_heating_without_Linj = mult_agn*self.agn_factors
    # calculate predicted Linj
    Linj = net_cool/agn_heating_without_Linj
    # convert Linj into L500
    Mbh = Linj/10**(34.5) # in Msun
    logL500 =  (np.log10(Mbh) - a_u(10, 0.11, 0.11)) / a_u(0.38, 0.3, 0.3)
    return np.power(10, logL500) * 1e44 





def log_lik1(p0, obs, nfws, variances, multagn=1, mx=0.01):
    log_s0 = p0[0]
    #log_mx = p0[1]

    # prior
    #if log_s0>0 or log_s0<-40:
    #    return -np.inf
    # DM particle mass is limited to being less than a proton mass
    #if log_mx<-4 or log_mx>log_mp: #10**log_mx>const.m_p.to(u.GeV, equivalencies=u.mass_energy()).value:
    #    return -np.inf

    if not (-40 <= log_s0 <= 0): #or not (-4 <= log_mx <= log_mp):
        return -np.inf

    # calculate model predicted L500
    #s0 = np.power(10., float(log_s0))
    #mx = np.power(10., float(log_mx))
    s0 = 10**log_s0
    mx = mx
    constant = (3*c0*s0/(((mx + mp_GeV) ** 2)) * GeV_to_erg)

    predicted_werr = np.array([nfw.L500_analytic(mx, constant, mult_agn=multagn) for nfw in nfws])
    variances_pred = np.array([max(p.plus, p.minus) for p in predicted_werr])
    predicted = np.array([p.value for p in predicted_werr])

    X2 = chi_squared(predicted, obs, variances, add_var=variances_pred)
    return -X2/2
