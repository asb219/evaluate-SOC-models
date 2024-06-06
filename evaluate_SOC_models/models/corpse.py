"""
Radiocarbon implementation of the CORPSE model (Sulman et al., 2014).

Adapted and modified from CORPSE-fire-response (Sulman, 2024a),
and using some parameter values from CORPSE-N (Sulman, 2024b).

* Source code of CORPSE-fire-response:
    Sulman, B. (2024a). "bsulman/CORPSE-fire-response: v1.0 (v1.0)".
    Zenodo. https://doi.org/10.5281/zenodo.10624992

* Source code of CORPSE-N:
    Sulman, B. (2024b). "bsulman/CORPSE-N: v1.0 (v1.0)". Zenodo.
    https://doi.org/10.5281/zenodo.10624990

* Associated manuscript:
    Sulman, B. N., et al. (2014). "Microbe-driven turnover offsets mineral-
    mediated storage of soil carbon under elevated CO2." Nature Climate
    Change, 4(12), 1099–1102. https://doi.org/10.1038/nclimate2436


Original work Copyright (C) 2024  Benjamin Sulman  (subject to CC BY 4.0)

Modified work Copyright (C) 2024  Alexander S. Brunmayr  <asb219@ic.ac.uk>

This file is part of the ``evaluate_SOC_models`` python package, subject to
the GNU General Public License v3 (GPLv3). You should have received a copy
of GPLv3 along with this file. If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import pandas as pd
import scipy.optimize
from loguru import logger

from .base import ModelEvaluationData


__all__ = ['CORPSEData']


class CORPSEData(ModelEvaluationData):

    model_name = 'CORPSE'

    datasets = [
        'forcing', 'preindustrial_forcing', 'constant_forcing',
        'raw_output', 'output', 'predicted', 'observed', 'error'
    ]

    all_pools = [
        'MB', # live microbial biomass
        'Ufast', # unprotected simple C
        'Uslow', # unprotected chem. resistant C
        'Unecr', # unprotected microbial necromass
        'Pfast', # protected simple C
        'Pslow', # protected chem. resistant C
        'Pnecr', # protected microbial necromass
    ]
    fraction_to_pool_dict = {
        'LF': ['Ufast', 'Uslow', 'Unecr'], # 'MB'
        'HF': ['Pfast', 'Pslow', 'Pnecr'],
        'bulk': ['MB', 'Ufast', 'Uslow', 'Unecr', 'Pfast', 'Pslow', 'Pnecr']
    }

    def __init__(self, entry_name, site_name, pro_name,
            spinup=10000, spinup_from_steady_state=200, # years
            *, save_pkl=True, save_csv=False, save_xlsx=False, **kwargs):

        super().__init__(entry_name, site_name, pro_name, save_pkl=save_pkl,
            save_csv=save_csv, save_xlsx=save_xlsx, **kwargs)

        self.spinup = spinup
        self.spinup_from_steady_state = spinup_from_steady_state


    def _process_forcing(self):
        forcing = self._forcing['dynamic'][
            ['Tsoil', 'Wsoil', 'NPP_ag', 'NPP_bg', 'Delta14Clit']
        ].copy()
        forcing['NPP_ag'] *= 365*24*60*60 # gC/m2/s -> gC/m2/year
        forcing['NPP_bg'] *= 365*24*60*60 # gC/m2/s -> gC/m2/year
        forcing['Fin'] = forcing['Delta14Clit'] / 1000 + 1
        forcing = forcing.drop(columns=['Delta14Clit'])
        forcing = forcing.resample('D').ffill() # daily time steps
        return forcing


    def _process_raw_output(self):

        # Radioactive decay rate of 14C
        decay14C = np.log(2) / 5730 # per year

        # Time step size
        delta_t = 1./365 # time units are years but take daily time steps

        # MODEL PARAMETERS
        # All variable names like in CORPSE-fire-response
        # Parameter values from Whitman_sims.py in CORPSE-fire-response
        # See parameter descriptions in docstring of `get_internal_flux_matrix`
        # Arrays are ordered like ['fast', 'slow', 'necro']
        vmaxref = np.array([9.0, 0.25, 4.5]) # 1/year
        Ea = np.array([5e3, 30e3, 5e3]) # J/mol
        kC = 0.01
        gas_diffusion_exp = 0.6
        substrate_diffusion_exp = 1.5
        minMicrobeC = 1e-3 # = 0.1%
        Tmic = 0.25 # years
        et = 0.6
        eup = np.array([0.6, 0.05, 0.6])
        tProtected = 75.0 # years
        protection_rate = np.array([0.3, 0.001, 1.5]) # 1/year

        # Get internal flux matrix and influx vector for this profile
        internal_flux_matrix = self.get_internal_flux_matrix(
            vmaxref, Ea, kC, gas_diffusion_exp, substrate_diffusion_exp,
            minMicrobeC, Tmic, et, eup, tProtected, protection_rate
        )
        influx_vector = self.get_influx_vector()

        # Get forcing data
        forcing = self['forcing']
        preindustrial_forcing = self['preindustrial_forcing']

        # Forcing for steady state
        T = preindustrial_forcing['Tsoil'].mean()
        W = preindustrial_forcing['Wsoil'].mean()
        NPP_ag = preindustrial_forcing['NPP_ag'].mean()
        NPP_bg = preindustrial_forcing['NPP_bg'].mean()
        I = influx_vector(NPP_ag, NPP_bg)
        Fin = 1.0

        # First guess of steady state
        # Order of pools = MB, Ufast, Uslow, Unecr, Pfast, Pslow, Pnecr
        steady_state_C_guess = np.array([1]+[100]*6, dtype=np.float64) # gC/m2
        steady_state_F_guess = np.array([0.95]*7, dtype=np.float64)

        # Find steady-state C stocks
        def func(C):
            A = internal_flux_matrix(C, T, W)
            return A.sum(axis=1) + I
        steady_state_C, success_C = self._find_steady_state(
            func, steady_state_C_guess, bounds=(0, 1e7), name='C'
        )

        # Find steady-state fraction modern
        if success_C:
            A = internal_flux_matrix(steady_state_C, T, W)
            def func(F):
                return A.dot(F) + (I * Fin) - (decay14C * steady_state_C * F)
            steady_state_F, success_F = self._find_steady_state(
                func, steady_state_F_guess, bounds=(0.1, 1.1), name='F'
            )
        else:
            steady_state_F = None
            success_F = False

        # Spinup
        C = steady_state_C if success_C else steady_state_C_guess
        F = steady_state_F if success_F else steady_state_F_guess
        CF = C * F
        forc = preindustrial_forcing[
            ['Tsoil', 'Wsoil', 'NPP_ag', 'NPP_bg', 'Fin']].values
        if success_C and success_F:
            spinup_years = self.spinup_from_steady_state
        else:
            spinup_years = self.spinup
        for _ in range(spinup_years):
            for T, W, NPP_ag, NPP_bg, Fin in forc:
                A = internal_flux_matrix(C, T, W)
                I = influx_vector(NPP_ag, NPP_bg)
                C += (A.sum(axis=1) + I) * delta_t
                CF += (A.dot(F) + I*Fin - decay14C*CF) * delta_t
                F = CF / C

        # Real run
        save_C = np.empty((len(forcing), len(C)), dtype=np.float64)
        save_F = np.empty((len(forcing), len(F)), dtype=np.float64)
        save_C[0] = C # initial condition
        save_F[0] = F # initial condition
        forc = forcing[['Tsoil', 'Wsoil', 'NPP_ag', 'NPP_bg', 'Fin']].values
        for i, (T, W, NPP_ag, NPP_bg, Fin) in enumerate(forc[:-1]):
            A = internal_flux_matrix(C, T, W)
            I = influx_vector(NPP_ag, NPP_bg)
            C += (A.sum(axis=1) + I) * delta_t
            CF += (A.dot(F) + I*Fin - decay14C*CF) * delta_t
            F = CF / C
            save_C[i+1] = C
            save_F[i+1] = F

        pools = self.all_pools
        pools14 = [p + '_F14C' for p in pools]
        columns = pools + pools14
        times = forcing.index

        df = pd.DataFrame(
            np.concatenate([save_C, save_F], axis=1),
            index=times, columns=columns
        )

        return df


    def get_internal_flux_matrix(self,
            vmaxref, Ea, kC, gas_diffusion_exp, substrate_diffusion_exp,
            minMicrobeC, Tmic, et, eup, tProtected, protection_rate):
        """
        Almost all variable names are like in CORPSE-fire-response.
        Arrays of shape (3,) are ordered like ['fast', 'slow', 'necro'].
        Arrays of shape (7,) are ordered like
            ['MB', 'Ufast', 'Uslow', 'Unecr', 'Pfast', 'Pslow', 'Pnecr'].

        Parameters
        ----------
        vmaxref : numpy.ndarray of shape (3,)
            max decomposition rate (1/year) at 20°C for each
            unprotected pool
        Ea : numpy.ndarray of shape (3,)
            activation energy (J/mol) for decomposition of each
            unprotected pool
        kC : float
            Michaelis-Menten half-saturation constant
            (g microbial biomass / g substrate)
        gas_diffusion_exp : float
            determines suppression of decomposition at high soil moisture
        substrate_diffusion_exp : float
            controls suppression of decomp at low soil moisture
        minMicrobeC : float
            minimum allowed microbe carbon as a fraction of unprotected
            carbon
        Tmic : float
            microbial turnover time (years)
        et : float
            carbon efficiency of microbe turnover
        eup : numpy.ndarray of shape (3,)
            carbon use efficiency of microbial carbon uptake from each
            unprotected pool
        tProtected : float
            turnover time (years) of protected C
        protection_rate : numpy.ndarray of shape (3,)
            protected C formation rate (1/year) for each unprotected pool

        Returns
        -------
        internal_flux_matrix: function
            __Parameters__
            C : numpy.ndarray with shape (7,)
                carbon stocks (gC/m2) at current time step
            T : float
                temperature (Kelvin) at current time step
            W : float
                soil water content (m3/m3) at current time step
            __Returns__
            A : numpy.ndarray of shape (7,7)
                internal carbon flux matrix (gC/m2/year) at current
                time step
        """

        cforc = self['constant_forcing']

        theta_resp_max = substrate_diffusion_exp / (gas_diffusion_exp
            * (1 + substrate_diffusion_exp / gas_diffusion_exp))
        aerobic_max = (theta_resp_max**substrate_diffusion_exp
            * (1 - theta_resp_max)**gas_diffusion_exp)

        if cforc['pro_soil_taxon'] == 'Alfisol':
            slope = 0.5945
            intercept = 2.2788
        else:
            slope = 0.4833
            intercept = 2.3282

        if False: # from CORPSE_array.py in CORPSE-fire-response
            claypercent = cforc['clay'] # percent
            BD = cforc['bd'] # g/cm3
            claymod = 10**(slope*np.log10(claypercent) + intercept) * BD * 1e-6

        else: # from code/CORPSE_integrate.py in CORPSE-N
            claypercent_reference = 20
            claypercent = cforc['clay'] # percent
            Qref = 10**(slope * np.log10(claypercent_reference) + intercept)
            Q = 10**(slope * np.log10(claypercent) + intercept)
            claymod = Q / Qref

        W_sat = 0.54 # m3/m3, Table 2 in supplement of Sulman et al. (2014)
        W_sat = max(W_sat, self['forcing']['Wsoil'].max() * 1.01)

        Tref = 293.15 # reference temperature (K)
        Rugas = 8.314472 # ideal gas constant (J/K/mol)

        pools = self.all_pools
        Npools = len(pools)
        Uidx = [i for i,p in enumerate(pools) if p[0]=='U']
        Pidx = [i for i,p in enumerate(pools) if p[0]=='P']
        Midx = pools.index('MB')
        Nidx = pools.index('Unecr')

        def internal_flux_matrix(C, T, W):
            vmax = vmaxref * np.exp(-Ea/(Rugas*T) + Ea/(Rugas*Tref)) # eq 3
            theta = W / W_sat
            decomp = ( # maximum potential carbon decomposition rate
                vmax * theta**substrate_diffusion_exp
                * (1-theta)**gas_diffusion_exp
                * C[Uidx] * C[Midx] / (C[Midx] + kC * C[Uidx].sum())
                / aerobic_max
            )

            microbeTurnover = (C[Midx] - minMicrobeC * C[Uidx].sum()) / Tmic
            microbeTurnover = max(0, microbeTurnover)
            deadmic_C_production = microbeTurnover * et
            microbeGrowth = decomp * eup # eq 4

            protectedCturnover = C[Pidx] / tProtected # eq 6
            protectedCprod = claymod * protection_rate * C[Uidx] # eq 6

            A = np.zeros((Npools,Npools), dtype=np.float64)
            A[Nidx,Midx] = deadmic_C_production # microbe death
            A[Midx,Uidx] = microbeGrowth
            A[Pidx,Uidx] = protectedCprod
            A[Uidx,Pidx] = protectedCturnover
            #assert all(A.flatten() >= 0)
            A[Midx,Midx] = - microbeTurnover
            A[Uidx,Uidx] = - decomp - protectedCprod
            A[Pidx,Pidx] = - protectedCturnover
            #assert all(A.diagonal() <= 0)
            #assert all(A.sum(axis=0) <= 0)

            return A

        return internal_flux_matrix


    def get_influx_vector(self):
        """
        Returns
        -------
        influx_vector: function
            __Parameters__
            NPP_ag : float
                above-ground carbon inputs (gC/m2/year) at current time step
            NPP_bg : float
                below-ground carbon inputs (gC/m2/year) at current time step
            __Returns__
            I : numpy.ndarray of shape (7,)
                carbon influx vector (gC/m2/year) at current time step
        """

        # Values from Table 2 in supplement of Sulman et al. (2014)
        if self['constant_forcing']['pro_lc_leaf_type'] == 'needleleaf':
            fraction_leaf_input_to_slow = 0.95
        else:
            fraction_leaf_input_to_slow = 0.7
        fraction_root_input_to_slow = 0.85

        pools = self.all_pools
        Npools = len(pools)
        Ufast_idx = pools.index('Ufast')
        Uslow_idx = pools.index('Uslow')

        def influx_vector(NPP_ag, NPP_bg):
            I = np.zeros(Npools, dtype=np.float64)
            I[Ufast_idx] = (
                NPP_ag * (1 - fraction_leaf_input_to_slow)
                + NPP_bg * (1 - fraction_root_input_to_slow)
            )
            I[Uslow_idx] = (
                NPP_ag * fraction_leaf_input_to_slow
                + NPP_bg * fraction_root_input_to_slow
            )
            return I

        return influx_vector


    def _get_Cstocks_Delta14C(self, raw_output, fraction):
        cpools = self.fraction_to_pool_dict.get(fraction, [fraction])
        c14pools = [p + '_F14C' for p in cpools]
        Cstocks = raw_output[cpools].sum(axis=1).values
        C14stocks = (
            raw_output[c14pools].values * raw_output[cpools].values
        ).sum(axis=1)
        Delta14C = (C14stocks / Cstocks - 1) * 1000
        Cstocks *= 1e-4 # gC/m2 -> gC/cm2
        return Cstocks, Delta14C
