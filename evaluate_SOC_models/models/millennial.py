"""
Radiocarbon implementation of Millennial v2 (Abramoff et al., 2022).

Function `derivs_V2_MM` is modified from R-code in "R/models/derivs_V2_MM.R"
(Abramoff & Xu, 2022), and fitted parameter values are from file "Fortran/
MillennialV2/simulationv2/soilpara_in_fit.txt" (Abramoff & Xu, 2022).

* Source code of Millennial v2:
    Abramoff, R., & Xu, X. (2022). "rabramoff/Millennial: First release
    of Millennial V2 (Version v2)". Zenodo.
    https://doi.org/10.5281/zenodo.6353519

* Associated manuscript:
    Abramoff, R. Z., et al. (2022). "Improved global-scale predictions of
    soil carbon stocks with Millennial Version 2". Soil Biology and
    Biochemistry, 164, 108466. https://doi.org/10.1016/j.soilbio.2021.108466


Original work Copyright (C) 2022  Rose Abramoff & Xiaofeng Xu  (MIT license)

Modified work Copyright (C) 2024  Alexander S. Brunmayr  <asb219@ic.ac.uk>

This file is part of the ``evaluate_SOC_models`` python package, subject to
the GNU General Public License v3 (GPLv3). You should have received a copy
of GPLv3 along with this file. If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import pandas as pd
import scipy.optimize

from .base import ModelEvaluationData


__all__ = ['MillennialData']


def derivs_V2_MM(state, parameters, forcing, spinup=False):
    """
    Most of the code and comments in this method are taken from
    https://github.com/rabramoff/Millennial/blob/v2/R/models/derivs_V2_MM.R

    Original_docstring = '''
        Title: derivs_V2_MM.R
        Author: Rose Abramoff
        Date: Sep 11, 2021
        This function contains the system of equations for Millennial V2,
            as described in Abramoff et al. (2021).
        The equation numbers correspond to those in Abramoff et al. (2021),
            where the parameters and defined and theory described.
    '''

    Parameters
    ----------
    state: list, tuple, numpy.ndarray of floats of shape (12,)
        C and 14C stocks (g/m2) of the 5 pools and of the respired CO2:
        [POM,LMWC,AGG,MIC,MAOM,CO2,POM14,LMWC14,AGG14,MIC14,MAOM14,CO214]
    parameters: pandas.Series
        Contains parameter values and constant forcing data
    forcing: list, tuple, numpy.ndarray of floats of shape (4,)
        Contains dynamic forcing data at current time
            forc_st: soil temperature in Kelvin
            forc_sw: soil water content in mm3/mm3
            forc_npp: net primary productivity in gC/m2/day
            forc_Fin: fraction modern of carbon input
    spinup: bool
        If spinup is True, set dCO2 = 0 and dCO214 = 0 so that the respired
        CO2 and CO214 stay zero during spinup

    Returns
    -------
    dstate: np.ndarray of shape (12,)
        Changes in C and 14C stocks in gC/m2/day
    """

    POM,LMWC,AGG,MIC,MAOM,CO2,POM14,LMWC14,AGG14,MIC14,MAOM14,CO214 = state
    p = parameters
    forc_st, forc_sw, forc_npp, forc_Fin = forcing


    ### Soil type properties

    #Equation 10
    kaff_lm = np.exp(-p.param_p1 * p.param_pH - p.param_p2) * p.kaff_des

    #Equation 11
    param_qmax = p.param_bulkd * p.param_claysilt * p.param_pc


    ### Hydrological properties

    #Equation 4
    scalar_wd = np.sqrt(forc_sw / p.porosity)

    #Equation 15
    scalar_wb = np.exp(p['lambda'] * -p.matpot) * (p.kamin + (1 - p.kamin)
        * np.sqrt((p.porosity - forc_sw) / p.porosity)) * scalar_wd


    ### Decomposition

    gas_const = 8.31446

    #Equation 3
    vmax_pl = p.alpha_pl * np.exp(-p.eact_pl / (gas_const * forc_st))

    #Equation 2
    # POM -> LMWC
    if POM>0 and MIC>0:
        f_PO_LM = vmax_pl * scalar_wd * POM * MIC / (p.kaff_pl + MIC)
        f14_PO_LM = f_PO_LM * POM14/POM
    else:
        f_PO_LM = f14_PO_LM = 0

    #Equation 5
    # POM -> AGG
    if POM>0:
        f_PO_AG = p.rate_pa * scalar_wd * POM
        f14_PO_AG = f_PO_AG * POM14/POM
    else:
        f_PO_AG = f14_PO_AG = 0

    #Equation 6
    # AGG -> MAOM + POM
    if AGG>0:
        f_AG_break = p.rate_break * scalar_wd * AGG
        f14_AG_break = f_AG_break * AGG14/AGG
    else:
        f_AG_break = f14_AG_break = 0

    #Equation 8
    # LMWC -> out of system leaching
    if LMWC>0:
        f_LM_leach = p.rate_leach * scalar_wd * LMWC
        f14_LM_leach = f_LM_leach * LMWC14/LMWC
    else:
        f_LM_leach = f14_LM_leach = 0

    #Equation 9
    # LMWC -> MAOM
    if LMWC>0 and MAOM>0:
        f_LM_MA = scalar_wd * kaff_lm * LMWC * (1 - MAOM / param_qmax)
        f14_LM_MA = f_LM_MA * LMWC14/LMWC
    else:
        f_LM_MA = f14_LM_MA = 0

    #Equation 12
    # MAOM -> LMWC
    if MAOM>0:
        f_MA_LM = p.kaff_des * MAOM / param_qmax
        f14_MA_LM = f_MA_LM * MAOM14/MAOM
    else:
        f_MA_LM = f14_MA_LM = 0

    #Equation 14
    vmax_lb = p.alpha_lb * np.exp(-p.eact_lb / (gas_const * forc_st))

    #Equation 13
    # LMWC -> MIC
    if LMWC>0 and MIC>0:
        f_LM_MB = vmax_lb * scalar_wb * MIC * LMWC / (p.kaff_lb + LMWC)
        f14_LM_MB = f_LM_MB * LMWC14/LMWC
    else:
        f_LM_MB = f14_LM_MB = 0

    #Equation 16
    # MIC -> MAOM/LMWC
    if MIC>0:
        f_MB_turn = p.rate_bd * MIC * MIC
        f14_MB_turn = f_MB_turn * MIC14/MIC
    else:
        f_MB_turn = f14_MB_turn = 0

    #Equation 18
    # MAOM -> AGG
    if MAOM>0:
        f_MA_AG = p.rate_ma * scalar_wd * MAOM
        f14_MA_AG = f_MA_AG * MAOM14/MAOM
    else:
        f_MA_AG = f14_MA_AG = 0

    #Equation 22
    # microbial growth flux, but is not used in mass balance

    #Equation 21
    # MIC -> atmosphere
    if MIC>0 and LMWC>0:
        cue = p.cue_ref - p.cue_t * (forc_st - (p.tae_ref + 273.15))
        f_MB_atm = f_LM_MB * (1 - cue)
        f14_MB_atm = f_MB_atm * MIC14/MIC
    else:
        f_MB_atm = f14_MB_atm = 0


    ### Update state variables

    decay14C = np.log(2)/5730 / 365 # radioactive decay rate of 14C per day

    #Equation 1
    dPOM = (forc_npp * p.param_pi + f_AG_break * p.param_pa
        - f_PO_AG - f_PO_LM)
    dPOM14 = (forc_Fin * forc_npp * p.param_pi + f14_AG_break * p.param_pa
        - f14_PO_AG - f14_PO_LM) - decay14C * POM14

    #Equation 7
    dLMWC = (forc_npp * (1 - p.param_pi) - f_LM_leach
        + f_PO_LM - f_LM_MA - f_LM_MB
        + f_MB_turn * (1 - p.param_pb) + f_MA_LM)
    dLMWC14 = (forc_Fin * forc_npp * (1 - p.param_pi) - f14_LM_leach
        + f14_PO_LM - f14_LM_MA - f14_LM_MB
        + f14_MB_turn * (1 - p.param_pb) + f14_MA_LM) - decay14C * LMWC14

    #Equation 17
    dAGG = f_MA_AG + f_PO_AG - f_AG_break
    dAGG14 = f14_MA_AG + f14_PO_AG - f14_AG_break - decay14C * AGG14

    #Equation 20
    dMIC = f_LM_MB - f_MB_turn - f_MB_atm
    dMIC14 = f14_LM_MB - f14_MB_turn - f14_MB_atm - decay14C * MIC14

    #Equation 19
    dMAOM = (f_LM_MA - f_MA_LM + f_MB_turn * p.param_pb - f_MA_AG
        + f_AG_break * (1. - p.param_pa))
    dMAOM14 = (f14_LM_MA - f14_MA_LM + f14_MB_turn * p.param_pb - f14_MA_AG
        + f14_AG_break * (1. - p.param_pa)) - decay14C * MAOM14

    # track CO2 from respiration
    dCO2 = 0 if spinup else f_MB_atm
    dCO214 = 0 if spinup else f14_MB_atm

    dstate = np.array([
        dPOM, dLMWC, dAGG, dMIC, dMAOM, dCO2,
        dPOM14, dLMWC14, dAGG14, dMIC14, dMAOM14, dCO214
    ])

    return dstate



class MillennialData(ModelEvaluationData):

    model_name = 'Millennial'

    datasets = [
        'forcing', 'preindustrial_forcing', 'constant_forcing',
        'raw_output', 'output', 'predicted', 'observed', 'error'
    ]

    all_pools = [
        'POM', # particulate organic matter
        'LMWC', # low molecular weight carbon
        'AGG', # strongly bound micro-aggregates
        'MIC', # microbial biomass
        'MAOM' # mineral-associated organic matter
    ]
    fraction_to_pool_dict = {
        'LF': ['POM'], # 'MIC', 'LMWC'
        'HF': ['AGG', 'MAOM'], # 'MIC'
        'bulk': ['POM', 'LMWC', 'AGG', 'MIC', 'MAOM']
    }

    soil_para_in_fit = pd.Series({
        # From file Fortran/MillennialV2/simulationv2/soilpara_in_fit.txt
        # in Github repository rabramoff/Millennial
        'param_pi': 0.66,
        'param_pa': 0.33,
        'kaff_pl': 6443.,
        'alpha_pl': 1.8e12,
        'eact_pl': 63909.,
        'rate_pa': 0.018,
        'rate_break': 0.02,
        'rate_leach': 0.0015,
        'kaff_des': 1,
        'param_p1': 0.12,
        'param_p2': 0.216,
        'kaff_lb': 774.6,
        'alpha_lb': 2.3e12,
        'eact_lb': 57865.,
        'rate_bd': 0.0045,
        'rate_ma': 0.0048,
        'cue_ref': 0.19,
        'cue_t': 0.012,
        'tae_ref': 15.,
        'matpot': 15.,
        'lambda': 2.1e-4,
        'porosity': 0.6,
        'kamin': 0.2,
        'param_pb': 0.5
    })

    def __init__(self, entry_name, site_name, pro_name,
            spinup=10000, spinup_from_steady_state=200, # years
            *, save_pkl=True, save_csv=False, save_xlsx=False, **kwargs):

        super().__init__(entry_name, site_name, pro_name, save_pkl=save_pkl,
            save_csv=save_csv, save_xlsx=save_xlsx, **kwargs)

        self.spinup = spinup
        self.spinup_from_steady_state = spinup_from_steady_state


    def _process_forcing(self):
        forcing = self._forcing['dynamic'][
            ['Tsoil', 'Wsoil', 'NPP', 'Delta14Clit']
        ].copy()
        forcing['NPP'] *= 24*60*60 # gC/m2/s -> gC/m2/day
        forcing['Fin'] = forcing['Delta14Clit'] / 1000 + 1
        forcing = forcing.drop(columns=['Delta14Clit'])
        forcing = forcing.resample('D').ffill()
        return forcing


    def _process_raw_output(self):

        cforc = self['constant_forcing']
        forcing = self['forcing']
        preindustrial_forcing = self['preindustrial_forcing']

        parameters = self.soil_para_in_fit.copy()
        parameters['param_pH'] = cforc['ph_h2o']
        parameters['param_bulkd'] = cforc['bd'] * 1000 # g/cm3 -> kg/m3
        parameters['param_pc'] = 0.86 # from Millennial's model_tutorial.Rmd
        parameters['param_claysilt'] = cforc[['clay','silt']].sum() # percent
        parameters['porosity'] = max( # porosity > max soil water content
            parameters['porosity'], forcing['Wsoil'].max() * 1.01)
        # I don't know the units of param_pc but hopefully it involves
        # percent and a conversion of kg to g because it gets multiplied
        # by param_claysilt (in percent) and by param_bulkd (in kg/m3,
        # while all C stocks are in g/m2)

        # Forcing for steady state
        forc_st = preindustrial_forcing['Tsoil'].mean()
        forc_sw = preindustrial_forcing['Wsoil'].mean()
        forc_npp = preindustrial_forcing['NPP'].mean()
        forc_Fin = 1.0
        forc = (forc_st, forc_sw, forc_npp, forc_Fin)

        # First guess of steady state
        # Order of pools: POM, LMWC, AGG, MIC, MAOM
        steady_state_C_guess = np.array([100,10,100,10,100], np.float64) # gC/m2
        steady_state_F_guess = 0.95 + np.zeros(5, np.float64)

        # Find steady-state C stocks
        def func(Cstocks):
            state = np.append(Cstocks, [0]*7) # append CO2 and C14
            return derivs_V2_MM(state, parameters, forc, True)[:5]
        steady_state_C, success_C = self._find_steady_state(
            func, steady_state_C_guess, bounds=(0, 1e7), name='C'
        )

        # Find steady-state fraction modern
        if success_C:
            def func(F):
                C14stocks = F * steady_state_C # C14stocks without CO214
                state = np.concatenate([steady_state_C, [0], C14stocks, [0]])
                deriv = derivs_V2_MM(state, parameters, forc, True)[6:-1]
                return deriv / steady_state_C
            steady_state_F, success_F = self._find_steady_state(
                func, steady_state_F_guess, bounds=(0.1, 1.1), name='F'
            )
        else:
            steady_state_F = None
            success_F = False

        # Spinup
        C = steady_state_C if success_C else steady_state_C_guess
        F = steady_state_F if success_F else steady_state_F_guess
        state = np.concatenate([C, [0], C * F, [0]])
        forc = preindustrial_forcing[['Tsoil', 'Wsoil', 'NPP', 'Fin']].values
        if success_C and success_F:
            spinup_years = self.spinup_from_steady_state
        else:
            spinup_years = self.spinup
        for _ in range(spinup_years):
            for forc_i in forc:
                state += derivs_V2_MM(state, parameters, forc_i, True)

        # Real run
        save = np.empty((len(forcing), len(state)), dtype=np.float64)
        save[0] = state # initial state
        forc = forcing[['Tsoil', 'Wsoil', 'NPP', 'Fin']].values
        for i, forc_i in enumerate(forc[:-1]):
            state += derivs_V2_MM(state, parameters, forc_i)
            save[i+1] = state

        times = forcing.index
        columns = [
            'POM', 'LMWC', 'AGG', 'MIC', 'MAOM', 'CO2',
            'POM14', 'LMWC14', 'AGG14', 'MIC14', 'MAOM14', 'CO214'
        ]
        save_df = pd.DataFrame(save, index=times, columns=columns)

        return save_df


    def _get_Cstocks_Delta14C(self, raw_output, fraction):
        cpools = self.fraction_to_pool_dict.get(fraction, [fraction])
        c14pools = [p + '14' for p in cpools]
        Cstocks = raw_output[cpools].sum(axis=1).values
        C14stocks = raw_output[c14pools].sum(axis=1).values
        Delta14C = (C14stocks / Cstocks - 1) * 1000
        Cstocks *= 1e-4 # gC/m2 -> gC/cm2
        return Cstocks, Delta14C
