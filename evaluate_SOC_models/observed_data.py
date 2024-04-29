import numpy as np
import pandas as pd
import scipy
from numba import njit

from evaluate_SOC_models.data_manager import Data

from evaluate_SOC_models.data_sources import ISRaDData, Graven2017CompiledRecordsData
from evaluate_SOC_models.path import TOPSOIL_MIN_DEPTH, TOPSOIL_MAX_DEPTH
from evaluate_SOC_models.path import SAVEOUTPUTPATH, SAVEALLDATAPATH


__all__ = ['SelectedISRaDData', 'AllObservedData', 'ObservedData']


class SelectedISRaDData(Data):

    datasets = ['data']

    def __init__(self, *, save_pkl=True, save_csv=True, save_xlsx=False):

        savedir = SAVEALLDATAPATH
        name = 'selected_israd'
        description = ('Observed data of 14C and relative C'
            ' of density fractions, and bulk 14C and SOC from ISRaD.')

        super().__init__(savedir, name, description,
            save_pkl=save_pkl, save_csv=save_csv, save_xlsx=save_xlsx)


    def _process_data(self):
        israd = ISRaDData(TOPSOIL_MIN_DEPTH, TOPSOIL_MAX_DEPTH)

        df = israd['topsoil_data'].copy()

        # Drop profiles where there is no 14C data for the fractions
        df = df.dropna(how='all', subset=['HF_14c', 'fLF_14c', 'oLF_14c'])
        #df = df.dropna(how='all', subset=['HF_c_perc', 'fLF_c_perc', 'oLF_c_perc'])

        # Fill missing bulk 14C from integrated fraction data
        _14c = df[['HF_14c', 'fLF_14c', 'oLF_14c']].values
        _c_perc = df[['HF_c_perc', 'fLF_c_perc', 'oLF_c_perc']].values
        fill_bulk_14c = (_14c * _c_perc).sum(axis=1) / _c_perc.sum(axis=1)
        fill_bulk_14c = pd.Series(fill_bulk_14c, index=df.index)
        df['bulk_14c'] = df['bulk_14c'].fillna(fill_bulk_14c)

        # Drop profiles where there is no bulk 14C data
        df = df.dropna(subset='bulk_14c')

        # Compute LF data from fLF and oLF data
        df['LF_c_perc'] = df['fLF_c_perc'] + df['oLF_c_perc']
        df['LF_14c'] = (
            df['fLF_14c'] * df['fLF_c_perc'] + df['oLF_14c'] * df['oLF_c_perc']
        ) / df['LF_c_perc']

        # Add information from pro_info, site_info, entry_info
        pro_info = israd['pro_info'][[
            'pro_peatland',
            'pro_permafrost',
            'pro_thermokarst',
            'pro_usda_soil_order',
            'pro_soil_series',
            'pro_soil_taxon',
            'pro_land_cover',
            'pro_lc_phenology',
            'pro_lc_leaf_type',
        ]]
        df = df.join(pro_info)

        site_info = israd['site_info'][['site_lat', 'site_long', 'site_elevation']]
        df = df.join(site_info)

        entry_info = israd['entry_info'][[
            'doi', 'compilation_doi', 'bibliographical_reference'
        ]]
        df = df.join(entry_info)

        return df.copy()



class AllObservedData(Data):

    datasets = ['data']

    def __init__(self, *, save_pkl=True, save_csv=True, save_xlsx=False):

        savedir = SAVEALLDATAPATH
        name = 'all_observed'
        description = ('All selected observed data of 14C and relative C'
            ' of density fractions, and bulk 14C and SOC from ISRaD.')

        super().__init__(savedir, name, description,
            save_pkl=save_pkl, save_csv=save_csv, save_xlsx=save_xlsx)


    def _process_data(self):
        obs = SelectedISRaDData().data[
            ['date', 'soc', 'bulk_14c'] +
            [f+v for v in ('_14c','_c_perc') for f in ('HF','LF','fLF','oLF')]
        ].copy()

        # "Normalize" 14C data to year 2000
        # like in Shi et al. (2020) https://doi.org/10.1038/s41561-020-0596-z
        # and Heckman et al. (2021) https://doi.org/10.1111/gcb.16023

        zones = ISRaDData().pro_info['pro_atm_zone'].str[:-3]
        F_atmosphere = Graven2017CompiledRecordsData()['F14C']

        def error(k, obs_year, obs_Delta14C, years, F_input):
            return abs(_one_pool_steady_state_model(k, obs_year, years, F_input) - obs_Delta14C)

        def normalize_to_2000(row):
            row = row.copy()
            zone = zones.loc[row.name]
            F_input = F_atmosphere[zone].values
            years = F_atmosphere.index.year.values
            obs_year = row['date'].year
            for fraction in ('bulk', 'HF', 'LF', 'fLF', 'oLF'):
                obs_Delta14C = row[fraction+'_14c']
                result = scipy.optimize.minimize(error, x0=0.01, bounds=[(1e-6, 1)],
                    args=(obs_year, obs_Delta14C, years, F_input), method='Nelder-Mead')
                k, = result.x
                row[fraction+'_k'] = k
                row[fraction+'_success'] = result.success
                row[fraction+'_14c_2000'] = _one_pool_steady_state_model(k, 2000, years, F_input)
            return row

        obs = obs.apply(normalize_to_2000, axis=1)

        return obs


@njit
def _one_pool_steady_state_model(k, out_year, years, F_input):
    lambda14C = 1.2e-4 # radioactive decay rate of 14C (per annum)
    k_lambda14C = k + lambda14C # k is the carbon turnover rate
    F = k / k_lambda14C # steady-state solution when F_input = 1
    for year, Fin in zip(years, F_input):
        F += (k * Fin) - (k_lambda14C * F)
        if year == out_year:
            Delta14C = (F - 1) * 1000
            return Delta14C
    raise ValueError



class ObservedData(Data):

    datasets = ['data', 'data_normalized_to_2000']

    def __init__(self, entry_name, site_name, pro_name, *,
            save_pkl=False, save_csv=False, save_xlsx=False):

        savedir = SAVEOUTPUTPATH / entry_name / site_name / pro_name / 'data'
        name = 'observed'
        description = ('Observed data of 14C and relative C'
            ' of density fractions, and bulk 14C and SOC from ISRaD.'
            f' Data for {entry_name} / {site_name} / {pro_name}.')

        super().__init__(savedir, name, description,
            save_pkl=save_pkl, save_csv=save_csv, save_xlsx=save_xlsx)

        self.entry_name = entry_name
        self.site_name = site_name
        self.pro_name = pro_name


    def _process_data(self):
        """ Returns a pandas DataFrame with date as index """
        israd_index = (self.entry_name, self.site_name, self.pro_name)
        obs = AllObservedData().data.loc[[israd_index]].set_index('date')
        return obs.copy()
