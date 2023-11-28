import numpy as np
import pandas as pd

from evaluate_SOC_models.data_manager import Data

from evaluate_SOC_models.data_sources import ISRaDData
from evaluate_SOC_models.path import TOPSOIL_MIN_DEPTH, TOPSOIL_MAX_DEPTH
from evaluate_SOC_models.path import SAVEOUTPUTPATH, SAVEALLDATAPATH


__all__ = ['SelectedISRaDData', 'AllObservedData', 'ObservedData']


class SelectedISRaDData(Data):

    datasets = ['data']

    def __init__(self, *, save_pkl=False, save_csv=False, save_xlsx=False):

        savedir = SAVEALLDATAPATH / 'data'
        name = 'selected_israd'
        description = ('Observed data of 14C and relative C'
            ' of density fractions, and bulk 14C and SOC from ISRaD.')

        super().__init__(savedir, name, description,
            save_pkl=save_pkl, save_csv=save_csv, save_xlsx=save_xlsx)


    def _process_data(self):
        israd = ISRaDData(TOPSOIL_MIN_DEPTH, TOPSOIL_MAX_DEPTH)
        #israd.purge_savedir('*topsoil_data.pkl.gz', ask=False, well_behaved=False)

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

        # # Get rid of permafrost, peatland, thermokarst, wetland soils
        # df = df[
        #     (df['pro_peatland'] != 'yes') &
        #     (df['pro_permafrost'] != 'yes') &
        #     (df['pro_thermokarst'] != 'yes') &
        #     (df['pro_land_cover'] != 'wetland')
        # ]
        #
        # # Select data from after 1990
        # df = df[df['date'] > '1990']

        return df.copy()



class AllObservedData(Data):

    datasets = ['data']

    def __init__(self, *, save_pkl=False, save_csv=False, save_xlsx=False):

        savedir = SAVEALLDATAPATH / 'data'
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
        return obs



class ObservedData(Data):

    datasets = ['data']

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
