import warnings
import re
import numpy as np
import pandas as pd

from data_manager.data import Data
from data_manager.file import Archive, FileFromURL
from data_manager.file import PandasExcelFile, FileFromArchive

from evaluate_SOC_models.path import DOWNLOADPATH, DATAPATH


__all__ = [
    'ISRaDArchive',
    'ISRaDExtraFile',
    'ISRaDTemplateInfoFile',
    'ISRaDExtraInfoFile',
    'ISRaDData'
]


class ISRaDArchive(FileFromURL, Archive):
    """
    Data archive from ISRaD v 2.5.5.2023-09-20 (commit 629b146)
    https://github.com/International-Soil-Radiocarbon-Database/ISRaD
    
    International Soil Radiocarbon Database, https://soilradiocarbon.org
    Paper: https://essd.copernicus.org/articles/12/61/2020/
    """

    def __init__(self):
        DL_link = (
            "https://github.com/International-Soil-Radiocarbon-Database/ISRaD/"
            "raw/629b14634d0ddaadf523c662f2cc45b4dad29ae5/ISRaD_data_files/"
            "database/ISRaD_database_files.zip"
        ) # ISRaD v 2.5.5.2023-09-20 (commit 629b146)
        filename = "ISRaD_database_files v 2.5.5.2023-09-20.zip"
        savedir = DOWNLOADPATH/'ISRaD/'
        super().__init__(savedir / filename, DL_link)



class ISRaDExtraFile(PandasExcelFile, FileFromArchive):

    def __init__(self, **kwargs):
        archive = ISRaDArchive()
        archived_filename = re.compile(r"ISRaD_extra_list_v.*\.xlsx") # regex
        filename = "ISRaD_extra.xlsx"
        savedir = DOWNLOADPATH/'ISRaD/'
        super().__init__(savedir / filename, archive, archived_filename, **kwargs)



class ISRaDTemplateInfoFile(PandasExcelFile, FileFromURL):

    def __init__(self, **kwargs):
        DL_link = (
            "https://github.com/International-Soil-Radiocarbon-Database/ISRaD/"
            "raw/master/Rpkg/inst/extdata/ISRaD_Template_Info.xlsx?raw=true"
        )
        filename = "ISRaD_Template_Info.xlsx"
        savedir = DOWNLOADPATH / 'ISRAD'
        super().__init__(savedir / filename, DL_link, **kwargs)

    def _read(self, *args, **kwargs):
        with warnings.catch_warnings():
            # Ignore warning from openpyxl
            message = "Unknown extension is not supported and will be removed"
            warnings.filterwarnings("ignore", message=message)
            return super()._read(*args, **kwargs)



class ISRaDExtraInfoFile(PandasExcelFile, FileFromURL):

    def __init__(self, **kwargs):
        DL_link = (
            "https://github.com/International-Soil-Radiocarbon-Database/ISRaD/"
            "raw/master/Rpkg/inst/extdata/ISRaD_Extra_Info.xlsx?raw=true"
        )
        filename = "ISRaD_Extra_Info.xlsx"
        savedir = DOWNLOADPATH / 'ISRAD'
        super().__init__(savedir / filename, DL_link, **kwargs)

    def _read(self, *args, **kwargs):
        with warnings.catch_warnings():
            # Ignore warning from openpyxl
            message = "Unknown extension is not supported and will be removed"
            warnings.filterwarnings("ignore", message=message)
            return super()._read(*args, **kwargs)



class ISRaDData(Data):
    " International Soil Radiocarbon Database, https://soilradiocarbon.org "


    datasets = ['variable_info', 'entry_info', 'site_info', 'pro_info',
        'layer_bulk_data', 'layer_density_fraction_data', 'topsoil_data']


    def __init__(self, topsoil_min_depth=10, topsoil_max_depth=20, *,
            save_pkl=True, save_csv=False, save_xlsx=False):

        topsoil_min_depth = int(np.round(topsoil_min_depth, 0))
        topsoil_max_depth = int(np.round(topsoil_max_depth, 0))

        assert topsoil_min_depth < topsoil_max_depth, f'{topsoil_min_depth} < {topsoil_max_depth}'

        savedir = DATAPATH / 'ISRaD'
        name = 'israd'
        description = ('International Soil Radiocarbon Database. '
            'The "topsoil_*" datasets are integrated over the top '
            f'{topsoil_min_depth} to {topsoil_max_depth}cm of the soil.')

        super().__init__(savedir, name, description,
            save_pkl=save_pkl, save_csv=save_csv, save_xlsx=save_xlsx)

        # Specify top and bottom boundaries of integrated data in file naming
        for d in ['topsoil_data']:
            depth = f'top_{topsoil_min_depth}_to_{topsoil_max_depth}cm_' + d
            name_depth = self.name + '_' + depth
            self._file_groups['pickle'][d].filename = name_depth + '.pkl.gz'
            self._file_groups['csv'][d].filename = name_depth + '.csv'
            self._file_groups['excel'][d].sheet_name = depth

        self.topsoil_min_depth = topsoil_min_depth
        self.topsoil_max_depth = topsoil_max_depth

        self.sourcefiles = {
            'data': ISRaDExtraFile(),
            'template_info': ISRaDTemplateInfoFile(),
            'extra_info': ISRaDExtraInfoFile()
        }


    def _process_variable_info(self):

        # Combine all excel sheets into one pandas DataFrame
        sheet_names = [
            'metadata', 'site', 'profile', 'flux', 'layer',
            'interstitial', 'fraction', 'incubation'
        ]
        info1 = self.sourcefiles['template_info'].read(sheet_name=sheet_names)
        info2 = self.sourcefiles['extra_info'].read(sheet_name=sheet_names)
        info = pd.concat([ # order is important for drop_duplicates later on
            i[s] for i in (info1, info2) for s in sheet_names
        ], ignore_index=True)

        # Clean up, get rid of duplicates, set Column_Name as index
        info['Units/Info'].fillna(info['Units/info'], inplace=True)
        info['Vocab'].fillna(info['Controlled_Vocab/Values'], inplace=True)
        info = info.drop(columns=[
            'Controlled_Vocab/Values', 'Extra_product',
            'Units/info', 'Required'
        ]).drop_duplicates(
            subset='Column_Name', keep='first', ignore_index=True
        ).set_index('Column_Name')

        # Change all column names to lower case to prevent further confusion
        info.columns = [column_name.lower() for column_name in info.columns]
        info.index.name = info.index.name.lower()

        return info


    def _process_entry_info(self):
        entry_info = self.sourcefiles['data'].read(
            sheet_name = 'metadata',
            # usecols = {'entry_name', 'doi', 'bibliographical_reference'}
        ).drop_duplicates('entry_name').set_index('entry_name').sort_index()
        # drop_duplicates is necessary because Gaudinski_2001 is listed twice
        #assert entry_info.index.is_unique
        return entry_info


    def _process_site_info(self):
        site_info = self.sourcefiles['data'].read(
            sheet_name = 'site',
            #usecols = {'entry_name', 'site_name', 'site_lat', 'site_long'}
        ).set_index(['entry_name', 'site_name']).sort_index()
        assert site_info.index.is_unique
        return site_info


    def _process_pro_info(self):
        pro_info = self.sourcefiles['data'].read(
            sheet_name = 'profile'
        ).set_index(['entry_name', 'site_name', 'pro_name']).sort_index()
        pro_info['pro_atm_zone'].replace({
            'NHc14':'NH', 'SHc14':'SH', 'Tropicsc14':'Tropics'
        }, inplace=True)
        assert pro_info.index.is_unique
        return pro_info


    def _process_layer_bulk_data(self):

        dtypes = {
            'entry_name': 'str',
            'site_name': 'str',
            'pro_name': 'str',
            'lyr_name': 'str',
            'lyr_obs_date_y': 'str',
            'lyr_obs_date_m': 'str',
            'lyr_obs_date_d': 'str',
            'lyr_top': 'float', # cm
            'lyr_bot': 'float', # cm
            'lyr_hzn': 'str',
            #'lyr_bd_samp': 'float', # g/cm3
            #'lyr_bd_samp_fill_extra': 'float', # g/cm3
            'lyr_bd_samp_filled': 'float', # g/cm3
            'lyr_sand_tot_psa': 'float', # percent surface area
            'lyr_silt_tot_psa': 'float', # percent surface area
            'lyr_clay_tot_psa': 'float', # percent surface area
            #'lyr_ph_cacl': 'float',
            'lyr_ph_h2o': 'float',
            #'lyr_ph': 'float', # method other than H2O or CaCl
            #'lyr_c_inorg': 'float', # percent weight
            #'lyr_c_org': 'float', # percent weight
            #'lyr_c_org_fill_extra': 'float', # percent weight
            'lyr_c_org_filled': 'float', # percent weight
            #'lyr_c_tot': 'float', # percent weight
            #'lyr_soc': 'float', # g/cm2
            #'lyr_soc_fill_extra': 'float', # g/cm2
            'lyr_soc_filled': 'float', # g/cm2
            'lyr_soc_sigma': 'float', # g/cm2
            'lyr_son': 'float', # g/cm2
            'lyr_c_to_n': 'float',
            'lyr_15n': 'float', # permille
            'lyr_13c': 'float', # permille
            'lyr_14c': 'float', # permille
            'lyr_14c_fill_extra': 'float', # permille
            'lyr_14c_sigma': 'float', # permille
            'lyr_14c_sd': 'float', # permille
        }

        lyr = self.sourcefiles['data'].read(
            sheet_name = 'layer',
            usecols = lambda x: x in dtypes,
            dtype = dtypes
        ).rename(columns={
            'lyr_bd_samp_filled': 'lyr_bd',
            'lyr_soc_filled': 'lyr_soc',
            'lyr_c_org_filled': 'lyr_c_org',
            'lyr_sand_tot_psa': 'lyr_sand',
            'lyr_silt_tot_psa': 'lyr_silt',
            'lyr_clay_tot_psa': 'lyr_clay'
        }).set_index(
            ['entry_name', 'site_name', 'pro_name', 'lyr_name']
        ).sort_index()

        lyr['lyr_obs_date'] = pd.to_datetime({
            'year': lyr['lyr_obs_date_y'],
            'month': lyr['lyr_obs_date_m'].fillna('7'),
            'day': lyr['lyr_obs_date_d'].fillna('1')
        })

        return lyr


    def _process_layer_density_fraction_data(self):

        dtypes = {
            'entry_name': 'str',
            'site_name': 'str',
            'pro_name': 'str',
            'lyr_name': 'str',
            'frc_name': 'str',
            'frc_aggregate_dis': 'str',
            'frc_scheme': 'str',
            #'frc_scheme_units': 'str', # "g cm^-3" for density fractionation
            'frc_lower': 'float',
            'frc_upper': 'float',
            #'frc_agent': 'str', # usually SPT for density fractionation
            'frc_property': 'str',
            'frc_14c': 'float', # permille
            'frc_c_perc': 'float', # percent of fraction C wrt bulk C
        }

        frc = self.sourcefiles['data'].read(
            sheet_name = 'fraction',
            usecols = lambda x: x in dtypes,
            dtype = dtypes
        ).set_index(
            ['entry_name', 'site_name', 'pro_name', 'lyr_name']
        ).sort_index()

        # Select density fractionation data
        frc = frc[frc['frc_scheme'] == 'density']#.copy()

        # Set shorter names for density fractions
        frc['frc_property'] = frc['frc_property'].replace({
            'heavy': 'HF', 'free light': 'fLF', 'occluded light': 'oLF'})

        # Select layers where 'oLF' appears at least once
        index = frc[frc['frc_property'] == 'oLF'].index.unique()
        frc = frc.loc[index]

        # Select layers where 'sonication' appears at least once
        index = frc[frc['frc_aggregate_dis'] == 'sonication'].index.unique()
        frc = frc.loc[index]

        # Fix something dumb
        frc['frc_upper'] = frc['frc_upper'].replace({0: np.inf})

        def merge_density_cutoffs(df):
            if len(df) == 1:
                return df.squeeze()

            # Take average of repeat measurements (not sure if there are any)
            df = df.groupby(level='cutoff_intervals').mean()
            if len(df) == 1:
                return df.squeeze()

            # Combine different cutoffs for the same fraction
            if df.index.is_overlapping: # too hard to implement, so return NaN
                return pd.Series(np.nan, columns=df.columns)
            df['frc_14c'] *= df['frc_c_perc']
            series = df.sum(axis=0, skipna=False)
            series['frc_14c'] /= series['frc_c_perc']

            return series

        # Combine sub-fractions, e.g. combine heavy fractions with density
        # cut-offs 1.8-2.0 g/cm3 and 2.0-inf g/cm3 into one single HF
        index_names = frc.index.names
        interval_index = pd.IntervalIndex.from_arrays(
            frc['frc_lower'], frc['frc_upper'], name='cutoff_intervals'
        )
        frc_14c_c_perc = frc.reset_index().set_index(interval_index).groupby(
            index_names+['frc_property'], as_index=False
        )[['frc_14c', 'frc_c_perc']].apply(
            merge_density_cutoffs
        ).pivot(
            values=['frc_14c', 'frc_c_perc'],
            columns=['frc_property'],
            index=index_names
        )
        frc_14c_c_perc.columns = [
            fracname + var[3:] for var,fracname in frc_14c_c_perc.columns
        ] # turn MultiIndex column ('fLF','frc_14c') into 'fLF_14c', etc.

        # Drop rows where all values are NaN
        frc_14c_c_perc.dropna(how='all', axis=0, inplace=True)

        # For Crow_2015, select lyr_name == 'surface2', which uses a method
        # based on Golchin et al. (1994)
        index = frc_14c_c_perc.loc[['Crow_2015']].index
        drop_index = [i for i in index if i[-1] != 'surface2']
        frc_14c_c_perc.drop(index=drop_index, inplace=True)

        return frc_14c_c_perc


    def _process_topsoil_data(self):

        lyr_bulk = self['layer_bulk_data'][[
            'lyr_obs_date',
            'lyr_top', # cm
            'lyr_bot', # cm
            'lyr_bd', # g/cm3
            'lyr_sand', # percent surface area
            'lyr_silt', # percent surface area
            'lyr_clay', # percent surface area
            'lyr_ph_h2o',
            'lyr_c_org', # percent weight
            'lyr_soc', # g/cm2
            'lyr_son', # g/cm2
            'lyr_15n', # permille
            'lyr_13c', # permille
            'lyr_14c', # permille
        ]]

        lyr_frac = self['layer_density_fraction_data'][[
            'HF_14c', 'fLF_14c', 'oLF_14c', # permille
            'HF_c_perc', 'fLF_c_perc', 'oLF_c_perc' # % weight of bulk carbon
        ]]

        lyr = pd.concat([lyr_bulk, lyr_frac], axis=1)

        min_depth = self.topsoil_min_depth
        max_depth = self.topsoil_max_depth

        # Select topsoil data and kick out data where top == bot
        topsoil = lyr[
            (lyr['lyr_top'] >= 0)
            & (lyr['lyr_bot'] <= max_depth)
            & (lyr['lyr_top'] < min_depth)
            & (lyr['lyr_top'] != lyr['lyr_bot']) # so many top == bot ...
        ].copy()

        # Prepare to integrate data over depth
        topsoil['lyr_height'] = topsoil['lyr_bot'] - topsoil['lyr_top']
        topsoil['lyr_bulk_mass'] = topsoil['lyr_bd'] * topsoil['lyr_height']
        for f in ['HF', 'fLF', 'oLF']:
            topsoil[f+'_c_abs'] = topsoil[f+'_c_perc'] * topsoil['lyr_soc']

        integrator_dict = {
            'lyr_bd': 'lyr_height',
            'lyr_sand': 'lyr_bulk_mass',
            'lyr_silt': 'lyr_bulk_mass',
            'lyr_clay': 'lyr_bulk_mass',
            'lyr_ph_h2o': 'lyr_bulk_mass',
            'lyr_c_org': 'lyr_bulk_mass',
            'lyr_15n': 'lyr_son',
            'lyr_13c': 'lyr_soc',
            'lyr_14c': 'lyr_soc',
            'HF_14c': 'HF_c_abs',
            'fLF_14c': 'fLF_c_abs',
            'oLF_14c': 'oLF_c_abs',
            'HF_c_perc': 'lyr_soc',
            'fLF_c_perc': 'lyr_soc',
            'oLF_c_perc': 'lyr_soc'
        }
        integrands = list(integrator_dict.keys())
        integrators = list(integrator_dict.values())

        def integrate(df):
            if len(df) == 1:
                return df.squeeze()

            # Take average over repeat measurements
            df = df.groupby(['lyr_top', 'lyr_bot'], as_index=False).mean()
            if len(df) == 1:
                return df.squeeze()

            df[integrands] *= df[integrators].values

            intervals = pd.IntervalIndex.from_arrays(
                df['lyr_top'], df['lyr_bot'], closed='neither'
            )
            sorted_intervals = intervals.sort_values()

            if not intervals.is_overlapping \
            and all(sorted_intervals[:-1].right == sorted_intervals[1:].left):
                # we have a contiguous non-overlapping integration domain
                series = df.sum(axis=0, skipna=False)
            else:
                series = pd.Series(0, index=df.columns, dtype='float')
                bounds = sorted(set(
                    [i.left for i in intervals] + [i.right for i in intervals]
                ))
                for lower, upper in zip(bounds[:-1], bounds[1:]):
                    interval = pd.Interval(lower, upper, closed='neither')
                    overlapping_index = intervals.overlaps(interval)
                    overlapping = df[overlapping_index]
                    if len(overlapping) == 0: # If domain is non-contiguous,
                        return pd.Series(np.nan, index=df.columns) # return NaN
                    overlapping_intervals = intervals[overlapping_index]
                    weights = (interval.length
                        / overlapping_intervals.length.values[:,None])
                    series += (overlapping * weights).mean(axis=0, skipna=True)

            series[integrands] /= series[integrators].values
            series['lyr_top'] = df['lyr_top'].min()
            series['lyr_bot'] = df['lyr_bot'].max()
            return series

        # Integrate over layers for each profile and for each sampling date
        topsoil = topsoil.groupby([ # 'lyr_name' is the index for the groups
            'entry_name', 'site_name', 'pro_name', 'lyr_obs_date'
        ]).apply(integrate).reset_index('lyr_obs_date')

        # Kick out the profiles that do not cover enough of the topsoil
        topsoil = topsoil[
            (topsoil['lyr_top'] == 0) & (topsoil['lyr_bot'] >= min_depth)
        ]

        topsoil = topsoil.rename(columns={
            'lyr_obs_date': 'date',
            'lyr_bd': 'bd',
            'lyr_sand': 'sand',
            'lyr_silt': 'silt',
            'lyr_clay': 'clay',
            'lyr_ph_h2o': 'ph_h2o',
            'lyr_c_org': 'c_org',
            'lyr_soc': 'soc',
            'lyr_son': 'son',
            'lyr_15n': 'bulk_15n',
            'lyr_13c': 'bulk_13c',
            'lyr_14c': 'bulk_14c'
        })[[
            'date',
            'lyr_top', # cm
            'lyr_bot', # cm
            'bd', # g/cm3
            'sand', # percent surface area
            'silt', # percent surface area
            'clay', # percent surface area
            'ph_h2o',
            'c_org', # percent weight
            'soc', # g/cm2
            'son', # g/cm2
            'bulk_15n', # permille
            'bulk_13c', # permille
            'bulk_14c', # permille
            'HF_14c', # permille
            'fLF_14c', # permille
            'oLF_14c', # permille
            'HF_c_perc', # percent weight of bulk carbon
            'fLF_c_perc', # percent weight of bulk carbon
            'oLF_c_perc' # percent weight of bulk carbon
        ]].copy()

        return topsoil
