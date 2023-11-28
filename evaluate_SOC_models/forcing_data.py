import numpy as np
import pandas as pd

from evaluate_SOC_models.data_manager import Data

from evaluate_SOC_models.observed_data import SelectedISRaDData
from evaluate_SOC_models.data_sources import \
    SoilGridsPointData, CESM2LEOutputData, ISIMIP2aNitrogenDepositionData

from evaluate_SOC_models.path import SAVEOUTPUTPATH, SAVEALLDATAPATH


__all__ = ['AllConstantForcingData', 'ForcingData']


class AllConstantForcingData(Data):

    datasets = ['data']

    def __init__(self, *, save_pkl=False, save_csv=False, save_xlsx=False):

        savedir = SAVEALLDATAPATH / 'data'
        name = 'all_constant_forcing'
        description = ('Constant forcing data from ISRaD and SoilGrids')

        super().__init__(savedir, name, description,
            save_pkl=save_pkl, save_csv=save_csv, save_xlsx=save_xlsx)


    def _process_data(self):
        forc = SelectedISRaDData().data[[
            'site_lat', 'site_long', 'site_elevation', 'pro_land_cover',
            'lyr_top', 'lyr_bot',
            'bd', 'sand', 'silt', 'clay', 'ph_h2o',
            'pro_soil_taxon', 'pro_lc_leaf_type'
        ]].copy()

        # Supplement missing measurements with data from SoilGrids
        for i in range(len(forc)):
            forci = forc.iloc[i]
            isnull = forci[['bd','sand','silt','clay','ph_h2o']].isnull()
            if not isnull.sum():
                continue
            lat, lon, top, bot = forci[['site_lat','site_long','lyr_top','lyr_bot']]
            sg = SoilGridsPointData(lat, lon, top, bot)
            for c, fill in zip(['bd','sand','silt','clay','ph_h2o'], isnull):
                if fill:
                    dataset = 'integrated_' + ('pH' if c=='ph_h2o' else c)
                    forc.iloc[i, forc.columns.get_loc(c)] = sg[dataset].item()

        return forc



class ForcingData(Data):

    datasets = ['constant', 'dynamic']

    def __init__(self, entry_name, site_name, pro_name, *,
            save_pkl=True, save_csv=False, save_xlsx=False):

        savedir = SAVEOUTPUTPATH / entry_name / site_name / pro_name / 'data'
        name = 'forcing'
        description = ('')

        super().__init__(savedir, name, description,
            save_pkl=save_pkl, save_csv=save_csv, save_xlsx=save_xlsx)

        self.entry_name = entry_name
        self.site_name = site_name
        self.pro_name = pro_name


    def _process_constant(self):
        """ Returns a pandas Series """
        israd_index = (self.entry_name, self.site_name, self.pro_name)
        forc = SelectedISRaDData().data.loc[israd_index, [
            'site_lat', 'site_long', 'site_elevation', 'pro_land_cover',
            'lyr_top', 'lyr_bot',
            'bd', 'sand', 'silt', 'clay', 'ph_h2o',
            'pro_soil_taxon', 'pro_lc_leaf_type'
        ]].copy()
        forc['fraction_NPP_bg_in_layer'] = self._get_fraction_NPP_bg_in_layer(
            forc.lyr_top, forc.lyr_bot, forc.pro_land_cover, forc.site_lat)

        # Fill missing values in ISRaD data with Soigrids data
        isnull = forc[['bd','sand','silt','clay','ph_h2o']].isnull()
        if isnull.sum():
            lat, lon, top, bot = forc[['site_lat','site_long','lyr_top','lyr_bot']]
            sg = SoilGridsPointData(lat, lon, top, bot)
            for c, fill in zip(['bd','sand','silt','clay','ph_h2o'], isnull):
                if fill:
                    dataset = 'integrated_' + ('pH' if c=='ph_h2o' else c)
                    forc[c] = sg[dataset].item()

        return forc


    def _process_dynamic(self):
        """ Returns a pandas DataFrame """

        lat, lon, top, bot, frac = self['constant'][[
            'site_lat', 'site_long', 'lyr_top', 'lyr_bot',
            'fraction_NPP_bg_in_layer'
        ]]
        cesm2le = CESM2LEOutputData(lat, lon, top, bot)
        isimip2 = ISIMIP2aNitrogenDepositionData(lat, lon)

        forc = pd.DataFrame({
            'NPP_ag': cesm2le['NPP_ag'], # gC/m2/s, above-ground NPP
            'NPP_bg_total': cesm2le['NPP_bg'], # gC/m2/s, total below-ground NPP
            'NPP_bg': frac * cesm2le['NPP_bg'], # gC/m2/s, below-ground NPP in depth layer
            'NPP_total': cesm2le['NPP_ag'] + cesm2le['NPP_bg'], # gC/m2/s
            'NPP': cesm2le['NPP_ag'] + frac * cesm2le['NPP_bg'], # gC/m2/s
            'GPP_total': cesm2le['GPP'], # gC/m2/s
            'Tsoil': cesm2le['Tsoil'], # Kelvin, temperature of soil
            'Wsoil': cesm2le['Wsoil'], # mm3/mm3, water content of soil
            'Delta14Clit': cesm2le['Delta14Clit'], # permille, D14C of litter
            'NHx': isimip2['NHx'], # gN/m2/month, NHx deposition
            'NOy': isimip2['NOy'], # gN/m2/month, NOy deposition
            'CNlit': cesm2le['Clit'] / cesm2le['Nlit'] # gC/gN of litter
        }).loc['1850':'2014']

        # Fill monthly NHx and NOy deposition over 1850-1860 period
        # with average monthly rates over 1860-1870 period
        month_num = lambda idx: idx.month # returns month number
        fill = forc.loc['1860':'1869', ['NHx','NOy']].groupby(month_num).mean()
        forc.loc['1850':'1859', 'NHx'] = np.tile(fill['NHx'].values, 10)
        forc.loc['1850':'1859', 'NOy'] = np.tile(fill['NOy'].values, 10)

        return forc


    @staticmethod
    def _get_fraction_NPP_bg_in_layer(top, bot, land_cover, latitude):
        """
        Find fraction of below-ground NPP (NPP_bg) which enters the soil
        between depths `top` and `bot` for a given `land_cover` and `latitude`.
        Uses the model from Xiao et al. (2023), DOI: 10.1111/geb.13705
        Coefficients b are from Table S4.

        Parameters
        ----------
        top : scalar > 0
            Top of soil layer
        bot : scalar > 0
            Bottom of soil layer such that bot > top
        land_cover : str
            ISRaD pro_land_cover, should be in ('bare', 'cultivated', 'forest',
            'rangeland/grassland', 'shrubland', 'urban', 'wetland', 'tundra')
        latitude : scalar
            Latitude in degrees
        """

        assert bot > top

        b_all_biomes = 16.4 # ±0.6 "All biomes"
        b_dict = { # map ISRaD's pro_land_cover to Table S4's biome type
            'bare': {
                'desert': 30.8, # ±6.8 "Deserts"
                'other': b_all_biomes
            },
            'cultivated': 17.5, # ±1.8 "Croplands"
            'forest': {
                'tropical': 17.5, # ±1.2 "Tropical/Subtropical forests"
                'temperate': 15.7, # ±1.1 "Temperate forests"
                'boreal': 11.5 # ±1.5 "Boreal forests"
            },
            'rangeland/grassland': {
                'tropical': 26.4, # ±5.7 "Tropical/Subtropical grasslands/savannas"
                'temperate': 19.2 # ±1.6 "Temperate grasslands"
            },
            'shrubland': 21.2, # ±1.8 "Mediterranean/Montane shrublands"
            'urban': b_all_biomes,
            'wetland': b_all_biomes,
            'tundra': 9.27 # ±1.5 "Tundra"
        }

        # Select correct b coefficient (this only works for our selected sites)
        if land_cover == 'forest':
            climate = 'tropical' if latitude < 24 else 'temperate'
            b = b_dict[land_cover][climate]
        elif land_cover == 'rangeland/grassland':
            climate = 'tropical' if latitude < 24 else 'temperate'
            b = b_dict[land_cover][climate]
        else:
            assert land_cover != 'bare'
            b = b_dict[land_cover]

        # NPP_bg depth distribution = a * exp(- depth / b)
        # Integrated from x0 to x1 =  a * b * ( exp(-x0/b) - exp(-x1/b) )
        NPP_bg_in_layer = np.exp(-top/b) - np.exp(-bot/b)
        NPP_bg_total = np.exp(-0/b) - np.exp(-np.inf/b) # = 1
        fraction_NPP_bg_in_layer = NPP_bg_in_layer / NPP_bg_total

        return fraction_NPP_bg_in_layer
